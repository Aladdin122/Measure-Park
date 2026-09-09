import os
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import Sam3Processor, Sam3Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading SAM 3 once on worker startup: {DEVICE}")
MODEL = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")

def detect_wheels_sam3_fit_ellipse(img_cv, car_box, model, processor, device):
    """
    Primary Wheel Engine: SAM 3 -> Contour Extraction -> cv2.fitEllipse
    Detects left and right wheels independently in the lower region of the car.
    Returns ((left_x, left_y), (right_x, right_y)) or None if detection fails.
    """
    cx, cy, cw, ch = car_box
    roi_ymin = cy + int(ch * 0.45)
    roi_ymax = min(img_cv.shape[0], cy + ch + 10)
    roi_xmin = max(0, cx - 10)
    roi_xmax = min(img_cv.shape[1], cx + cw + 10)

    roi_crop = img_cv[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    if roi_crop.size == 0:
        return None

    try:
        roi_pil = Image.fromarray(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB))
        inputs = processor(images=roi_pil, text="car wheel", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.25, mask_threshold=0.3, target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        if len(results["masks"]) == 0:
            return None

        detected_centers = []
        for mask_tensor in results["masks"]:
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)

            if area < 80 or len(largest_cnt) < 5:
                continue

            (ellipse_cx, ellipse_cy), (d1, d2), angle = cv2.fitEllipse(largest_cnt)

            axis_ratio = min(d1, d2) / max(1e-5, max(d1, d2))
            if axis_ratio < 0.40:
                continue

            gx = roi_xmin + int(round(ellipse_cx))
            gy = roi_ymin + int(round(ellipse_cy))

            detected_centers.append({"x": gx, "y": gy, "area": area})

        if not detected_centers:
            return None

        mid_x = cx + (cw / 2.0)
        left_cands = [c for c in detected_centers if c["x"] < mid_x]
        right_cands = [c for c in detected_centers if c["x"] >= mid_x]

        if not left_cands or not right_cands:
            return None

        best_left = max(left_cands, key=lambda c: c["area"])
        best_right = max(right_cands, key=lambda c: c["area"])

        return (best_left["x"], best_left["y"]), (best_right["x"], best_right["y"])

    except Exception as e:
        print(f"[WARN] SAM 3 wheel fitEllipse error: {e}")
        return None

def process_vehicle_overhangs(
    image_path,
    car_length_cm=450.0,
    output_path="nextout.jpg",
    model=None,
    processor=None,
    device=None,
):
    device = device or DEVICE
    model = MODEL if model is None else model
    processor = PROCESSOR if processor is None else processor

    print(f"[INFO] Running Stage 1 on device: {device}")

    if not os.path.exists(image_path):
        return {
            "status": "error",
            "message": f"Missing input image file: {image_path}",
        }

    car_length_cm = float(car_length_cm)
    if car_length_cm <= 0:
        return {
            "status": "error",
            "message": "car_length_cm must be greater than zero",
        }

    image_pil = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)

    if img_cv is None:
        return {
            "status": "error",
            "message": f"Could not read input image: {image_path}",
        }

    # ========================================================
    # STAGE 1: Segmentation & Core Mask Generation
    # ========================================================

    text_prompt = "car"
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

    print("[INFO] Segmenting vehicle profile...")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.3, mask_threshold=0.4, target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    if len(results["masks"]) == 0:
        return {
            "status": "error",
            "message": "SAM 3 failed to isolate the vehicle structure.",
        }

    mask_areas = [torch.sum(m).item() for m in results["masks"]]
    largest_mask_idx = np.argmax(mask_areas)
    car_mask = results["masks"][largest_mask_idx].cpu().numpy().astype(np.uint8) * 255

    car_coords = cv2.findNonZero(car_mask)
    cx, cy, cw, ch = cv2.boundingRect(car_coords)
    car_xmin, car_xmax = cx, cx + cw
    car_ymin, car_ymax = cy, cy + ch

    pixels_per_cm = cw / car_length_cm

    # ========================================================
    # STAGE 1.5: SILHOUETTE MULTI-SAMPLE MAJORITY VOTING
    # ========================================================
    sample_offsets = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    votes_left_is_front = 0
    votes_right_is_front = 0

    for pct in sample_offsets:
        x_left = cx + int(cw * pct)
        x_right = cx + cw - int(cw * pct)

        x_left = np.clip(x_left, 0, car_mask.shape[1] - 1)
        x_right = np.clip(x_right, 0, car_mask.shape[1] - 1)

        col_left_pixels = np.where(car_mask[:, x_left] > 0)[0]
        col_right_pixels = np.where(car_mask[:, x_right] > 0)[0]

        if len(col_left_pixels) > 0 and len(col_right_pixels) > 0:
            y_top_left = np.min(col_left_pixels)
            y_top_right = np.min(col_right_pixels)

            if y_top_left > y_top_right:
                votes_left_is_front += 1
            elif y_top_right > y_top_left:
                votes_right_is_front += 1

    if votes_left_is_front > votes_right_is_front:
        front_side = "left"
    elif votes_right_is_front > votes_left_is_front:
        front_side = "right"
    else:
        y_left_avg = np.mean([np.min(np.where(car_mask[:, x] > 0)[0]) for x in range(cx + int(cw*0.1), cx + int(cw*0.2)) if len(np.where(car_mask[:, x] > 0)[0]) > 0])
        y_right_avg = np.mean([np.min(np.where(car_mask[:, x] > 0)[0]) for x in range(cx + int(cw*0.8), cx + int(cw*0.9)) if len(np.where(car_mask[:, x] > 0)[0]) > 0])
        front_side = "left" if y_left_avg > y_right_avg else "right"

    print(f"[INFO] Multi-Sample Majority Voting: Left={votes_left_is_front}, Right={votes_right_is_front} -> FRONT is on the {front_side.upper()}")

    # ========================================================
    # STAGE 2: DUAL WHEEL CENTER DETECTION ENGINE
    # ========================================================
    print("[INFO] Attempting Primary SAM 3 + fitEllipse wheel detection...")
    car_box = (cx, cy, cw, ch)
    sam_wheels = detect_wheels_sam3_fit_ellipse(img_cv, car_box, model, processor, device)

    if sam_wheels is not None:
        (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y) = sam_wheels
        print(f"[INFO] SAM 3 + fitEllipse locked wheel centers: Left=({left_wheel_x}, {left_wheel_y}), Right=({right_wheel_x}, {right_wheel_y})")
    else:
        print("[WARN] SAM 3 wheel detection failed. Executing Hough wheel-pair fallback...")
        lower_half_y = int(car_ymin + (ch * 0.55))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        wheel_zone_blur = cv2.GaussianBlur(gray[lower_half_y:car_ymax, car_xmin:car_xmax], (9, 9), 0)

        circles = cv2.HoughCircles(
            wheel_zone_blur,
            cv2.HOUGH_GRADIENT, dp=1.0, minDist=int(ch * 0.20),
            param1=50, param2=25, minRadius=int(ch * 0.12), maxRadius=int(ch * 0.25)
        )

        wheel_candidates = []
        if circles is not None:
            circles = np.float32(circles[0])
            for circle in circles:
                local_x, local_y, radius = circle
                global_x = car_xmin + int(local_x)
                global_y = lower_half_y + int(local_y)
                if car_xmin < global_x < car_xmax:
                    wheel_candidates.append((global_x, global_y, radius))

        valid_pairs = []
        num_candidates = len(wheel_candidates)
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                c1, c2 = wheel_candidates[i], wheel_candidates[j]
                x1, y1, r1 = c1
                x2, y2, r2 = c2

                radius_similarity = min(r1, r2) / max(r1, r2)
                same_ground_level = abs(y1 - y2) < ch * 0.12
                valid_wheelbase = abs(x2 - x1) > cw * 0.35

                if radius_similarity > 0.70 and same_ground_level and valid_wheelbase:
                    score = radius_similarity + (abs(x2 - x1) / cw) - (abs(y1 - y2) / ch)
                    valid_pairs.append((c1, c2, score))

        if len(valid_pairs) > 0:
            valid_pairs.sort(key=lambda item: item[2], reverse=True)
            best_pair = valid_pairs[0]
            c1, c2 = best_pair[0], best_pair[1]

            if c1[0] < c2[0]:
                left_wheel_x, left_wheel_y = c1[0], c1[1]
                right_wheel_x, right_wheel_y = c2[0], c2[1]
            else:
                left_wheel_x, left_wheel_y = c2[0], c2[1]
                right_wheel_x, right_wheel_y = c1[0], c1[1]
            print(f"[INFO] Hough Circles locked wheel centers: Left=({left_wheel_x}, {left_wheel_y}), Right=({right_wheel_x}, {right_wheel_y})")
        else:
            return {
                "status": "error",
                "message": "Wheel centers could not be detected.",
            }

    if front_side == "left":
        front_wheel_x = left_wheel_x
        rear_wheel_x = right_wheel_x
    else:
        front_wheel_x = right_wheel_x
        rear_wheel_x = left_wheel_x

    # ========================================================
    # STAGE 3: MATTE COMPOSITING & TOP PADDING
    # ========================================================
    background_canvas = np.full_like(img_cv, 255)
    target_purple_bgr = (215, 120, 195)
    purple_overlay = np.full_like(img_cv, target_purple_bgr)

    purple_tinted_car = cv2.addWeighted(img_cv, 0.35, purple_overlay, 0.65, 0)

    smooth_mask = cv2.GaussianBlur(car_mask, (5, 5), 0).astype(float) / 255.0
    smooth_mask_3ch = np.repeat(smooth_mask[:, :, np.newaxis], 3, axis=2)
    canvas = (purple_tinted_car * smooth_mask_3ch + background_canvas * (1.0 - smooth_mask_3ch)).astype(np.uint8)

    line_thickness = 2
    font_scale = 0.50
    text_thickness = 2
    top_padding = 100

    canvas = cv2.copyMakeBorder(
        canvas,
        top_padding,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    car_ymin += top_padding
    car_ymax += top_padding
    left_wheel_y_padded = left_wheel_y + top_padding
    right_wheel_y_padded = right_wheel_y + top_padding

    # ========================================================
    # STAGE 4: Geometric Calculations & Dynamic UI Rendering
    # ========================================================
    orange_bgr = (32, 120, 245)
    green_bgr = (75, 215, 140)
    black_bgr = (0, 0, 0)

    top_y = car_ymin - 15
    bottom_y = car_ymax + 15

    if front_side == "left":
        cv2.rectangle(canvas, (car_xmin, top_y), (front_wheel_x, bottom_y), orange_bgr, line_thickness)
        cv2.rectangle(canvas, (rear_wheel_x, top_y), (car_xmax, bottom_y), green_bgr, line_thickness)
    else:
        cv2.rectangle(canvas, (front_wheel_x, top_y), (car_xmax, bottom_y), orange_bgr, line_thickness)
        cv2.rectangle(canvas, (car_xmin, top_y), (rear_wheel_x, bottom_y), green_bgr, line_thickness)

    cv2.circle(canvas, (int(left_wheel_x), int(left_wheel_y_padded)), 5, (0, 0, 255), -1)
    cv2.circle(canvas, (int(left_wheel_x), int(left_wheel_y_padded)), 7, (255, 255, 255), 2)

    cv2.circle(canvas, (int(right_wheel_x), int(right_wheel_y_padded)), 5, (0, 0, 255), -1)
    cv2.circle(canvas, (int(right_wheel_x), int(right_wheel_y_padded)), 7, (255, 255, 255), 2)

    ruler_y = int(car_ymin - 75)
    tick_length = 20
    cv2.line(canvas, (car_xmin, ruler_y), (car_xmax, ruler_y), black_bgr, line_thickness)
    cv2.line(canvas, (car_xmin, ruler_y - tick_length), (car_xmin, ruler_y + tick_length), black_bgr, line_thickness)
    cv2.line(canvas, (car_xmax, ruler_y - tick_length), (car_xmax, ruler_y + tick_length), black_bgr, line_thickness)

    front_overhang_cm = abs(front_wheel_x - (car_xmin if front_side == "left" else car_xmax)) / pixels_per_cm
    rear_overhang_cm = abs(rear_wheel_x - (car_xmax if front_side == "left" else car_xmin)) / pixels_per_cm

    font = cv2.FONT_HERSHEY_SIMPLEX

    ruler_label = f"{car_length_cm:.1f} cm"
    (tw, th), _ = cv2.getTextSize(ruler_label, font, font_scale, text_thickness)
    mid_car_x = car_xmin + (cw // 2)
    cv2.putText(canvas, ruler_label, (mid_car_x - (tw // 2), ruler_y - 15), font, font_scale, black_bgr, text_thickness, cv2.LINE_AA)

    front_label = f"{front_overhang_cm:.1f} cm"
    (fw, fh), _ = cv2.getTextSize(front_label, font, font_scale, text_thickness)
    if front_side == "left":
        front_box_w = front_wheel_x - car_xmin
        f_text_x = car_xmin + (front_box_w // 2) - (fw // 2)
    else:
        front_box_w = car_xmax - front_wheel_x
        f_text_x = front_wheel_x + (front_box_w // 2) - (fw // 2)
    f_text_y = top_y + (fh // 2)

    cv2.rectangle(canvas, (f_text_x - 4, top_y - fh - 2), (f_text_x + fw + 4, top_y + fh + 2), (255, 255, 255), -1)
    cv2.putText(canvas, front_label, (f_text_x, f_text_y), font, font_scale, black_bgr, text_thickness, lineType=cv2.LINE_AA)

    rear_label = f"{rear_overhang_cm:.1f} cm"
    (rw, rh), _ = cv2.getTextSize(rear_label, font, font_scale, text_thickness)
    if front_side == "left":
        rear_box_w = car_xmax - rear_wheel_x
        r_text_x = rear_wheel_x + (rear_box_w // 2) - (rw // 2)
    else:
        rear_box_w = rear_wheel_x - car_xmin
        r_text_x = car_xmin + (rear_box_w // 2) - (rw // 2)
    r_text_y = top_y + (rh // 2)

    cv2.rectangle(canvas, (r_text_x - 4, top_y - rh - 2), (r_text_x + rw + 4, top_y + rh + 2), (255, 255, 255), -1)
    cv2.putText(canvas, rear_label, (r_text_x, r_text_y), font, font_scale, black_bgr, text_thickness, lineType=cv2.LINE_AA)

    # Saves directly to your Colab disk
    cv2.imwrite(output_path, canvas)
    print(f"[SUCCESS] High-accuracy unified pipeline completed. Output saved: {output_path}")

    # Renders inside the Colab output cell
    cv2_imshow(canvas)

if __name__ == "__main__":
    process_vehicle_overhangs(image_path="chev.jpg", car_length_cm=450.0, output_path="nextout.jpg")