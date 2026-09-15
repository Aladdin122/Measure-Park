import time
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam3Processor, Sam3Model

# ── Singleton model (loaded once at worker startup, shared across all requests) ──
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading SAM 3 once on worker startup: {_DEVICE}")
_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(_DEVICE)
_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
print("[INFO] SAM 3 loaded successfully.")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _segment_car(image_pil: Image.Image, threshold: float = 0.30, mask_threshold: float = 0.40):
    """Run SAM 3 with prompt='car' and return list of (mask_np, area) sorted largest first."""
    inputs = _PROCESSOR(
        images=image_pil, text="car", return_tensors="pt"
    ).to(_DEVICE)
    with torch.inference_mode():
        outputs = _MODEL(**inputs)
    results = _PROCESSOR.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"]
    if len(masks) == 0:
        return []

    items = [
        (m.cpu().numpy().astype(np.uint8) * 255, torch.sum(m).item())
        for m in masks
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def _detect_wheel_sam3(park_cv: np.ndarray, roi_bounds: tuple):
    """
    Try SAM 3 wheel detection inside roi_bounds.
    Returns (gx, gy, conf) or None.
    """
    roi_xmin, roi_xmax, roi_ymin, roi_ymax = roi_bounds
    roi_cv = park_cv[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    if roi_cv.size == 0:
        return None

    try:
        roi_pil = Image.fromarray(cv2.cvtColor(roi_cv, cv2.COLOR_BGR2RGB))
        inputs = _PROCESSOR(
            images=roi_pil, text="car wheel", return_tensors="pt"
        ).to(_DEVICE)
        with torch.inference_mode():
            outputs = _MODEL(**inputs)
        results = _PROCESSOR.post_process_instance_segmentation(
            outputs,
            threshold=0.25,
            mask_threshold=0.30,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        if len(results["masks"]) == 0:
            return None

        best_candidate = None
        best_score = -1.0

        for mask_tensor in results["masks"]:
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)

            if area < 80 or len(largest_cnt) < 5:
                continue

            (ecx, ecy), (d1, d2), _ = cv2.fitEllipse(largest_cnt)
            gx = roi_xmin + int(round(ecx))
            gy = roi_ymin + int(round(ecy))

            if (
                gx < 0
                or gx >= park_cv.shape[1]
                or gy < 0
                or gy >= park_cv.shape[0]
            ):
                continue

            axis_ratio = min(d1, d2) / max(1e-5, max(d1, d2))
            if axis_ratio < 0.40:
                continue

            score = area * axis_ratio
            if score > best_score:
                best_score = score
                best_candidate = (gx, gy, 0.95)

        return best_candidate

    except Exception as exc:
        print(f"[WARN] SAM 3 wheel detection error: {exc}")
        return None


def _detect_wheel_hough(
    roi_gray: np.ndarray,
    expected_x: int,
    roi_xmin: int,
    roi_ymin: int,
    ch: int,
):
    """Fallback Hough circle wheel detection. Returns (gx, gy, conf)."""
    if roi_gray.size == 0:
        return expected_x, roi_ymin + roi_gray.shape[0] // 2, 0.20

    roi_blur = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    min_r = max(8, int(ch * 0.08))
    max_r = int(ch * 0.24)

    for p2 in [22, 18, 14, 10]:
        circles = cv2.HoughCircles(
            roi_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=12,
            param1=45,
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            candidates = []
            for local_x, local_y, _ in circles:
                gx = roi_xmin + local_x
                gy = roi_ymin + local_y
                if abs(gx - expected_x) <= int(ch * 0.35):
                    candidates.append({"x": gx, "y": gy, "err": abs(gx - expected_x)})

            if candidates:
                best = min(candidates, key=lambda c: c["err"])
                cluster = [
                    c for c in candidates if abs(c["x"] - best["x"]) <= int(ch * 0.06)
                ]
                return (
                    int(np.median([c["x"] for c in cluster])),
                    int(np.median([c["y"] for c in cluster])),
                    0.85,
                )

    return expected_x, roi_ymin + roi_gray.shape[0] // 2, 0.20


def _compute_similarity(
    tmpl_gray: np.ndarray,
    tmpl_hsv: np.ndarray,
    tmpl_mask: np.ndarray,
    crop_gray: np.ndarray,
    crop_hsv: np.ndarray,
    crop_mask: np.ndarray,
    sift,
    bf,
) -> float:
    """SIFT + colour histogram + shape similarity score for template matching."""
    if (
        tmpl_gray is None
        or crop_gray is None
        or tmpl_gray.size == 0
        or crop_gray.size == 0
    ):
        return 0.0

    th, tw = tmpl_gray.shape[:2]
    crop_gray_r = cv2.resize(crop_gray, (tw, th), interpolation=cv2.INTER_CUBIC)
    crop_hsv_r = cv2.resize(crop_hsv, (tw, th), interpolation=cv2.INTER_CUBIC)
    crop_mask_r = cv2.resize(crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

    kp_t, des_t = sift.detectAndCompute(tmpl_gray, None)
    kp_c, des_c = sift.detectAndCompute(crop_gray_r, None)

    good = 0
    if des_t is not None and des_c is not None and len(des_c) > 1:
        matches = bf.knnMatch(des_t, des_c, k=2)
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance:
                    good += 1

    hist_t = cv2.calcHist([tmpl_hsv], [0, 1], tmpl_mask, [50, 60], [0, 180, 0, 256])
    hist_c = cv2.calcHist([crop_hsv_r], [0, 1], crop_mask_r, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_t, hist_t, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_c, hist_c, 0, 1, cv2.NORM_MINMAX)
    color_corr = cv2.compareHist(hist_t, hist_c, cv2.HISTCMP_CORREL)

    shape_score = 1.0 / (
        1.0 + cv2.matchShapes(tmpl_mask, crop_mask_r, cv2.CONTOURS_MATCH_I1, 0.0)
    )

    ar_t = tw / max(1, th)
    ar_c = crop_gray.shape[1] / max(1, crop_gray.shape[0])
    aspect_score = min(ar_t, ar_c) / max(ar_t, ar_c)

    return (good * 3.0) + (color_corr * 80.0) + (shape_score * 40.0) + (aspect_score * 30.0)


# ── Public API ────────────────────────────────────────────────────────────────

def measure_parking_gap(
    ref_image_path: str,
    park_image_path: str,
    side: str,
    overhang_cm: float,
    output_path: str,
) -> dict:
    """
    Measure the bumper-to-obstacle gap in the park image.

    Parameters
    ----------
    ref_image_path  : path to the Stage 1 input image (clean car photo, used as
                      template to identify the user's car in the park scene).
    park_image_path : path to the parking scene photo.
    side            : "front" or "back" — which end of the car faces the gap.
    overhang_cm     : the overhang value in cm from Stage 1 output
                      (front overhang if side=="front", rear if side=="back").
    output_path     : where to write the annotated result image.

    Returns
    -------
    dict with keys: status, gap_cm, side, output_path  (or status + message on error).
    """
    t_start = time.time()
    side = side.strip().lower()

    if side not in ("front", "back"):
        return {"status": "error", "message": "side must be 'front' or 'back'"}
    if overhang_cm <= 0:
        return {"status": "error", "message": "overhang_cm must be a positive number"}

    print(f"[INFO] Running Stage 2 on device: {_DEVICE}")

    # ── 1. Load reference image and build template ────────────────────────────
    ref_cv = cv2.imread(ref_image_path)
    if ref_cv is None:
        return {"status": "error", "message": "Could not read reference image"}

    ref_pil = Image.open(ref_image_path).convert("RGB")
    ref_items = _segment_car(ref_pil, threshold=0.30, mask_threshold=0.40)
    if not ref_items:
        return {"status": "error", "message": "SAM 3 could not segment the car in the reference image"}

    ref_mask = ref_items[0][0]
    ref_gray = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY)
    ref_hsv  = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2HSV)

    rx, ry, rw, rh = cv2.boundingRect(cv2.findNonZero(ref_mask))
    tmpl_gray = ref_gray[ry:ry + rh, rx:rx + rw]
    tmpl_hsv  = ref_hsv[ry:ry + rh, rx:rx + rw]
    tmpl_mask = ref_mask[ry:ry + rh, rx:rx + rw]

    # ── 2. Load park image and segment cars ───────────────────────────────────
    park_cv = cv2.imread(park_image_path)
    if park_cv is None:
        return {"status": "error", "message": "Could not read park image"}

    park_h, park_w = park_cv.shape[:2]
    park_pil = Image.open(park_image_path).convert("RGB")

    park_items = _segment_car(park_pil, threshold=0.30, mask_threshold=0.40)
    if len(park_items) < 2:
        return {
            "status": "error",
            "message": "SAM 3 could not detect two cars in the park image",
        }

    # Take two largest masks
    mask_a, _ = park_items[0]
    mask_b, _ = park_items[1]

    coords_a = cv2.boundingRect(cv2.findNonZero(mask_a))
    coords_b = cv2.boundingRect(cv2.findNonZero(mask_b))

    # Sort into left / right by horizontal centre
    if (coords_a[0] + coords_a[2] // 2) < (coords_b[0] + coords_b[2] // 2):
        left_mask, left_coords = mask_a, coords_a
        right_mask, right_coords = mask_b, coords_b
    else:
        left_mask, left_coords = mask_b, coords_b
        right_mask, right_coords = mask_a, coords_a

    # ── 3. Identify which mask matches the reference car ─────────────────────
    gray_park = cv2.cvtColor(park_cv, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    lx, ly, lw, lh = left_coords
    rx2, ry2, rw2, rh2 = right_coords

    match_left = _compute_similarity(
        tmpl_gray, tmpl_hsv, tmpl_mask,
        gray_park[ly:ly + lh, lx:lx + lw],
        cv2.cvtColor(park_cv[ly:ly + lh, lx:lx + lw], cv2.COLOR_BGR2HSV),
        left_mask[ly:ly + lh, lx:lx + lw],
        sift, bf,
    )
    match_right = _compute_similarity(
        tmpl_gray, tmpl_hsv, tmpl_mask,
        gray_park[ry2:ry2 + rh2, rx2:rx2 + rw2],
        cv2.cvtColor(park_cv[ry2:ry2 + rh2, rx2:rx2 + rw2], cv2.COLOR_BGR2HSV),
        right_mask[ry2:ry2 + rh2, rx2:rx2 + rw2],
        sift, bf,
    )

    if match_left > match_right:
        target_mask, target_box = left_mask, left_coords
        other_mask,  other_box  = right_mask, right_coords
        gap_side = "right"
        target_pos = "left"
    else:
        target_mask, target_box = right_mask, right_coords
        other_mask,  other_box  = left_mask, left_coords
        gap_side = "left"
        target_pos = "right"

    print(f"[INFO] User car identified on the {target_pos.upper()} side.")

    # ── 4. Detect the relevant wheel center ──────────────────────────────────
    cx, cy, cw, ch = target_box
    ox, oy, ow, oh = other_box

    y_indices, _ = np.where(target_mask > 0)
    y_bottom = int(np.max(y_indices)) if len(y_indices) > 0 else cy + ch

    # The wheel we need is on the side facing the gap
    if gap_side == "right":
        expected_wheel_x = cx + cw - int(ch * 0.425)
    else:
        expected_wheel_x = cx + int(ch * 0.425)

    roi_bounds = (
        max(cx, expected_wheel_x - int(ch * 0.45)),
        min(cx + cw, expected_wheel_x + int(ch * 0.45)),
        max(cy, int(cy + ch * 0.48)),
        min(park_h, y_bottom + 10),
    )

    wheel_res = _detect_wheel_sam3(park_cv, roi_bounds)
    if wheel_res:
        active_wheel_x, active_wheel_y, wheel_conf = wheel_res
        print(f"[INFO] Wheel detected via SAM 3 (conf={wheel_conf:.2f})")
    else:
        active_wheel_x, active_wheel_y, wheel_conf = _detect_wheel_hough(
            gray_park[roi_bounds[2]:roi_bounds[3], roi_bounds[0]:roi_bounds[1]],
            expected_wheel_x,
            roi_bounds[0],
            roi_bounds[2],
            ch,
        )
        print(f"[INFO] Wheel detected via Hough fallback (conf={wheel_conf:.2f})")

    # ── 5. Calculate bumper positions and gap ─────────────────────────────────
    ref_pixels_x   = np.where(target_mask > 0)[1]
    other_pixels_x = np.where(other_mask > 0)[1]

    if gap_side == "left":
        ref_bumper_x      = int(np.min(ref_pixels_x))
        obstacle_bumper_x = int(np.max(other_pixels_x))
    else:
        ref_bumper_x      = int(np.max(ref_pixels_x))
        obstacle_bumper_x = int(np.min(other_pixels_x))

    overhang_px = abs(active_wheel_x - ref_bumper_x)
    if overhang_px == 0:
        return {"status": "error", "message": "Could not establish pixel scale (wheel aligned with bumper)"}

    px_per_cm  = overhang_px / overhang_cm
    gap_pixels = abs(ref_bumper_x - obstacle_bumper_x)
    gap_cm     = gap_pixels / px_per_cm

    print(f"[INFO] Side used: {side.upper()}  |  Overhang: {overhang_cm:.1f} cm  |  Gap: {gap_cm:.1f} cm")

    # ── 6. Visualise ─────────────────────────────────────────────────────────
    bg      = np.full_like(park_cv, 255)
    overlay = cv2.addWeighted(
        park_cv, 0.35,
        np.full_like(park_cv, (215, 120, 195)), 0.65, 0,
    )
    combined_mask = cv2.bitwise_or(target_mask, other_mask)
    smooth_m = np.repeat(
        (cv2.GaussianBlur(combined_mask, (5, 5), 0).astype(float) / 255.0)[:, :, np.newaxis],
        3, axis=2,
    )
    canvas = (overlay * smooth_m + bg * (1.0 - smooth_m)).astype(np.uint8)

    # Draw bounding boxes
    cv2.rectangle(canvas, (ox, oy - 15), (ox + ow, oy + oh + 15), (165, 80, 140), 2)
    cv2.rectangle(canvas, (cx, cy - 15), (cx + cw, cy + ch + 15), (80, 140, 165), 2)

    # Draw wheel centre
    cv2.circle(canvas, (active_wheel_x, active_wheel_y), 5, (0, 0, 255), -1)
    cv2.circle(canvas, (active_wheel_x, active_wheel_y), 7, (255, 255, 255), 2)

    # Draw gap measurement line
    mid_y = cy + ch // 2
    cv2.arrowedLine(canvas, (ref_bumper_x, mid_y), (obstacle_bumper_x, mid_y), (0, 200, 0), 2, tipLength=0.05)
    cv2.arrowedLine(canvas, (obstacle_bumper_x, mid_y), (ref_bumper_x, mid_y), (0, 200, 0), 2, tipLength=0.05)

    # Draw label
    label = f"Gap ({side}): {gap_cm:.1f} cm"
    (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cx_box = min(ref_bumper_x, obstacle_bumper_x) + gap_pixels // 2
    cv2.rectangle(
        canvas,
        (cx_box - lbl_w // 2 - 10, 35),
        (cx_box + lbl_w // 2 + 10, 35 + lbl_h + 14),
        (255, 255, 255), -1,
    )
    cv2.putText(
        canvas, label,
        (cx_box - lbl_w // 2, 35 + lbl_h + 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )

    cv2.imwrite(output_path, canvas)

    total_time = time.time() - t_start
    print(f"[SUCCESS] Stage 2 completed in {total_time:.2f}s — gap = {gap_cm:.1f} cm")

    return {
        "status": "success",
        "gap_cm": round(gap_cm, 2),
        "side": side,
        "wheel_detection_confidence": round(wheel_conf, 2),
        "output_path": output_path,
    }