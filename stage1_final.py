import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


# ============================================================
# LOAD SAM 3 ONCE WHEN THE RUNPOD WORKER STARTS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading SAM 3 once on worker startup: {DEVICE}")

MODEL = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
MODEL.eval()

PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")

print("[INFO] SAM 3 loaded successfully.")


# ============================================================
# WHEEL DETECTION
# SAM 3 -> CONTOUR -> FIT ELLIPSE
# ============================================================

def detect_wheels_sam3_fit_ellipse(
    img_cv,
    car_box,
    model,
    processor,
    device
):
    """
    Detect both wheel centers using SAM 3.

    Returns:
        ((left_x, left_y), (right_x, right_y))

    Or:
        None
    """

    cx, cy, cw, ch = car_box

    roi_ymin = cy + int(ch * 0.45)
    roi_ymax = min(img_cv.shape[0], cy + ch + 10)

    roi_xmin = max(0, cx - 10)
    roi_xmax = min(img_cv.shape[1], cx + cw + 10)

    roi_crop = img_cv[
        roi_ymin:roi_ymax,
        roi_xmin:roi_xmax
    ]

    if roi_crop.size == 0:
        return None

    try:

        roi_pil = Image.fromarray(
            cv2.cvtColor(
                roi_crop,
                cv2.COLOR_BGR2RGB
            )
        )

        inputs = processor(
            images=roi_pil,
            text="car wheel",
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():

            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.25,
            mask_threshold=0.30,
            target_sizes=inputs.get(
                "original_sizes"
            ).tolist()
        )[0]

        if len(results["masks"]) == 0:
            return None

        detected_centers = []

        for mask_tensor in results["masks"]:

            mask_np = (
                mask_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
                * 255
            )

            contours, _ = cv2.findContours(
                mask_np,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            largest_cnt = max(
                contours,
                key=cv2.contourArea
            )

            area = cv2.contourArea(
                largest_cnt
            )

            if area < 80:
                continue

            if len(largest_cnt) < 5:
                continue

            (
                (ellipse_cx, ellipse_cy),
                (d1, d2),
                _
            ) = cv2.fitEllipse(
                largest_cnt
            )

            axis_ratio = (
                min(d1, d2)
                /
                max(
                    1e-5,
                    max(d1, d2)
                )
            )

            if axis_ratio < 0.40:
                continue

            gx = (
                roi_xmin
                + int(
                    round(
                        ellipse_cx
                    )
                )
            )

            gy = (
                roi_ymin
                + int(
                    round(
                        ellipse_cy
                    )
                )
            )

            detected_centers.append(
                {
                    "x": gx,
                    "y": gy,
                    "area": area
                }
            )

        if not detected_centers:
            return None

        mid_x = (
            cx
            + cw / 2.0
        )

        left_candidates = [
            c
            for c in detected_centers
            if c["x"] < mid_x
        ]

        right_candidates = [
            c
            for c in detected_centers
            if c["x"] >= mid_x
        ]

        if (
            not left_candidates
            or not right_candidates
        ):
            return None

        best_left = max(
            left_candidates,
            key=lambda c: c["area"]
        )

        best_right = max(
            right_candidates,
            key=lambda c: c["area"]
        )

        return (
            (
                best_left["x"],
                best_left["y"]
            ),
            (
                best_right["x"],
                best_right["y"]
            )
        )

    except Exception as e:

        print(
            "[WARN] SAM 3 wheel "
            f"detection failed: {e}"
        )

        return None


# ============================================================
# STAGE 1 MAIN FUNCTION
# ============================================================

def process_vehicle_overhangs(
    image_path,
    car_length_cm,
    output_path="nextout.jpg",
    model=None,
    processor=None,
    device=None
):

    device = device or DEVICE
    model = MODEL if model is None else model
    processor = (
        PROCESSOR
        if processor is None
        else processor
    )

    print(
        f"[INFO] Running Stage 1 "
        f"on device: {device}"
    )

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        return {
            "status": "error",
            "message":
                "Input image does not exist."
        }

    try:

        car_length_cm = float(
            car_length_cm
        )

    except (TypeError, ValueError):

        return {
            "status": "error",
            "message":
                "car_length_cm must be numeric."
        }

    if car_length_cm <= 0:

        return {
            "status": "error",
            "message":
                "car_length_cm must be greater than zero."
        }

    img_cv = cv2.imread(
        image_path
    )

    if img_cv is None:

        return {
            "status": "error",
            "message":
                "OpenCV could not read the input image."
        }

    image_pil = Image.open(
        image_path
    ).convert("RGB")


    # ========================================================
    # 1. CAR SEGMENTATION
    # ========================================================

    print(
        "[INFO] Segmenting vehicle..."
    )

    inputs = processor(
        images=image_pil,
        text="car",
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():

        outputs = model(**inputs)

    results = (
        processor
        .post_process_instance_segmentation(
            outputs,
            threshold=0.30,
            mask_threshold=0.40,
            target_sizes=inputs.get(
                "original_sizes"
            ).tolist()
        )[0]
    )

    if len(results["masks"]) == 0:

        return {
            "status": "error",
            "message":
                "SAM 3 could not detect the car."
        }

    mask_areas = [
        torch.sum(mask).item()
        for mask
        in results["masks"]
    ]

    largest_mask_idx = int(
        np.argmax(
            mask_areas
        )
    )

    car_mask = (
        results["masks"][
            largest_mask_idx
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
        * 255
    )

    car_coords = cv2.findNonZero(
        car_mask
    )

    if car_coords is None:

        return {
            "status": "error",
            "message":
                "Invalid vehicle mask."
        }

    cx, cy, cw, ch = (
        cv2.boundingRect(
            car_coords
        )
    )

    if cw <= 0:

        return {
            "status": "error",
            "message":
                "Invalid detected car width."
        }

    car_xmin = cx
    car_xmax = cx + cw

    car_ymin = cy
    car_ymax = cy + ch

    pixels_per_cm = (
        cw
        /
        car_length_cm
    )


    # ========================================================
    # 2. FRONT / REAR DETECTION
    # ========================================================

    sample_offsets = [
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20
    ]

    votes_left_is_front = 0
    votes_right_is_front = 0

    for pct in sample_offsets:

        x_left = (
            cx
            + int(
                cw * pct
            )
        )

        x_right = (
            cx
            + cw
            - int(
                cw * pct
            )
        )

        x_left = int(
            np.clip(
                x_left,
                0,
                car_mask.shape[1] - 1
            )
        )

        x_right = int(
            np.clip(
                x_right,
                0,
                car_mask.shape[1] - 1
            )
        )

        left_pixels = np.where(
            car_mask[
                :,
                x_left
            ] > 0
        )[0]

        right_pixels = np.where(
            car_mask[
                :,
                x_right
            ] > 0
        )[0]

        if (
            len(left_pixels) > 0
            and
            len(right_pixels) > 0
        ):

            y_top_left = np.min(
                left_pixels
            )

            y_top_right = np.min(
                right_pixels
            )

            if (
                y_top_left
                >
                y_top_right
            ):

                votes_left_is_front += 1

            elif (
                y_top_right
                >
                y_top_left
            ):

                votes_right_is_front += 1


    if (
        votes_left_is_front
        >
        votes_right_is_front
    ):

        front_side = "left"

    elif (
        votes_right_is_front
        >
        votes_left_is_front
    ):

        front_side = "right"

    else:

        left_values = []
        right_values = []

        left_start = (
            cx
            + int(cw * 0.10)
        )

        left_end = (
            cx
            + int(cw * 0.20)
        )

        right_start = (
            cx
            + int(cw * 0.80)
        )

        right_end = (
            cx
            + int(cw * 0.90)
        )

        for x in range(
            left_start,
            left_end
        ):

            pixels = np.where(
                car_mask[:, x] > 0
            )[0]

            if len(pixels) > 0:

                left_values.append(
                    np.min(
                        pixels
                    )
                )

        for x in range(
            right_start,
            right_end
        ):

            pixels = np.where(
                car_mask[:, x] > 0
            )[0]

            if len(pixels) > 0:

                right_values.append(
                    np.min(
                        pixels
                    )
                )

        if (
            left_values
            and
            right_values
        ):

            front_side = (
                "left"
                if np.mean(
                    left_values
                )
                >
                np.mean(
                    right_values
                )
                else "right"
            )

        else:

            return {
                "status": "error",
                "message":
                    "Could not determine vehicle orientation."
            }


    print(
        "[INFO] Vehicle front side: "
        f"{front_side.upper()}"
    )


    # ========================================================
    # 3. WHEEL CENTER DETECTION
    # ========================================================

    print(
        "[INFO] Detecting wheel centers..."
    )

    car_box = (
        cx,
        cy,
        cw,
        ch
    )

    sam_wheels = (
        detect_wheels_sam3_fit_ellipse(
            img_cv,
            car_box,
            model,
            processor,
            device
        )
    )


    # --------------------------------------------------------
    # PRIMARY: SAM 3
    # --------------------------------------------------------

    if sam_wheels is not None:

        (
            (
                left_wheel_x,
                left_wheel_y
            ),
            (
                right_wheel_x,
                right_wheel_y
            )
        ) = sam_wheels

        wheel_method = (
            "sam3_fitEllipse"
        )


    # --------------------------------------------------------
    # FALLBACK: HOUGH CIRCLES
    # --------------------------------------------------------

    else:

        print(
            "[WARN] SAM wheel detection "
            "failed. Trying Hough Circles."
        )

        lower_half_y = int(
            car_ymin
            + ch * 0.55
        )

        gray = cv2.cvtColor(
            img_cv,
            cv2.COLOR_BGR2GRAY
        )

        wheel_zone = gray[
            lower_half_y:car_ymax,
            car_xmin:car_xmax
        ]

        wheel_zone_blur = (
            cv2.GaussianBlur(
                wheel_zone,
                (9, 9),
                0
            )
        )

        circles = cv2.HoughCircles(
            wheel_zone_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(
                1,
                int(ch * 0.20)
            ),
            param1=50,
            param2=25,
            minRadius=max(
                1,
                int(ch * 0.12)
            ),
            maxRadius=max(
                2,
                int(ch * 0.25)
            )
        )

        wheel_candidates = []

        if circles is not None:

            circles = np.float32(
                circles[0]
            )

            for circle in circles:

                local_x, local_y, radius = (
                    circle
                )

                global_x = (
                    car_xmin
                    + int(local_x)
                )

                global_y = (
                    lower_half_y
                    + int(local_y)
                )

                if (
                    car_xmin
                    <
                    global_x
                    <
                    car_xmax
                ):

                    wheel_candidates.append(
                        (
                            global_x,
                            global_y,
                            radius
                        )
                    )

        valid_pairs = []

        for i in range(
            len(
                wheel_candidates
            )
        ):

            for j in range(
                i + 1,
                len(
                    wheel_candidates
                )
            ):

                c1 = wheel_candidates[i]
                c2 = wheel_candidates[j]

                x1, y1, r1 = c1
                x2, y2, r2 = c2

                radius_similarity = (
                    min(r1, r2)
                    /
                    max(r1, r2)
                )

                same_ground_level = (
                    abs(y1 - y2)
                    <
                    ch * 0.12
                )

                wheelbase_ratio = (
                    abs(x2 - x1)
                    /
                    cw
                )

                valid_wheelbase = (
                    0.35
                    <
                    wheelbase_ratio
                    <
                    0.80
                )

                if (
                    radius_similarity > 0.70
                    and
                    same_ground_level
                    and
                    valid_wheelbase
                ):

                    score = (
                        radius_similarity
                        +
                        wheelbase_ratio
                        -
                        abs(y1 - y2) / ch
                    )

                    valid_pairs.append(
                        (
                            c1,
                            c2,
                            score
                        )
                    )

        if not valid_pairs:

            return {
                "status": "error",
                "message":
                    "Wheel centers could not be detected."
            }

        valid_pairs.sort(
            key=lambda item:
                item[2],
            reverse=True
        )

        c1, c2, _ = (
            valid_pairs[0]
        )

        if c1[0] < c2[0]:

            left_wheel_x = int(
                c1[0]
            )

            left_wheel_y = int(
                c1[1]
            )

            right_wheel_x = int(
                c2[0]
            )

            right_wheel_y = int(
                c2[1]
            )

        else:

            left_wheel_x = int(
                c2[0]
            )

            left_wheel_y = int(
                c2[1]
            )

            right_wheel_x = int(
                c1[0]
            )

            right_wheel_y = int(
                c1[1]
            )

        wheel_method = (
            "hough_fallback"
        )


    # ========================================================
    # 4. ASSIGN FRONT AND REAR WHEELS
    # ========================================================

    if front_side == "left":

        front_wheel_x = (
            left_wheel_x
        )

        rear_wheel_x = (
            right_wheel_x
        )

    else:

        front_wheel_x = (
            right_wheel_x
        )

        rear_wheel_x = (
            left_wheel_x
        )


    # ========================================================
    # 5. CALCULATE OVERHANG DIMENSIONS
    # ========================================================

    front_edge_x = (
        car_xmin
        if front_side == "left"
        else car_xmax
    )

    rear_edge_x = (
        car_xmax
        if front_side == "left"
        else car_xmin
    )

    front_overhang_cm = (
        abs(
            front_wheel_x
            -
            front_edge_x
        )
        /
        pixels_per_cm
    )

    rear_overhang_cm = (
        abs(
            rear_wheel_x
            -
            rear_edge_x
        )
        /
        pixels_per_cm
    )


    # ========================================================
    # 6. CREATE ANNOTATED OUTPUT IMAGE
    # ========================================================

    background_canvas = (
        np.full_like(
            img_cv,
            255
        )
    )

    purple_overlay = (
        np.full_like(
            img_cv,
            (
                215,
                120,
                195
            )
        )
    )

    purple_car = (
        cv2.addWeighted(
            img_cv,
            0.35,
            purple_overlay,
            0.65,
            0
        )
    )

    smooth_mask = (
        cv2.GaussianBlur(
            car_mask,
            (5, 5),
            0
        )
        .astype(float)
        /
        255.0
    )

    smooth_mask_3ch = (
        np.repeat(
            smooth_mask[
                :,
                :,
                np.newaxis
            ],
            3,
            axis=2
        )
    )

    canvas = (
        purple_car
        *
        smooth_mask_3ch
        +
        background_canvas
        *
        (
            1.0
            -
            smooth_mask_3ch
        )
    ).astype(np.uint8)


    top_padding = 100

    canvas = (
        cv2.copyMakeBorder(
            canvas,
            top_padding,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(
                255,
                255,
                255
            )
        )
    )

    padded_car_ymin = (
        car_ymin
        +
        top_padding
    )

    padded_car_ymax = (
        car_ymax
        +
        top_padding
    )

    left_wheel_y_padded = (
        left_wheel_y
        +
        top_padding
    )

    right_wheel_y_padded = (
        right_wheel_y
        +
        top_padding
    )


    orange_bgr = (
        32,
        120,
        245
    )

    green_bgr = (
        75,
        215,
        140
    )

    black_bgr = (
        0,
        0,
        0
    )


    line_thickness = 2
    font_scale = 0.50
    text_thickness = 2

    top_y = (
        padded_car_ymin
        -
        15
    )

    bottom_y = (
        padded_car_ymax
        +
        15
    )


    if front_side == "left":

        cv2.rectangle(
            canvas,
            (
                car_xmin,
                top_y
            ),
            (
                front_wheel_x,
                bottom_y
            ),
            orange_bgr,
            line_thickness
        )

        cv2.rectangle(
            canvas,
            (
                rear_wheel_x,
                top_y
            ),
            (
                car_xmax,
                bottom_y
            ),
            green_bgr,
            line_thickness
        )

    else:

        cv2.rectangle(
            canvas,
            (
                front_wheel_x,
                top_y
            ),
            (
                car_xmax,
                bottom_y
            ),
            orange_bgr,
            line_thickness
        )

        cv2.rectangle(
            canvas,
            (
                car_xmin,
                top_y
            ),
            (
                rear_wheel_x,
                bottom_y
            ),
            green_bgr,
            line_thickness
        )


    # Wheel centers

    cv2.circle(
        canvas,
        (
            int(
                left_wheel_x
            ),
            int(
                left_wheel_y_padded
            )
        ),
        5,
        (
            0,
            0,
            255
        ),
        -1
    )

    cv2.circle(
        canvas,
        (
            int(
                right_wheel_x
            ),
            int(
                right_wheel_y_padded
            )
        ),
        5,
        (
            0,
            0,
            255
        ),
        -1
    )


    # ========================================================
    # FULL CAR RULER
    # ========================================================

    ruler_y = (
        padded_car_ymin
        -
        75
    )

    cv2.line(
        canvas,
        (
            car_xmin,
            ruler_y
        ),
        (
            car_xmax,
            ruler_y
        ),
        black_bgr,
        2
    )

    cv2.line(
        canvas,
        (
            car_xmin,
            ruler_y - 20
        ),
        (
            car_xmin,
            ruler_y + 20
        ),
        black_bgr,
        2
    )

    cv2.line(
        canvas,
        (
            car_xmax,
            ruler_y - 20
        ),
        (
            car_xmax,
            ruler_y + 20
        ),
        black_bgr,
        2
    )


    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    # Full length text

    ruler_label = (
        f"{car_length_cm:.1f} cm"
    )

    (
        text_width,
        _
    ), _ = cv2.getTextSize(
        ruler_label,
        font,
        font_scale,
        text_thickness
    )

    mid_car_x = (
        car_xmin
        +
        cw // 2
    )

    cv2.putText(
        canvas,
        ruler_label,
        (
            mid_car_x
            -
            text_width // 2,
            ruler_y - 15
        ),
        font,
        font_scale,
        black_bgr,
        text_thickness,
        cv2.LINE_AA
    )


    # ========================================================
    # FRONT LABEL
    # ========================================================

    front_label = (
        f"{front_overhang_cm:.1f} cm"
    )

    (
        fw,
        fh
    ), _ = cv2.getTextSize(
        front_label,
        font,
        font_scale,
        text_thickness
    )

    if front_side == "left":

        front_box_width = (
            front_wheel_x
            -
            car_xmin
        )

        front_text_x = (
            car_xmin
            +
            front_box_width // 2
            -
            fw // 2
        )

    else:

        front_box_width = (
            car_xmax
            -
            front_wheel_x
        )

        front_text_x = (
            front_wheel_x
            +
            front_box_width // 2
            -
            fw // 2
        )


    cv2.rectangle(
        canvas,
        (
            front_text_x - 4,
            top_y - fh - 2
        ),
        (
            front_text_x + fw + 4,
            top_y + fh + 2
        ),
        (
            255,
            255,
            255
        ),
        -1
    )

    cv2.putText(
        canvas,
        front_label,
        (
            front_text_x,
            top_y + fh // 2
        ),
        font,
        font_scale,
        black_bgr,
        text_thickness,
        cv2.LINE_AA
    )


    # ========================================================
    # REAR LABEL
    # ========================================================

    rear_label = (
        f"{rear_overhang_cm:.1f} cm"
    )

    (
        rw,
        rh
    ), _ = cv2.getTextSize(
        rear_label,
        font,
        font_scale,
        text_thickness
    )

    if front_side == "left":

        rear_box_width = (
            car_xmax
            -
            rear_wheel_x
        )

        rear_text_x = (
            rear_wheel_x
            +
            rear_box_width // 2
            -
            rw // 2
        )

    else:

        rear_box_width = (
            rear_wheel_x
            -
            car_xmin
        )

        rear_text_x = (
            car_xmin
            +
            rear_box_width // 2
            -
            rw // 2
        )


    cv2.rectangle(
        canvas,
        (
            rear_text_x - 4,
            top_y - rh - 2
        ),
        (
            rear_text_x + rw + 4,
            top_y + rh + 2
        ),
        (
            255,
            255,
            255
        ),
        -1
    )

    cv2.putText(
        canvas,
        rear_label,
        (
            rear_text_x,
            top_y + rh // 2
        ),
        font,
        font_scale,
        black_bgr,
        text_thickness,
        cv2.LINE_AA
    )


    # ========================================================
    # 7. SAVE OUTPUT
    # ========================================================

    output_directory = (
        os.path.dirname(
            output_path
        )
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    saved = cv2.imwrite(
        output_path,
        canvas
    )

    if not saved:

        return {
            "status": "error",
            "message":
                "Failed to save output image."
        }


    # ========================================================
    # 8. RETURN RESULT TO HANDLER
    # ========================================================

    result = {
        "status": "success",

        "car_length_cm":
            float(
                car_length_cm
            ),

        "front_side":
            front_side,

        "front_overhang_cm":
            round(
                float(
                    front_overhang_cm
                ),
                2
            ),

        "rear_overhang_cm":
            round(
                float(
                    rear_overhang_cm
                ),
                2
            ),

        "wheel_detection_method":
            wheel_method,

        "left_wheel_center": {
            "x":
                int(
                    left_wheel_x
                ),

            "y":
                int(
                    left_wheel_y
                )
        },

        "right_wheel_center": {
            "x":
                int(
                    right_wheel_x
                ),

            "y":
                int(
                    right_wheel_y
                )
        },

        "output_path":
            output_path
    }

    print(
        "[SUCCESS] Stage 1 completed."
    )

    print(
        f"[RESULT] Front Overhang: "
        f"{front_overhang_cm:.2f} cm"
    )

    print(
        f"[RESULT] Rear Overhang: "
        f"{rear_overhang_cm:.2f} cm"
    )

    return result
