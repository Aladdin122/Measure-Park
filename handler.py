import base64
import os
import tempfile
import uuid
from urllib.parse import urlparse

import requests
import runpod

from stage1 import process_vehicle_overhangs
from stage2 import measure_parking_gap


# ── Shared download helper ────────────────────────────────────────────────────

def _download_image(image_url: str, destination: str) -> None:
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("image_url must be an http or https URL")

    with requests.get(image_url, stream=True, timeout=60) as response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"URL did not return an image: {content_type}")

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Stage 1 handler ───────────────────────────────────────────────────────────

def _handle_stage1(job_input: dict, request_id: str) -> dict:
    image_url    = job_input.get("image_url")
    car_length_cm = job_input.get("car_length_cm")
    return_image  = job_input.get("return_image_base64", True)

    if not image_url:
        return {"status": "error", "message": "Missing required field: image_url"}
    if car_length_cm is None:
        return {"status": "error", "message": "Missing required field: car_length_cm"}

    try:
        car_length_cm = float(car_length_cm)
        if car_length_cm <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {"status": "error", "message": "car_length_cm must be a positive number"}

    with tempfile.TemporaryDirectory(prefix=f"stage1_{request_id}_") as workdir:
        input_path  = os.path.join(workdir, "input.jpg")
        output_path = os.path.join(workdir, "stage1_result.jpg")

        try:
            _download_image(image_url, input_path)

            result = process_vehicle_overhangs(
                image_path=input_path,
                car_length_cm=car_length_cm,
                output_path=output_path,
            )

            if not isinstance(result, dict):
                return {"status": "error", "message": "Stage 1 returned an unexpected result"}
            if result.get("status") != "success":
                return result

            result.pop("output_path", None)

            if return_image and os.path.exists(output_path):
                result["output_image_base64"] = _encode_image(output_path)

            return result

        except requests.RequestException as exc:
            return {"status": "error", "message": f"Failed to download image: {exc}"}
        except Exception as exc:
            print(f"[ERROR] Stage 1 handler failed: {exc}")
            return {"status": "error", "message": str(exc)}


# ── Stage 2 handler ───────────────────────────────────────────────────────────

def _handle_stage2(job_input: dict, request_id: str) -> dict:
    """
    Expected input fields:
      ref_image_url   – URL of the Stage 1 input image (clean car photo used as template).
      park_image_url  – URL of the parking scene photo.
      side            – "front" or "back".
      overhang_cm     – The overhang value in cm from Stage 1 output.
      return_image_base64 – (optional, default True) whether to include annotated image.
    """
    ref_image_url  = job_input.get("ref_image_url")
    park_image_url = job_input.get("park_image_url")
    side           = job_input.get("side")
    overhang_cm    = job_input.get("overhang_cm")
    return_image   = job_input.get("return_image_base64", True)

    if not ref_image_url:
        return {"status": "error", "message": "Missing required field: ref_image_url"}
    if not park_image_url:
        return {"status": "error", "message": "Missing required field: park_image_url"}
    if not side:
        return {"status": "error", "message": "Missing required field: side ('front' or 'back')"}
    if overhang_cm is None:
        return {"status": "error", "message": "Missing required field: overhang_cm"}

    try:
        overhang_cm = float(overhang_cm)
        if overhang_cm <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {"status": "error", "message": "overhang_cm must be a positive number"}

    with tempfile.TemporaryDirectory(prefix=f"stage2_{request_id}_") as workdir:
        ref_path   = os.path.join(workdir, "ref.jpg")
        park_path  = os.path.join(workdir, "park.jpg")
        output_path = os.path.join(workdir, "stage2_result.jpg")

        try:
            _download_image(ref_image_url,  ref_path)
            _download_image(park_image_url, park_path)

            result = measure_parking_gap(
                ref_image_path=ref_path,
                park_image_path=park_path,
                side=side,
                overhang_cm=overhang_cm,
                output_path=output_path,
            )

            if not isinstance(result, dict):
                return {"status": "error", "message": "Stage 2 returned an unexpected result"}
            if result.get("status") != "success":
                return result

            result.pop("output_path", None)

            if return_image and os.path.exists(output_path):
                result["output_image_base64"] = _encode_image(output_path)

            return result

        except requests.RequestException as exc:
            return {"status": "error", "message": f"Failed to download image: {exc}"}
        except Exception as exc:
            print(f"[ERROR] Stage 2 handler failed: {exc}")
            return {"status": "error", "message": str(exc)}


# ── Main handler (routes by stage field) ─────────────────────────────────────

def handler(job):
    job_input  = job.get("input") or {}
    request_id = job.get("id") or str(uuid.uuid4())

    stage = str(job_input.get("stage", "1")).strip()

    if stage == "1":
        return _handle_stage1(job_input, request_id)
    elif stage == "2":
        return _handle_stage2(job_input, request_id)
    else:
        return {"status": "error", "message": f"Unknown stage: '{stage}'. Use '1' or '2'."}


runpod.serverless.start({"handler": handler})
