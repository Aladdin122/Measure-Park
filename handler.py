import base64
import os
import tempfile
import uuid
from urllib.parse import urlparse

import requests
import runpod

from stage1 import process_vehicle_overhangs


def _download_image(image_url: str, destination: str) -> None:
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("image_url must be an http or https URL")

    with requests.get(image_url, stream=True, timeout=60) as response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"image_url did not return an image: {content_type}")

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def handler(job):
    job_input = job.get("input") or {}

    image_url = job_input.get("image_url")
    car_length_cm = job_input.get("car_length_cm")
    return_image_base64 = job_input.get("return_image_base64", True)

    if not image_url:
        return {
            "status": "error",
            "message": "Missing required field: image_url",
        }

    if car_length_cm is None:
        return {
            "status": "error",
            "message": "Missing required field: car_length_cm",
        }

    try:
        car_length_cm = float(car_length_cm)
        if car_length_cm <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {
            "status": "error",
            "message": "car_length_cm must be a positive number",
        }

    request_id = job.get("id") or str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix=f"stage1_{request_id}_") as workdir:
        input_path = os.path.join(workdir, "input.jpg")
        output_path = os.path.join(workdir, "stage1_result.jpg")

        try:
            _download_image(image_url, input_path)

            result = process_vehicle_overhangs(
                image_path=input_path,
                car_length_cm=car_length_cm,
                output_path=output_path,
            )

            if not isinstance(result, dict):
                return {
                    "status": "error",
                    "message": "Stage 1 returned an unexpected result",
                }

            if result.get("status") != "success":
                return result

            # Local RunPod files disappear with the worker, so for the first API
            # version we return the generated image as Base64.
            # Later this can be replaced with an upload to S3/R2/Supabase Storage.
            result.pop("output_path", None)

            if return_image_base64:
                with open(output_path, "rb") as image_file:
                    result["output_image_base64"] = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

            return result

        except requests.RequestException as exc:
            return {
                "status": "error",
                "message": f"Failed to download image: {exc}",
            }
        except Exception as exc:
            print(f"[ERROR] Stage 1 handler failed: {exc}")
            return {
                "status": "error",
                "message": str(exc),
            }


runpod.serverless.start({"handler": handler})
