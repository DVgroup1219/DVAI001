"""Face size, position, and orientation estimators."""

from __future__ import annotations

from src.modules.face_detection.detectors.base import RawFaceDetection
from src.modules.face_detection.models import BoundingBox


def estimate_face_size(bbox: BoundingBox, image_width: int, image_height: int) -> str:
    """
    Classify face size relative to image area.

    small  < 5%
    medium 5–15%
    large  > 15%
    """
    image_area = max(1, image_width * image_height)
    ratio = bbox.area / image_area * 100
    if ratio < 5.0:
        return "small"
    if ratio < 15.0:
        return "medium"
    return "large"


def estimate_position(bbox: BoundingBox, image_width: int, image_height: int) -> str:
    """Estimate object position using a 3x3 grid."""
    cx, cy = bbox.center
    col = "left" if cx < image_width / 3 else "right" if cx > 2 * image_width / 3 else "center"
    row = "top" if cy < image_height / 3 else "bottom" if cy > 2 * image_height / 3 else "middle"

    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def estimate_orientation(face: RawFaceDetection) -> str:
    """
    Estimate face orientation from MediaPipe keypoints.

    Returns front, left, right, or unknown.
    """
    keypoints = face.keypoints
    if not keypoints:
        aspect = face.bbox.width / max(1, face.bbox.height)
        if aspect < 0.75:
            return "profile"
        return "front"

    right_eye = keypoints.get("right_eye")
    left_eye = keypoints.get("left_eye")
    nose = keypoints.get("nose_tip")
    right_ear = keypoints.get("right_ear")
    left_ear = keypoints.get("left_ear")

    if right_eye and left_eye and nose:
        eye_dist = abs(left_eye[0] - right_eye[0])
        nose_to_right = abs(nose[0] - right_eye[0])
        nose_to_left = abs(nose[0] - left_eye[0])
        if eye_dist > 0:
            balance = abs(nose_to_right - nose_to_left) / eye_dist
            if balance < 0.25:
                return "front"
            return "left" if nose_to_right < nose_to_left else "right"

    if right_ear and not left_ear:
        return "right"
    if left_ear and not right_ear:
        return "left"

    return "unknown"
