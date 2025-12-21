import os
import cv2

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 1,
    max_frames: int = 30
) -> int:
    """
    Extract frames from a video at fixed FPS.

    Returns:
        int: number of frames extracted
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(video_fps // fps), 1)

    count = 0
    saved = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    return saved
