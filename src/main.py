import cv2
from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.utils import draw_tracks, video_writer
import os

VIDEO_PATH = './15sec_input_720p.mp4'
# VIDEO_PATH = './man.mp4'
# MODEL_PATH = 'models/yolov11.pt'
MODEL_PATH = 'models/best.pt'
OUTPUT_PATH = './cosineStricterwithHighWaitBeforeID.mp4'


# Ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = PlayerDetector(MODEL_PATH)
    tracker = PlayerTracker()

    ret, frame = cap.read()
    if not ret:
        print("Failed to load video.")
        return

    out = video_writer(OUTPUT_PATH, frame.shape[:2], fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detector.detect(frame)
        tracked_players = tracker.update(boxes, frame)
        annotated = draw_tracks(frame, tracked_players)
        out.write(annotated)

    cap.release()
    out.release()
    print("Video saved at:", OUTPUT_PATH)


if __name__ == '__main__':
    main()
