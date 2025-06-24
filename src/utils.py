import cv2

def draw_tracks(frame, tracked_players):
    for track_id, bbox in tracked_players:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def video_writer(output_path, frame_shape, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frame_shape
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
