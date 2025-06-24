from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=100, n_init=3, nms_max_overlap=0.6, max_cosine_distance=0.3)

    def update(self, boxes, frame):
        if len(boxes) == 0:
            return []
        # boxes: [x, y, w, h, conf] per player
        #print(type(frame), frame.shape if hasattr(frame, 'shape') else None)
        tracks = self.tracker.update_tracks(boxes, frame=frame)
        tracked_players = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 3:
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            tracked_players.append((track_id, ltrb))
        return tracked_players