from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2
import time
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter

class PlayerTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=150, n_init=5, nms_max_overlap=0.6, max_cosine_distance=0.28, nn_budget=200, embedder="torchreid")
        self.reid_memory = {}  # {temp_id: {embedding, color_hist, last_seen, reassigned_id}}
        self.id_map = {}  # {temp_id: stable_id}
        self.next_stable_id = 1
        self.max_position_shift = 50  # Max pixels a player can jump between frames
        self.kalman_filters = {}
        self.prev_gray = None
        self.prev_velocity = {}  # {stable_id: np.array([vx, vy])}
        self.velocity_threshold = 25.0  # You can tune this based on your frame rate and resolution
        self.prev_centroids = {}  # {stable_id: np.array([cx, cy])}



        # self.position_history = defaultdict(lambda: deque(maxlen=5))  # Save last 5 positions per track
        # self.jump_threshold = 50  # You can tune this based on frame resolution

    # def update(self, boxes, frame):
    #     if len(boxes) == 0:
    #         return []
    #     # boxes: [x, y, w, h, conf] per player
    #     #print(type(frame), frame.shape if hasattr(frame, 'shape') else None)
    #     tracks = self.tracker.update_tracks(boxes, frame=frame)
    #     tracked_players = []
    #     for track in tracks:
    #         if not track.is_confirmed() or track.time_since_update > 3:
    #             continue
    #         track_id = track.track_id
    #         ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
    #         tracked_players.append((track_id, ltrb))
    #     return tracked_players


    def get_color_histogram(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

         # Mask out green (Hue around 60 ± some margin)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)  # Invert mask to ignore green

        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)  # ✅ enforce float32
        return hist

    def create_kalman_filter(self, initial_bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)  # x, y, s, r + velocity
        dt = 1.0  # time step

        # State: [x, y, s, r, vx, vy, vs]
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        kf.R *= 1.0  # measurement uncertainty
        kf.P *= 10.0
        kf.Q *= 0.01

        # Convert bbox (x1, y1, x2, y2) → (cx, cy, area, aspect_ratio)
        x1, y1, x2, y2 = initial_bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)

        kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        return kf

    def update(self, boxes, frame):
        if len(boxes) == 0:
            return []

        
        tracks = self.tracker.update_tracks(boxes, frame=frame)
        tracked_players = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 3:
                continue

            temp_id = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]

            ###

            # cx = (ltrb[0] + ltrb[2]) // 2
            # cy = (ltrb[1] + ltrb[3]) // 2
            # current_position = np.array([cx, cy])

            # # Check if there's a previous position for this track
            # if len(self.position_history[temp_id]) > 0:
            #     prev_position = self.position_history[temp_id][-1]
            #     movement = np.linalg.norm(current_position - prev_position)

            #     if movement > self.jump_threshold:
            #         print(f"[WARNING] Sudden jump for ID {temp_id}: {movement:.2f} pixels — skipping")
            #         continue  # Skip this ID due to suspicious jump

            # # Add current position to history
            # self.position_history[temp_id].append(current_position)

            # tracked_players.append((temp_id, ltrb))

            ####
            # === Check memory for re-ID ===
            if not hasattr(track, 'features') or len(track.features) == 0:
                continue
            embedding = np.mean(track.features, axis=0)  # average embedding

            hist = self.get_color_histogram(frame, ltrb)
            

            matched_id = None
            best_score = float('inf')

            for stable_id, mem in self.reid_memory.items():
                if hist is None or mem['color'] is None:
                    continue
                emb_sim = np.linalg.norm(mem['embedding'] - embedding)
                color_sim = cv2.compareHist(mem['color'], hist, cv2.HISTCMP_BHATTACHARYYA)
                score = emb_sim + color_sim  # simple combined metric

                # if 'ltrb' in mem:
                #     iou = self.compute_iou(ltrb, mem['ltrb'])
                #     if iou > 0.5 and emb_sim < 0.5:
                #         matched_id = stable_id
                #         break

                if emb_sim < 0.2 and color_sim < 0.175 :  # 0-2.8
                    if score < best_score:
                        best_score = score
                        matched_id = stable_id

            # Assign a stable ID
            if matched_id:
                self.id_map[temp_id] = matched_id
            elif temp_id not in self.id_map:
                self.id_map[temp_id] = self.next_stable_id
                self.next_stable_id += 1

            stable_id = self.id_map[temp_id]
            #tracked_players.append((stable_id, ltrb))

            ###
            if stable_id not in self.kalman_filters:
                self.kalman_filters[stable_id] = self.create_kalman_filter(ltrb)
            kf = self.kalman_filters[stable_id]

            # Convert bbox to measurement: [cx, cy, area, aspect_ratio]
            x1, y1, x2, y2 = ltrb
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            s = (x2 - x1) * (y2 - y1)
            r = (x2 - x1) / (y2 - y1 + 1e-6)
            measurement = np.array([cx, cy, s, r])

            kf.predict()
            kf.update(measurement)

            # Use filtered output from Kalman filter
            cx, cy, s, r = kf.x[:4].flatten()
            ###

            # Calculate velocity
            if stable_id in self.prev_centroids:
                prev_cx, prev_cy = self.prev_centroids[stable_id]
                velocity = np.array([cx - prev_cx, cy - prev_cy])
                speed = np.linalg.norm(velocity)

                # Check sudden jump in velocity or direction change
                if stable_id in self.prev_velocity:
                    delta_v = velocity - self.prev_velocity[stable_id]
                    if np.linalg.norm(delta_v) > self.velocity_threshold:
                        print(f"[Velocity Warning] Sudden movement for ID {stable_id}, skipping frame")
                        continue  # Skip this ID assignment due to suspicious movement

                self.prev_velocity[stable_id] = velocity
            else:
                self.prev_velocity[stable_id] = np.array([0, 0])  # Initialize

            self.prev_centroids[stable_id] = np.array([cx, cy])

            ###
            w = np.sqrt(s * r)
            h = s / (w + 1e-6)

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            smoothed_ltrb = [x1, y1, x2, y2]

            tracked_players.append((stable_id, smoothed_ltrb))
            ####

            # Update memory
            self.reid_memory[stable_id] = {
                'embedding': embedding,
                'color': hist,
                'last_seen': time.time(),
                #'ltrb': smoothed_ltrb ##
            }

        # Clean up old memory
        current_time = time.time()
        self.reid_memory = {
            k: v for k, v in self.reid_memory.items()
            if current_time - v['last_seen'] < 15      # Anyways our video is less than 15 seconds
        }
        active_ids = set(self.reid_memory.keys())
        self.kalman_filters = {k: v for k, v in self.kalman_filters.items() if k in active_ids}
        self.prev_velocity = {k: v for k, v in self.prev_velocity.items() if k in active_ids}
        self.prev_centroids = {k: v for k, v in self.prev_centroids.items() if k in active_ids}

        return tracked_players
