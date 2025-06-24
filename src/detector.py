from ultralytics import YOLO
import torch
import numpy as np
from ultralytics.utils.ops import non_max_suppression

#{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

class PlayerDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = YOLO(model_path)
        print(self.model.names) 
        self.device = device

    def detect(self, frame):
        results = self.model.predict(source=frame, device=self.device, conf=0.8, iou=0.8, verbose=True)
        #results = non_max_suppression(results[0], iou_threshold=0.5)
        boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls) == 2:  # We can also set it to allow for 3 i.e., referee 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    boxes.append(([x1, y1, x2 - x1, y2 - y1], conf, 2))
        
        return boxes

