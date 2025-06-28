from ultralytics import YOLO
import torch
import numpy as np
from ultralytics.utils.ops import non_max_suppression

#{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

class PlayerDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = YOLO(model_path)
        #print(self.model.names) 
        self.device = device

    def detect(self, frame):
        results = self.model.predict(source=frame, device=self.device, conf=0.84, iou=0.3, verbose=True)
        #results = non_max_suppression(results[0], iou_threshold=0.5)
        boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls) == 2:  # We can also set it to allow for 3 i.e., referee 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    #boxes.append(([x1, y1, x2 - x1, y2 - y1], conf, 2))
                    w, h = x2 - x1, y2 - y1

                # === SHRINK BOXES (focus more on player, less background) ===
                    
                    shrink_ratio = 0.21
                    if(w<=20):
                        shrink_ratio = 0.05
                    elif(w<=25):
                        shrink_ratio = 0.11
                    elif(w<=30):
                        shrink_ratio = 0.17
                    x1 += int(w * shrink_ratio)
                    x2 -= int(w * shrink_ratio)
                    shrink_ratio = 0
                    # if(h<=35):
                    #     shrink_ratio = 0.03
                    # elif(h<=40):
                    #     shrink_ratio = 0.08
                    # elif(h<=45):
                    #     shrink_ratio = 0.13
                    y1 += int(h * shrink_ratio)
                    y2 -= int(h * shrink_ratio)
                    w, h = x2 - x1, y2 - y1
                    # print("Width is", w) # Mostly above 20
                    # print("Height is", h) #mostly above 35
                    # print("Product is", w*h)
                    # Filter invalid (shrunken to zero or negative size)
                    if w > 0 and h > 0:
                        #boxes.append([x1, y1, w, h, conf, 2])  # class_id 2 for player
                        if(w*h>600):
                            boxes.append(([x1, y1, x2 - x1, y2 - y1], conf, 2))
        
        # formatted_boxes = []
        # xyxys = []
        # for i in boxes:
        #     box = i[0]
        #     x1, y1, w, h = box
        #     x2 = x1 + w
        #     y2 = y1 + h
        #     x1 = x1 + w * 0.1
        #     y1 = y1 + h * 0.1
        #     x2 = x2 - w * 0.1
        #     y2 = y2 - h * 0.1
        #     if (w * h) < 1200:
        #         continue
        #     xyxys.append([x1, y1, x2, y2])
        #     formatted_boxes.append(([x1, y1, x2, y2], conf, 2))

        # #boxes = np.array(formatted_boxes, dtype=np.float32)

        return boxes

