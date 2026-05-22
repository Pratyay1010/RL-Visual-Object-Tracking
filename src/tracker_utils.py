import cv2
import numpy as np

def apply_action(box, action, move_ratio=0.05, scale_ratio=0.05):
    """Apply discrete action to bounding box."""
    x, y, w, h = box.astype(float)
    
    if action == 0:      # move left
        x -= move_ratio * w
    elif action == 1:    # move right
        x += move_ratio * w
    elif action == 2:    # move up
        y -= move_ratio * h
    elif action == 3:    # move down
        y += move_ratio * h
    elif action == 4:    # grow
        w *= (1 + scale_ratio)
        h *= (1 + scale_ratio)
    elif action == 5:    # shrink
        w *= (1 - scale_ratio)
        h *= (1 - scale_ratio)
    
    return np.array([x, y, w, h], dtype=np.float32)

def clamp_box(box, img_shape):
    """Clamp bounding box to image boundaries."""
    H, W = img_shape[:2]
    x, y, w, h = box
    
    # Handle NaN/Inf
    if np.any(np.isnan(box)) or np.any(np.isinf(box)):
        return np.array([0, 0, 50, 50], dtype=np.float32)
    
    w = np.clip(w, 5, W)
    h = np.clip(h, 5, H)
    x = np.clip(x, 0, W - w)
    y = np.clip(y, 0, H - h)
    
    return np.array([x, y, w, h], dtype=np.float32)

def iou(box1, box2):
    """Compute Intersection over Union between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xa1, ya1 = x1, y1
    xa2, ya2 = x1 + w1, y1 + h1
    xb1, yb1 = x2, y2
    xb2, yb2 = x2 + w2, y2 + h2
    
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    
    return inter_area / union

def draw_bbox(frame, box, color=(0, 255, 0), thickness=2):
    """Draw bounding box on frame."""
    x, y, w, h = box.astype(int)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame