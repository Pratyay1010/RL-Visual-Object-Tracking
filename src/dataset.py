import os
import cv2
import numpy as np

def load_groundtruth(path):
    """Load groundtruth bounding boxes from OTB format."""
    if not os.path.isfile(path):
        return None
    
    gt = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Handle various separators
            line = line.replace("\t", ",").replace(" ", ",")
            parts = line.split(",")
            if len(parts) < 4:
                continue
            
            try:
                x, y, w, h = map(float, parts[:4])
                gt.append([x, y, w, h])
            except:
                continue
    
    return np.array(gt, dtype=np.float32) if gt else None

def load_sequence(seq_dir, seq_name):
    """Load frames and groundtruth for a sequence."""
    seq_path = os.path.join(seq_dir, seq_name)
    img_dir = os.path.join(seq_path, "img")
    
    # Load frames
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    frames = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(valid_ext)
    ])
    
    # Load groundtruth
    gt_path = os.path.join(seq_path, "groundtruth_rect.txt")
    gt = load_groundtruth(gt_path)
    
    return frames, gt

def load_frame(path):
    """Load image frame."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    return img