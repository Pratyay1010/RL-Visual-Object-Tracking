import cv2
import torch
import numpy as np

from ppo_agent import PPO
from feature_extractor import ResNetExtractor
from tracker_utils import apply_action, clamp_box, draw_bbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RLTracker:
    """RL-based visual tracker using trained PPO policy."""
    
    def __init__(self, model_path, state_dim=516, action_dim=6):
        self.feature_extractor = ResNetExtractor().eval()
        self.agent = PPO(state_dim, action_dim)
        self.agent.load(model_path)
        
    def track_video(self, video_path, init_bbox=None):
        """Run tracker on video."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to load video")
            return
        
        # Initialize bounding box
        if init_bbox is None:
            print("Select object to track...")
            bbox = cv2.selectROI("Select Object", frame, False)
            cv2.destroyAllWindows()
            tracker_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
        else:
            tracker_box = np.array(init_bbox, dtype=np.float32)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W = frame.shape[:2]
            tracker_box = clamp_box(tracker_box, frame.shape)
            
            # Extract features from current crop
            x, y, w, h = tracker_box.astype(int)
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                crop = np.zeros((128, 128, 3), dtype=np.uint8)
            
            with torch.no_grad():
                feat = self.feature_extractor(crop).detach().cpu().numpy()
            
            # Build state and select action
            state = np.concatenate([feat, tracker_box / 500.0]).astype(np.float32)
            action, _, _ = self.agent.select_action(state)
            
            # Update bounding box
            tracker_box = apply_action(tracker_box, action)
            tracker_box = clamp_box(tracker_box, frame.shape)
            
            # Visualize
            frame = draw_bbox(frame, tracker_box)
            cv2.imshow("RL Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Path to trained model
    MODEL_PATH = "../models/ppo_tracker_best.pth"
    VIDEO_PATH = "../assets/demo/demo.mp4"

    tracker = RLTracker(MODEL_PATH)
    tracker.track_video(VIDEO_PATH)

if __name__ == "__main__":
    main()