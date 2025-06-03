"""
Description:
    A comprehensive, multi-threaded AI-based vision pipeline that takes in frames from an IMU camera
    and performs a variety of AI tasks, including:
      - Face detection
      - Object detection (YOLOv5)
      - Semantic segmentation (DeepLabv3)
      - Monocular depth estimation (MiDaS)
      - Human pose estimation (MediaPipe)
      - Hand gesture recognition (MediaPipe)
      - Object tracking (SORT placeholder)
      - Visual overlays and logging
"""

import os
import sys
import cv2
import time
import threading
import queue
import logging
import numpy as np

# Torch and torchvision for loading AI models
import torch
import torchvision.transforms as T
from torchvision import models

# MediaPipe for pose and hand detection
import mediapipe as mp

# Additional dependencies: filterpy (for SORT), Pillow
# pip install opencv-python torch torchvision mediapipe filterpy numpy Pillow

# ----------------------------
# Configuration & Logging
# ----------------------------
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Global GPU flag
USE_CUDA = torch.cuda.is_available()

# ----------------------------
# Utility Functions
# ----------------------------
def resize_and_pad(image, size=(640, 640), pad_color=(114, 114, 114)):
    """
    Resize an image to fit within 'size' while maintaining aspect ratio,
    then pad with 'pad_color' to match exactly 'size'.
    """
    h0, w0 = image.shape[:2]
    r = min(size[0] / float(h0), size[1] / float(w0))
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = size[1] - new_unpad[0], size[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_padded = cv2.copyMakeBorder(
        image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )
    return image_padded, r, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Applies Non-Maximum Suppression on inference results to filter overlapping boxes.
    prediction: (num_boxes x 6) tensor [x1, y1, x2, y2, conf, cls]
    Returns list of filtered boxes per image.
    """
    # Placeholder: In practice, import and use YOLOv5’s NMS implementation.
    return prediction

def load_class_names(namesfile):
    """
    Load class names from a .txt file, one per line.
    """
    with open(namesfile, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    return names

# ----------------------------
# Camera Handler
# ----------------------------
class CameraHandler:
    """
    Captures frames from a webcam (or video file) in a separate thread,
    pushing them into a queue for downstream processing.
    """
    def __init__(self, src=0, width=1280, height=720, queue_size=5):
        self.src = src
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {self.src}")
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        logging.info("CameraHandler initialized")

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
                logging.error("Camera disconnected")
                break
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                logging.error("Empty frame received")
                break
            if not self.queue.full():
                self.queue.put(frame)
            time.sleep(0.005)

    def read(self):
        return self.queue.get()

    def more(self):
        return not self.queue.empty()

    def stop(self):
        self.stopped = True
        self.cap.release()
        logging.info("CameraHandler stopped")

# ----------------------------
# Face Detector (Haar Cascade)
# ----------------------------
class FaceDetector:
    """
    A simple face detector using OpenCV's Haar Cascades.
    """
    def __init__(self, model_path='haarcascade_frontalface_default.xml'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found. Download from OpenCV repo.")
        self.detector = cv2.CascadeClassifier(model_path)
        logging.info("FaceDetector loaded Haar Cascade")

    def detect(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
        )
        results = []
        for (x, y, w, h) in faces:
            results.append({'bbox': (x, y, w, h), 'confidence': None, 'class': 'face'})
        return results

# ----------------------------
# Object Detector (YOLOv5)
# ----------------------------
class ObjectDetector:
    """
    Loads a YOLOv5 model from Torch Hub and performs object detection.
    """
    def __init__(self, model_name='yolov5s', device=None, conf_thres=0.25, iou_thres=0.45):
        self.device = device or ('cuda' if USE_CUDA else 'cpu')
        logging.info(f"Loading YOLOv5 model: {model_name} on {self.device}")
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True).to(self.device)
        self.model.eval()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = self.model.names  # class names
        logging.info("YOLOv5 model loaded")

    def detect(self, frame):
        img, r, (dw, dh) = resize_and_pad(frame, size=(640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img).to(self.device)
        img = img.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            pred = self.model(img)[0]  # (num_boxes, 6): x1,y1,x2,y2,conf,cls
            pred = pred.cpu()
            pred = pred[pred[:, 4] > self.conf_thres]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        results = []
        if pred is not None:
            for *box, conf, cls in pred:
                x1, y1, x2, y2 = box
                # Rescale coordinates back to original frame
                x1 = int((x1 - dw) / r)
                y1 = int((y1 - dh) / r)
                x2 = int((x2 - dw) / r)
                y2 = int((y2 - dh) / r)
                cls = int(cls)
                label = self.names[cls]
                results.append({'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'confidence': float(conf),
                                'class': label})
        return results

# ----------------------------
# Semantic Segmenter (DeepLabv3)
# ----------------------------
class SemanticSegmenter:
    """
    Loads a DeepLabV3 ResNet101 model for semantic segmentation.
    """
    def __init__(self, device=None):
        self.device = device or ('cuda' if USE_CUDA else 'cpu')
        logging.info(f"Loading DeepLabV3 model on {self.device}")
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(self.device)
        self.model.eval()
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((520, 520)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        logging.info("DeepLabV3 model loaded")

    def segment(self, frame):
        img = self.preprocess(frame).to(self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img)['out'][0]  # (21, H, W)
        output_predictions = output.argmax(0).cpu().numpy()  # segmentation map
        # Resize back to original frame size
        seg_map = cv2.resize(output_predictions.astype(np.uint8),
                             (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        return seg_map  # Each pixel value is class index

# ----------------------------
# Depth Estimator (MiDaS)
# ----------------------------
class DepthEstimator:
    """
    Monocular depth estimation using MiDaS v2.1.
    """
    def __init__(self, model_type="MiDaS_small", device=None):
        self.device = device or ('cuda' if USE_CUDA else 'cpu')
        logging.info(f"Loading MiDaS model: {model_type} on {self.device}")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device).eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "MiDaS_small":
            self.transform = self.transforms.small_transform
        else:
            self.transform = self.transforms.default_transform
        logging.info("MiDaS model loaded")

    def estimate(self, frame):
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(input_img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_img.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        # Normalize depth for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        return depth_vis  # Grayscale depth map

# ----------------------------
# Pose Estimator (MediaPipe)
# ----------------------------
class PoseEstimator:
    """
    Human pose estimation using MediaPipe.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=1,
                                      smooth_landmarks=True,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        logging.info("MediaPipe PoseEstimator initialized")

    def estimate(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]), lm.visibility))
        return landmarks  # List of (x, y, visibility)

# ----------------------------
# Hand Gesture Recognizer (MediaPipe)
# ----------------------------
class HandGestureRecognizer:
    """
    Hand landmark detection using MediaPipe to recognize simple gestures.
    """
    def __init__(self, max_num_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        logging.info("MediaPipe HandGestureRecognizer initialized")

    def recognize(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gestures = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for visualization (optional)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # Simple gesture rule: if thumb tip (landmark 4) is above index finger tip (landmark 8)
                lm = hand_landmarks.landmark
                h, w, _ = frame.shape
                x4, y4 = int(lm[4].x * w), int(lm[4].y * h)
                x8, y8 = int(lm[8].x * w), int(lm[8].y * h)
                if y4 < y8:
                    gestures.append("thumbs_up")
                else:
                    gestures.append("neutral")
        return gestures  # List of recognized gestures per hand

# ----------------------------
# Object Tracker (SORT placeholder)
# ----------------------------
class SortTracker:
    """
    A simple SORT tracker implementation (placeholder).
    For an actual implementation, integrate with a SORT/Deep SORT library.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        # Placeholder attributes
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = {}  # track_id -> {'bbox', 'age', 'hits'}
        logging.info("SortTracker initialized (placeholder)")

    def update(self, detections):
        """
        Update tracks given a list of detections (each as {'bbox', 'confidence', 'class'}).
        Returns updated tracks list.
        """
        updated_tracks = []
        # Very naive: assign each detection a new track
        for det in detections:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {'bbox': det['bbox'], 'age': 0, 'hits': 1, 'class': det['class']}
            updated_tracks.append({'track_id': track_id, 'bbox': det['bbox'], 'class': det['class']})
        return updated_tracks

# ----------------------------
# Visualizer
# ----------------------------
class Visualizer:
    """
    Draws bounding boxes, segmentation overlays, depth maps, pose skeletons,
    and gesture labels onto frames.
    """
    def __init__(self, class_colors=None):
        # Pre-define some colors per class
        self.class_colors = class_colors or {}
        self.default_color = (0, 255, 0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(self, frame, detections):
        for det in detections:
            x, y, w, h = det['bbox']
            cls = det['class']
            conf = det['confidence']
            color = self.class_colors.get(cls, self.default_color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(frame, label, (x, y - 10), self.font, 0.5, color, 1)

    def draw_segmentation(self, frame, seg_map):
        # Create a random color palette (for 21 Pascal VOC classes)
        palette = np.random.randint(0, 255, (21, 3), dtype=np.uint8)
        colored_mask = palette[seg_map]
        overlay = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)
        return overlay

    def draw_depth(self, frame, depth_map):
        # Convert depth map to a color map (Inferno)
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        h, w = frame.shape[:2]
        depth_color_resized = cv2.resize(depth_color, (w // 4, h // 4))
        # Place depth map in the top-left corner
        frame[0: h // 4, 0: w // 4] = depth_color_resized

    def draw_pose(self, frame, landmarks):
        # Draw simple circles at each landmark
        for (x, y, vis) in landmarks:
            if vis > 0.5:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    def draw_tracker(self, frame, tracks):
        for trk in tracks:
            x, y, w, h = trk['bbox']
            cls = trk['class']
            tid = trk['track_id']
            color = self.class_colors.get(cls, (255, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID {tid}", (x, y - 10), self.font, 0.5, color, 1)

    def draw_gestures(self, frame, gestures):
        for idx, gesture in enumerate(gestures):
            cv2.putText(frame, gesture, (10, 30 + idx * 20), self.font, 0.7, (0, 255, 255), 2)

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    # Instantiate all components
    cam_handler = CameraHandler(src=0, width=1280, height=720, queue_size=5)
    face_detector = FaceDetector(model_path='haarcascade_frontalface_default.xml')
    obj_detector = ObjectDetector(model_name='yolov5s', conf_thres=0.3, iou_thres=0.45)
    segmenter = SemanticSegmenter()
    depth_estimator = DepthEstimator(model_type="MiDaS_small")
    pose_estimator = PoseEstimator()
    hand_recognizer = HandGestureRecognizer(max_num_hands=2)
    tracker = SortTracker()
    visualizer = Visualizer(class_colors={'person': (0, 255, 0),
                                          'car': (255, 0, 0),
                                          'bicycle': (0, 0, 255)})

    cv2.namedWindow("AI Pipeline", cv2.WINDOW_NORMAL)

    while True:
        if not cam_handler.more():
            time.sleep(0.01)
            continue

        frame = cam_handler.read()
        orig_frame = frame.copy()

        # 1. Face Detection
        faces = face_detector.detect(frame)

        # 2. Object Detection
        objects = obj_detector.detect(frame)

        # 3. Object Tracking
        tracks = tracker.update(objects)

        # 4. Semantic Segmentation
        seg_map = segmenter.segment(frame)

        # 5. Depth Estimation
        depth_map = depth_estimator.estimate(frame)

        # 6. Pose Estimation
        landmarks = pose_estimator.estimate(frame)

        # 7. Hand Gesture Recognition
        gestures = hand_recognizer.recognize(frame)

        # 8. Visual Overlays
        # Draw segmentation overlay
        frame = visualizer.draw_segmentation(frame, seg_map)

        # Draw detected objects and faces
        visualizer.draw_detections(frame, objects + faces)

        # Draw tracked objects
        visualizer.draw_tracker(frame, tracks)

        # Draw pose landmarks
        visualizer.draw_pose(frame, landmarks)

        # Draw hand gestures
        visualizer.draw_gestures(frame, gestures)

        # Draw depth map overlay
        visualizer.draw_depth(frame, depth_map)

        # Display
        cv2.imshow("AI Pipeline", frame)

        # Logging (every ~30 frames)
        if int(time.time() * 10) % 30 == 0:
            logging.info(f"Faces: {len(faces)}, Objects: {len(objects)}, Gestures: {gestures}")

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    # Cleanup
    cam_handler.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()