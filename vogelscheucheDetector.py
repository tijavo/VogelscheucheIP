import numpy as np
import cv2
from hailo_platform import (
    VDevice,
    HEF,
    ConfigureParams,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    HailoStreamInterface
)
from pathlib import Path
from picamera2 import Picamera2
import time
from datetime import datetime
import os
import json
from PIL import Image


class CombinedDetector:
    def __init__(self,
                 yolo_model_path: str,
                 resnet_model_path: str,
                 class_names_path: str,
                 output_dir: str = "detected_objects",
                 confidence_threshold: float = 0.5,
                 heronscore_threshold: int = 60,
                 debug_mode: bool = True,
                 offset = 5):
        """
        Initializes the combined detector with YOLO and ResNet
        """
        self.confidence_threshold = confidence_threshold
        self.bounding_box_offset = offset
        self.debug_mode = debug_mode
        self.heronscore_threshold = heronscore_threshold
        
        # Load class names for ResNet
        with open(class_names_path, 'r') as file:
            self.class_names = json.load(file)

        # Hailo Setup: Shared Device
        self.device = VDevice()

        # YOLO Setup
        self.yolo_hef = HEF(yolo_model_path)

        # ResNet Setup
        self.resnet_hef = HEF(resnet_model_path)

        # Configure Device for both models
        yolo_params = ConfigureParams.create_from_hef(hef=self.yolo_hef, interface=HailoStreamInterface.PCIe)
        resnet_params = ConfigureParams.create_from_hef(hef=self.resnet_hef, interface=HailoStreamInterface.PCIe)
        self.yolo_network_groups = self.device.configure(self.yolo_hef, yolo_params)
        self.resnet_network_groups = self.device.configure(self.resnet_hef, resnet_params)
        self.yolo_network_group = self.yolo_network_groups[0]
        self.resnet_network_group = self.resnet_network_groups[0]

        # YOLO Streams
        self.yolo_input_params = InputVStreamParams.make(self.yolo_network_group, format_type=FormatType.UINT8)
        self.yolo_output_params = OutputVStreamParams.make(self.yolo_network_group, format_type=FormatType.FLOAT32)
        self.yolo_input_info = self.yolo_hef.get_input_vstream_infos()[0]
        self.yolo_output_info = self.yolo_hef.get_output_vstream_infos()[0]

        # ResNet Streams
        self.resnet_input_params = InputVStreamParams.make(self.resnet_network_group, format_type=FormatType.UINT8)
        self.resnet_output_params = OutputVStreamParams.make(self.resnet_network_group, format_type=FormatType.UINT8)
        self.resnet_input_info = self.resnet_hef.get_input_vstream_infos()[0]
        self.resnet_output_info = self.resnet_hef.get_output_vstream_infos()[0]

        # Camera Setup
        self.camera = Picamera2()
        config = self.camera.create_still_configuration(main={"size": (1920, 1080)})
        self.camera.configure(config)

        # Output directories
        self.output_dir = output_dir
        if debug_mode:
            self.session_dir = self._create_session_dir()
        print("YOLO Model:", yolo_model_path)
        print("ResNet Model:", resnet_model_path)
        print("Output Directory:", self.session_dir)

    def _create_session_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def start_camera(self):
        self.camera.start()
        time.sleep(2)

    def stop_camera(self):
        self.camera.stop()

    def capture_image(self) -> np.ndarray:
        image = self.camera.capture_array()
        if self.debug_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(self.session_dir, f"raw_capture_{timestamp}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return image

    def capture_image_fake(self,path) -> np.ndarray:
        image = Image.open(path)
        image = np.array(image)
        if self.debug_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(self.session_dir, f"raw_capture_{timestamp}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return image
    
    def _preprocess(self, image: Image.Image, model_w: int, model_h: int) -> Image.Image:
        """
        Resize image with unchanged aspect ratio using padding.

        Args:
            image (PIL.Image.Image): Input image.
            model_w (int): Model input width.
            model_h (int): Model input height.

        Returns:
            PIL.Image.Image: Preprocessed and padded image.
        """
        img_w, img_h = image.size
        scale = min(model_w / img_w, model_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        image = image.resize((new_img_w, new_img_h), Image.Resampling.BICUBIC)

        padded_image = Image.new('RGB', (model_w, model_h), (114, 114, 114))
        padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
        return padded_image

    def process_frame(self, image: np.ndarray) -> list:
        """Processes a frame with YOLO and ResNet"""
        results = []
        
        # YOLO Detection
        detections = self.process_frame_yolo(image)
        if not detections:
            print("No objects detected")
            return results
        
        bboxes = detections['detection_boxes']
        if len(bboxes) == 0:
            print("No Bounding Boxes found")
            return results
        
        img_w, img_h = image.shape[1], image.shape[0]
        
        for i,box in enumerate(bboxes):
            x1, y1, x2, y2 = box
            
            if self.debug_mode:
                print(f"\nBox {i} transformation:")
                print(f"Original normalized coords: {x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}")
            
            # Erst horizontale Spiegelung (x-Achse wird gespiegelt)
            mirror_x1 = 1 - x2  # x1 wird zu (1-x2)
            mirror_x2 = 1 - x1  # x2 wird zu (1-x1)
            mirror_y1 = y1      # y-Koordinaten bleiben zunächst gleich
            mirror_y2 = y2
            
            if self.debug_mode:
                print(f"After mirroring: {mirror_x1:.3f}, {mirror_y1:.3f}, {mirror_x2:.3f}, {mirror_y2:.3f}")
            
            # Dann 90 Grad Rotation nach links
            # Bei 90° Links-Rotation: x wird zu y, y wird zu (1-x)
            temp_x1 = mirror_y1          # Neue x1 kommt von y1
            temp_x2 = mirror_y2          # Neue x2 kommt von y2
            temp_y1 = 1 - mirror_x2      # Neue y1 ist (1-x2)
            temp_y2 = 1 - mirror_x1      # Neue y2 ist (1-x1)
                
            # Skaliere auf Bildgröße mit unterschiedlichem Padding für Höhe und Breite
            padding_factor_x = 0.15  # 15% horizontales Padding
            padding_factor_y = 0.35  # 35% vertikales Padding für mehr Höhe
            
            # Berechne die Breite und Höhe der Box
            width = (temp_x2 - temp_x1) * img_w
            height = (temp_y2 - temp_y1) * img_h
            
            # Berechne das Padding in Pixeln, mit mehr Padding nach oben
            pad_x = int(width * padding_factor_x)
            pad_y = int(height * padding_factor_y)
            pad_y_top = int(height * padding_factor_y * 1.2)  # 20% mehr Padding nach oben
            
            # Wende das Padding an und stelle sicher, dass wir innerhalb der Bildgrenzen bleiben
            x1 = max(0, int(temp_x1 * img_w) - pad_x)
            x2 = min(img_w, int(temp_x2 * img_w) + pad_x)
            y1 = max(0, int(temp_y1 * img_h) - pad_y_top)  # Mehr Padding nach oben
            y2 = min(img_h, int(temp_y2 * img_h) + pad_y)
            
            # Überspringe ungültige oder zu kleine Boxen
            if x2 <= x1 or y2 <= y1 or x2 - x1 < 20 or y2 - y1 < 20:
                if self.debug_mode:
                    print(f"Skipping invalid/small box: {x2-x1}x{y2-y1}")
                continue
            
            # Schneide Objekt aus
            cropped = image[y1:y2, x1:x2]
            results.append(self.process_frame_resnet(cropped))
        return results
    
    def process_frame_resnet(self, image_array: np.ndarray) -> list:
        """Processes a frame with ResNet"""

        # ResNet Classification
        image = Image.fromarray(image_array)
        
        resnet_input = np.array([self._preprocess(image, 224, 224)])
        
        with InferVStreams(self.resnet_network_group, self.resnet_input_params, self.resnet_output_params) as resnet_pipeline:
            resnet_input_data = {self.resnet_input_info.name: np.expand_dims(resnet_input, axis=0)}
            with self.resnet_network_group.activate():
                resnet_outputs = resnet_pipeline.infer(resnet_input_data)
            classifications = self._process_resnet_output(resnet_outputs[self.resnet_output_info.name])

        return classifications
        
    def _process_resnet_output(self, results, n = 3) -> list:
        """Processes ResNet output"""
        max_indices = np.argpartition(results[0], -1* n)[-1*n:]
        result = []
        for i in max_indices:
            result.append({
                'class': self.class_names[str(i+1)],
                'confidence': results[0][i],
                'index': i
            })
        return result
    
    def process_frame_yolo(self, image_array: np.ndarray) -> list:
        """Processes a frame with YOLO"""
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # YOLO Detection
        image = Image.fromarray(image_array)
        yolo_input = np.array([self._preprocess(image, 640, 640)])
        with InferVStreams(self.yolo_network_group, self.yolo_input_params, self.yolo_output_params) as yolo_pipeline:
            yolo_input_data = {self.yolo_input_info.name: yolo_input}
            with self.yolo_network_group.activate():
                yolo_outputs = yolo_pipeline.infer(yolo_input)
            detections = self._process_yolo_output(yolo_outputs[self.yolo_output_info.name][0])
            
        return detections
    
    def _process_yolo_output(self, output_data: list, threshold: float = 0.5) -> dict:
        """
        Extract detections from the input data.

        Args:
            input_data (list): Raw detections from the model.
            threshold (float): Score threshold for filtering detections. Defaults to 0.5.

        Returns:
            dict: Filtered detection results.
        """
        boxes, scores, classes = [], [], []
        num_detections = 0
        
        for i, detection in enumerate(output_data):
            if len(detection) == 0:
                continue
            for det in detection:
                if len(det) == 0:
                    continue
                bbox, score = det[:4], det[4]
                if score >= threshold:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(i)
                    num_detections += 1
                    
        return {
            'detection_boxes': boxes, 
            'detection_classes': classes, 
            'detection_scores': scores,
            'num_detections': num_detections
        }

    def is_heron_detected(self, results):
        for i,result in enumerate(results):
            heronscore = 0
            for det in result:
                if 'heron' in det['class'] or 'crane' in det['class']:
                    heronscore = heronscore + det['confidence']
            if heronscore > self.heronscore_threshold:
                return True
        return False


import argparse
import yaml
import sys

def load_yaml_file(file_path):
    """Lädt eine YAML-Datei und gibt deren Inhalt zurück."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Fehler beim Laden der YAML-Datei: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Ein Skript, welches von der Raspi Cam einen Input nimmt, ein Modell zur Objekterkennung, eins zur Classifizierung benutzt und schaut ob ein Fischreiher im Bild ist.")
    parser.add_argument(
        "--config",
        type=str,
        help="Pfad zur YAML-Konfigurationsdatei",
        required=False
    )
    args = parser.parse_args()
    
    if args.config:
        config = load_yaml_file(args.config)
        detector = CombinedDetector(
            yolo_model_path=config.get('yolo_model_path', './resources/yolov8n.hef'),
            resnet_model_path=config.get('resnet_model_path', './resources/resnet_v1_18.hef'),
            class_names_path=config.get('class_names_path', './resources/imagenet_names.json'),
            output_dir=config.get('output_dir', 'detected_objects'),
            confidence_threshold=config.get('confidence_threshold', 0.3),
            heronscore_threshold=config.get('heronscore_threshold', 60),
            debug_mode=config.get('debug_mode', False)
        )
    else:
        detector = CombinedDetector(
            yolo_model_path='./resources/yolov8n.hef',
            resnet_model_path='./resources/resnet_v1_18.hef',
            class_names_path='./resources/imagenet_names.json',
            output_dir='detected_objects',
            confidence_threshold=0.3,
            heronscore_threshold=60,
            debug_mode=True
        )
    try:
        detector.start_camera()
        while True:
            image = detector.capture_image_fake("./resources/Pictures/ReiherTest.png")
            results = detector.process_frame(image)
            if detector.is_heron_detected(results):
                print("Heron detected")
            else:
                print("No Heron detected")
                
    except KeyboardInterrupt:
        print("\nProgram terminated")
    finally:
        detector.stop_camera()


if __name__ == '__main__':
    main()
