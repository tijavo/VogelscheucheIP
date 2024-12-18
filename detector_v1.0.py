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


class CombinedDetector:
    def __init__(self,
                 yolo_model_path: str,
                 resnet_model_path: str,
                 class_names_path: str,
                 output_dir: str = "detected_objects",
                 confidence_threshold: float = 0.5,
                 debug_mode: bool = True):
        """
        Initializes the combined detector with YOLO and ResNet
        """
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode

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
        self.yolo_input_params = InputVStreamParams.make(self.yolo_network_group, format_type=FormatType.FLOAT32)
        self.yolo_output_params = OutputVStreamParams.make(self.yolo_network_group, format_type=FormatType.FLOAT32)
        self.yolo_input_info = self.yolo_hef.get_input_vstream_infos()[0]
        self.yolo_output_info = self.yolo_hef.get_output_vstream_infos()[0]

        # ResNet Streams
        self.resnet_input_params = InputVStreamParams.make(self.resnet_network_group, format_type=FormatType.FLOAT32)
        self.resnet_output_params = OutputVStreamParams.make(self.resnet_network_group, format_type=FormatType.FLOAT32)
        self.resnet_input_info = self.resnet_hef.get_input_vstream_infos()[0]
        self.resnet_output_info = self.resnet_hef.get_output_vstream_infos()[0]

        # Camera Setup
        self.camera = Picamera2()
        config = self.camera.create_still_configuration(main={"size": (1920, 1080)})
        self.camera.configure(config)

        # Output directories
        self.output_dir = output_dir
        self.session_dir = self._create_session_dir()
        self.debug_dir = os.path.join(self.session_dir, "debug") if debug_mode else None

        if debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
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
            cv2.imwrite(os.path.join(self.debug_dir, f"raw_capture_{timestamp}.jpg"), image)
        return image

    def capture_image_fake(self) -> np.ndarray:
        image = cv2.imread("ReiherTest.png")
        if self.debug_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(self.debug_dir, f"raw_capture_{timestamp}.jpg"), image)
        return image

    def _preprocess_yolo(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the image for YOLO"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_size = (640, 640)
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
        dw, dh = (target_size[1] - new_w) // 2, (target_size[0] - new_h) // 2
        padded[dh:dh+new_h, dw:dw+new_w] = resized
        normalized = padded.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def _preprocess_resnet(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the image for ResNet"""
        resized = cv2.resize(image, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def process_frame(self, image: np.ndarray) -> list:
        """Processes a frame with YOLO and ResNet"""
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # YOLO Detection
        yolo_input = self._preprocess_yolo(image)
        with InferVStreams(self.yolo_network_group, self.yolo_input_params, self.yolo_output_params) as yolo_pipeline:
            yolo_input_data = {self.yolo_input_info.name: yolo_input}
            with self.yolo_network_group.activate():
                yolo_outputs = yolo_pipeline.infer(yolo_input_data)
            detections = self._process_yolo_output(yolo_outputs[self.yolo_output_info.name])

        # Process YOLO detections
        if not detections:
            print("No objects detected")
        for idx, det in enumerate(detections):
            print(f"Object {idx + 1}: {det['bbox']} with confidence {det['confidence']:.2f}")
            if det['confidence'] >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                resnet_input = self._preprocess_resnet(cropped)
                with InferVStreams(self.resnet_network_group, self.resnet_input_params, self.resnet_output_params) as resnet_pipeline:
                    resnet_input_data = {self.resnet_input_info.name: resnet_input}
                    with self.resnet_network_group.activate():
                        resnet_outputs = resnet_pipeline.infer(resnet_input_data)
                    classifications = self._process_resnet_output(resnet_outputs[self.resnet_output_info.name])
                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'yolo_confidence': det['confidence'],
                    'classifications': classifications,
                })
        return results

    def _process_yolo_output(self, output_data) -> list:
        """Processes YOLO output"""
        detections = []
        for detection_arrays in output_data:
            for class_detections in detection_arrays:
                if isinstance(class_detections, np.ndarray) and class_detections.size > 0:
                    bbox = class_detections[:4]
                    confidence = float(class_detections[4])
                    if confidence >= self.confidence_threshold:
                        detections.append({'bbox': bbox, 'confidence': confidence})
        return detections

    def _process_resnet_output(self, output_data, top_k=3) -> list:
        """Processes ResNet output"""
        results = output_data[0]
        top_indices = np.argsort(results)[-top_k:][::-1]
        classifications = [{'class': self.class_names.get(str(idx + 1), "Unknown"), 'confidence': float(results[idx])} for idx in top_indices]
        return classifications


def main():
    detector = CombinedDetector(
        yolo_model_path='yolov8n.hef',
        resnet_model_path='resnet_v1_18.hef',
        class_names_path='imagenet_names.json',
        output_dir='detected_objects',
        confidence_threshold=0.3,
        debug_mode=True
    )
    try:
        detector.start_camera()
        if True:
            image = detector.capture_image_fake()
            results = detector.process_frame(image)
            for result in results:
                print(f"Object detected with confidence: {result['yolo_confidence']:.2f}")
    except KeyboardInterrupt:
        print("\nProgram terminated")
    finally:
        detector.stop_camera()


if __name__ == '__main__':
    main()
