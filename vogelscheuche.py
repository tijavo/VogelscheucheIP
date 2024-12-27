import argparse
import yaml
import sys
from vogelscheucheDetector import CombinedDetector

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
