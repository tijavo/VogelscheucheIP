import argparse
import yaml
import sys
import gpiod
import time
from vogelscheucheDetector import CombinedDetector
from vogelscheucheAlarm import Alarm

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
    else:
        config = {}
        
    detector = CombinedDetector(
            yolo_model_path=config.get('yolo_model_path', './resources/yolov8n.hef'),
            resnet_model_path=config.get('resnet_model_path', './resources/resnet_v1_18.hef'),
            class_names_path=config.get('class_names_path', './resources/imagenet_names.json'),
            output_dir=config.get('output_dir', 'detected_objects'),
            confidence_threshold=config.get('confidence_threshold', 0.3),
            heronscore_threshold=config.get('heronscore_threshold', 60),
            debug_mode=config.get('debug_mode', False)
        )
    
    alarm = Alarm(
        max_frames = config.get('max_frames', 10),
        activation_threshold = config.get('activation_threshold', 3),
    )
    
    PIR_Timeout = config.get('PIR_Timeout', 10)
    
    PIR_PIN = 23
    gpiod_chip = gpiod.Chip('gpiochip0')
    pir_line = gpiod_chip.get_line(PIR_PIN)
    pir_line.request(consumer='PIR', type=gpiod.LINE_REQ_DIR_IN)
    
    
    try:
        detector.start_camera()
        while True:
            time.sleep(0.2)
            if pir_line.get_value() == 1:
                print("Bewegung erkannt!")
                startTime = time.time()
                while startTime + PIR_Timeout > time.time():
                    image = detector.capture_image()
                    results = detector.process_frame(image)
                    if detector.is_heron_detected(results):
                        alarm.counter_plus()
                    else:
                        alarm.counter_minus()
                print("Scannen zuende")
                alarm.disable_sirene()
                
    except KeyboardInterrupt:
        print("\nProgram terminated")
    finally:
        detector.stop_camera()


if __name__ == '__main__':
    main()
