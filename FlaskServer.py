import argparse
import yaml
import sys
import gpiod
import time
from vogelscheucheDetector import CombinedDetector
from vogelscheucheAlarm import Alarm

from flask import Flask, Response, render_template, request
import threading
import cv2

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
            heronscore_threshold=config.get('heronscore_threshold', 60),
            debug_mode=config.get('debug_mode', False)
        )
    
    alarm = Alarm(
        max_frames = config.get('max_frames', 10),
        activation_threshold = config.get('activation_threshold', 3),
        SIRENE_PIN=config.get('SIRENE_PIN', 24)
    )
    
    PIR_Timeout = config.get('PIR_Timeout', 10)
    
    # GPIO-Setup
    PIR_PIN = config.get('PIR_PIN', 23)
    
    gpiod_chip = gpiod.Chip('gpiochip0')
    pir_line = gpiod_chip.get_line(PIR_PIN)
    pir_line.request(consumer='PIR', type=gpiod.LINE_REQ_DIR_IN)
    
    global image
    global results
    
    try:
        detector.start_camera()
        while True:
            time.sleep(0.2)
            if True or pir_line.get_value() == 1:
                print("Bewegung erkannt!")
                startTime = time.time()
                while startTime + PIR_Timeout > time.time():
                    image = detector.capture_image()
                    results = detector.process_frame(image)
                    print(results)
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


app = Flask(__name__)

@app.route('/')
def index():
    """Startseite mit eingebettetem Video-Stream"""
    return """
    <html>
        <head>
            <title>Raspberry Pi Kamera Stream</title>
        </head>
        <body>
            <h1>Live-Stream der Raspberry Pi Kamera</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    """
    
def generate():
    global image
    while True:
        if image is None:
            continue
        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Ruft den Video-Feed als HTTP-Stream ab"""
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    """Startet den Flask-Server"""
    print("Starting Flask server")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    print("Cursed Print")

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    main()
