# FH Aachen Vogelscheuchen IP

Dieses IP ist ein Projekt der FH Aachen unter dem Professor Ringbeck. Es handelt sich um eine Vogelscheuche, die mit einem Raspberry Pi und einer Kamera ausgestattet ist. Die Kamera nimmt Bilder auf und wertet diese aus. Zuerst wird ein Bild gemacht, dann wird das Bild analysiert und die Objekte auf dem Bild werden erkannt. Für jedes erkannte Objekt wird eine Klassifizerung durchgeführt. Wenn das Objekt als Fischreiher (Kranich) erkannt wird, wird ein Alarm ausgelöst. Der Alarm besteht aus einem lauten Geräusch, um den Vogel zu verscheuchen. 

## Models und AI-Hat

Der Raspberry Pi hat einen AI-Hat womit er selber die Bilder analysieren kann. Der AI-Hat ist ein Hailo 8L Chip, der auf dem Raspberry Pi aufgesteckt wird. Der Chip ist chnell und kann die Bilder selber in Echtzeit analysieren, ohne einen externen Server zu brauchen.

Für die Bilderkennung werden 2 Models benutzt. Eins ist das yolo Model, das die Objekte auf dem Bild erkennt. Das andere Model ist ein Resnet Model, das die Klassifizierung durchführt. Beide Model mussten extra für den Prozessor konvertiert und optimiert werden.

## Virtual Environment

Um das Projekt zu starten muss ein Virtual Environment mit den zugehörigen Python Packages erstellt werden. Dafür ist auf der Vogelscheuche schon ein Environment vorhanden. Dieses kann mit folgendem Befehl aktiviert werden:

```bash
source /home/vogelscheuche/Vogelscheuche_Workspace/WsEnv/bin/activate
```

### Neues Virtual Environment erstellen

Wichtig beim erstellen vom Venv ist es, dass die HailoRT Library installiert wird. Diese muss von der [Hailo Website](https://hailo.ai/developer-zone/software-downloads/) heruntergeladen werden, dafür braucht man einen Account. Wichtig sind auch die System Packages, die für die Kamera benutzt werden. Mehr dazu auf der [Github Seite von picamera2](https://github.com/raspberrypi/picamera2).

Die korrekte Numpy Version ist auch wichtig, da es sonst zu Fehlern kommen kann.

```bash
pip install numpy==1.26.4
```

## Starten des Projekts

Um das Projekt zu starten, muss das Environment aktiviert sein. 
In der vogelscheucheConfig.yaml Datei können die gewünschten Einstellungen geändert werden.

```bash
python vogelscheucheDetector.py --config vogelscheucheConfig.yaml
```

# Wiki

Im [Wiki](https://github.com/tijavo/VogelscheucheIP/wiki) sind alle wichtigen Informationen zum Projekt zu finden. Dort sind auch die Schritte zum erstellen des Models und des AI-Hats beschrieben.
