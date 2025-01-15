import time
import threading
import gpiod
import time


class Alarm:
    def __init__(
                    self, 
                    max_frames: int = 10, # How many frames will be captured
                    activation_threshold: int = 3, # How many frames need to be positive to trigger the alarm
                    SIRENE_PIN: int = 24
                 ):
        


        # GPIO-Modus setzen
        chip = gpiod.Chip('gpiochip4')
        self.sirene_line = chip.get_line(SIRENE_PIN)
        self.sirene_line.request(consumer='vogelscheucheAlarm', type=gpiod.LINE_REQ_DIR_OUT)
        self.sirene_line.set_value(0)
        
        self.max_frames = max_frames
        self.activation_threshold = activation_threshold
        
        self.blinking = False
        self.sirene_on = False
        
        self.counter = 0
        print("Alarm initialisiert")
            

    def enable_sirene(self):
        """
        Aktiviert die Sirene.
        """
        if self.sirene_on:
            return
        
        self.sirene_on = True
        print("Sirene aktiviert!")
        self.sirene_line.set_value(1)
  
    def disable_sirene(self):
        if not self.sirene_on:
            return
        print("Alarm ausgeschaltet")
        self.sirene_on = False
        self.sirene_line.set_value(0)
          
    def counter_plus(self):
        if self.counter < self.max_frames:
            self.counter += 1
        if self.counter >= self.activation_threshold:
            self.enable_sirene()   
            
    def counter_minus(self):
        if self.counter > 0:
            self.counter -= 1
        if self.counter < self.activation_threshold:
            self.disable_sirene()  
