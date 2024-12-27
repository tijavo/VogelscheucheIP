import time
import threading

class Alarm:
    
    def __init__(
                    self, 
                    alarm_duration: int = 10, # How long the alarm will be active
                    sirene_wait_time: int = 5, # How long to wait before the sirene starts
                    blink_interval: float = 0.5, # How long the LED will be on and off
                 ):
        
        self.alarm_duration = alarm_duration
        self.sirene_wait_time = sirene_wait_time
        self.timestamp_last_alarm = 0
        self.blink_interval = blink_interval
        
        self.blinking = False
        self.sirene_on = False
        
        print("Alarm initialisiert")
        print(f"Alarm Timeout: {alarm_duration}")
        print(f"Sirene WaitTime: {sirene_wait_time}")
        if self.sirene_wait_time > self.alarm_duration:
            print("Sirene WaitTime sollte nicht größer als Alarm Timeout sein")
        
    def blink_led(self, duration=10, blink_interval=0.5):
        end_time = time.time() + duration
        while time.time() < end_time and self.blinking:
            print("LED 1")
            #GPIO.output(LED_PIN, GPIO.HIGH)  # LED an
            time.sleep(blink_interval / 2)   # Halbe Blinkzeit an
            #GPIO.output(LED_PIN, GPIO.LOW)   # LED aus
            print("LED 0")
            time.sleep(blink_interval / 2)   # Halbe Blinkzeit aus
        self.disable_alarm()
        
    def enable_alarm(self):
        """
        Aktiviert den Alarm und lässt die LED blinken.
        """
        if self.blinking:
            return
        
        self.blinking = True
        print("Alarm aktiviert! LED beginnt zu blinken.")
        blink_thread = threading.Thread(target=self.blink_led, args=(self.alarm_duration,self.blink_interval))
        
        self.timestamp_last_alarm = time.time()
        blink_thread.start()
        return blink_thread

    def enable_sirene(self):
        """
        Aktiviert die Sirene.
        """
        if self.sirene_on:
            return
        
        self.sirene_on = True
        print("Sirene aktiviert!")
        #GPIO.output(SIRENE_PIN, GPIO.HIGH)

    def trigger_alarm(self):
        if self.timestamp_last_alarm == 0 or time.time() - self.timestamp_last_alarm > self.alarm_duration:
            self.enable_alarm()
        if self.timestamp_last_alarm != 0 and time.time() - self.timestamp_last_alarm > self.sirene_wait_time and not self.sirene_on:
            self.enable_sirene()
            

    def disable_alarm(self):
        print("Alarm ausgeschaltet")
        self.blinking = False
        self.sirene_on = False
        #GPIO.output(LED_PIN, GPIO.LOW)
        #GPIO.output(SIRENE_PIN, GPIO.LOW)