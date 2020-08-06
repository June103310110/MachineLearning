import RPi.GPIO as gpio
import time



class car_control():
    def __init__(self):
        gpio.setmode(gpio.BCM)
        gpio.setwarnings(False)
        global a1
        gpio.setup(23,gpio.OUT)
        a1 = gpio.PWM(23, 60)
        
        global b1
        gpio.setup(24,gpio.OUT)
        b1 = gpio.PWM(24, 60)
        
        global a2
        gpio.setup(5,gpio.OUT)
        a2 = gpio.PWM(5, 60)
        
        global b2
        gpio.setup(6,gpio.OUT)
        b2 = gpio.PWM(6, 60)
    def __version__(self):
        print('beta01')
    def still(self):
        a1.stop()
        b1.stop()
        a2.stop()
        b2.stop()
        gpio.cleanup()

    def forward(self):
        a1.start(0)
        b1.start(100)
        a2.start(0)
        b2.start(100)
    def left(self):
        a1.start(30)
        b1.start(0)
        a2.start(0)
        b2.start(0)   
    

#car_initail()

if __name__ == '__main__':
    c = car_control()
    c.left()
    time.sleep(1)
    c.still()