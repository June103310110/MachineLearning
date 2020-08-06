from hcsr04sensor import sensor
import time

TRIGGER_PIN = 27
ECHO_PIN = 17

def get_distance(TRIGGER_PIN, ECHO_PIN):
    #try:
        #print('按下 Ctrl-C 可以中斷程式')
        #while True:
    sr04 = sensor.Measurement(TRIGGER_PIN, ECHO_PIN)
    raw_measurement = sr04.raw_distance()
    distance = sr04.distance_metric(raw_measurement)
    #print(type(distance))
    #print('距離為 {:.1f} 公分'.format(distance))
    time.sleep(0.01)
    return distance
    #except KeyboardInterrupt:
    #    return 0
        #print('關閉程式 ')
        
if __name__ == '__main__':
    get_distance(TRIGGER_PIN, ECHO_PIN)