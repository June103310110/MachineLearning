from car import car_control
from hcsr04 import get_distance
from time import sleep

c = car_control()

try:
    while 1:
        dis = get_distance(27, 17)
        if dis > 30:
            c.forward()
        else:
            #print(dis)
            c.left()
        #sleep(0.1)
except KeyboardInterrupt:
    c.still()
    c = car_control()
    print('關閉程式 ')
    
#c.__version__()

#forward()

c.still()