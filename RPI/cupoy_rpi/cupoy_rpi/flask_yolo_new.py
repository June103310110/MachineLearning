# yolo  + Flask

import os
import sys
from flask import Flask
import time

sys.path.append('/home/pi/darknet/')
#from new_darknet import performDetect
app = Flask(__name__)


def default_yolo(): 
    path = os.getcwd()
    imgpath = '/home/pi/Downloads/tug.jpg'
    configPath = "./cfg/yolov3_training.cfg"
    weightPath = "./cfg/yolov3_training_last.weights"
    metaPath= "./data/obj.data"
    
    os.chdir('/home/pi/darknet')
    from darknet import performDetect

    a = performDetect(imagePath=imgpath, configPath=configPath,
                      weightPath=weightPath, metaPath=metaPath)

    os.chdir(path)
    return a

def defult_display():
    ans = default_yolo()
    text0 = 'detect: '+ans[0][0]+'\tconfidence: '\
        +str(format(ans[0][1], '.5f'))
    text2 = '\ntime is: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    reply = text0 + text2
    return reply


if __name__ == "__main__":
    
    reply = defult_display()
    
    @app.route("/")
    def abc():
        return reply
    
    app.run(host='10.17.4.132', port = 8080,
       debug = False, threaded = True)

