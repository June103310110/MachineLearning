# yolo  + Flask

import os
import argparse
import imghdr
import sys
from flask import Flask
import time
sys.path.append('/home/pi/darknet/')

app = Flask(__name__)


def default_yolo():
    path = os.getcwd()
   # def hello():
   # return 'hello, Here is June'
    imgpath = '/home/pi/Downloads/tug.jpg'
    os.chdir('/home/pi/darknet')
    from new_darknet import performDetect

    a = performDetect(imagePath=imgpath)
    os.chdir(path)
    return a
ans = default_yolo()
text0 = 'detect: '+ans[0][0]+'\tconfidence: '\
        +str(format(ans[0][1], '.5f'))
text2 = '\ntime is: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(text0+text2)


#@app.route("/")
#def hello():
#    a = default_yolo('/home/pi/Downloads/tug.jpg')
#    return a 

@app.route("/")
def hello():
    text0 = 'detect: '+ans[0][0]+'\tconfidence: '\
        +str(format(ans[0][1], '.5f'))
    text2 = '\ntime is: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return text0+text2

if __name__ == "__main__":

    app.run(host='10.17.4.132', port = 8080,
       debug = True, threaded = True)
    
    path = os.getcwd()
    os.chdir('/home/pi/darknet')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('img_abs_path')
    args = parser.parse_args()
    img_path = args.img_abs_path
    lis = ['jpeg', 'png', 'bnp']
    try:
        if imghdr.what(img_path) in lis:
            from new_darknet import performDetect
            #os.system('python new_darknet.py')
            a = performDetect(imagePath=img_path)
            print(a)
        #os.system("""./darknet detector test\
          #  ./data/obj.data\
          #  ./cfg/yolov3_training.cfg\
         #   ./cfg/yolov3_training_last.weights """\
        #    +img_path)
    except FileNotFoundError:
        print('Path error or not image')
        
    os.chdir(path)

    
