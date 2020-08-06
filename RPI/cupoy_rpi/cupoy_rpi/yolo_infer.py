import os
import argparse
import imghdr
import sys
sys.path.append('/home/pi/darknet/')


if __name__ == "__main__":
    os.chdir('/home/pi/darknet')
    from new_darknet import performDetect
    
    parser = argparse.ArgumentParser()
    parser.add_argument('img_abs_path')
    args = parser.parse_args()
    img_path = args.img_abs_path
    lis = ['jpeg', 'png', 'bnp']
    if imghdr.what(img_path) in lis:
        ans = performDetect(imagePath= img_path,
                      configPath = "./cfg/yolov3_training.cfg",
                      weightPath = "./cfg/yolov3_training_last.weights",
                      metaPath= "./data/obj.data")
        print(ans)
    else:
        print('Path error or not image')

