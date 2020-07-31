import os
import argparse
import imghdr
if __name__ == "__main__":
    os.chdir('/home/pi/darknet')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('img_abs_path')
    args = parser.parse_args()
    img_path = args.img_abs_path
    lis = ['jpeg', 'png', 'bnp']
    if imghdr.what(img_path) in lis:
        os.system("""./darknet detector test\
            ./data/obj.data\
            ./cfg/yolov3_training.cfg\
            ./cfg/yolov3_training_last.weights """\
            +img_path)
    else:
        print('Path error or not image')
