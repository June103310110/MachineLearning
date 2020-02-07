
# win10 ubuntu TFlite Label_image專案測試

#### 參考連結

```
1. 安裝TFlite
    - https://www.tensorflow.org/lite/guide/python
2. Google官方Label_image專案位置
    - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python
3. June的github專案位置
    - https://github.com/June103310110/MachineLearning/tree/master/win10_buntu_test/Label_image
4. label_image.py的google drive位置
    - https://drive.google.com/open?id=1x-sHSo_69lY9oUVEWp6OoY92wBN9OJl4
```


### 專案流程:
> ## 1. 安裝ubuntu 16.04 LTS
從windows的app store裡找到ubuntu 16.04 LTS並安裝。

![](https://i.imgur.com/oxazE7s.png)

<br>

> ## 2. ubuntu 環境設置


```
# 1.升級apt
sudo apt update

# 2.安裝pip3(Note: 用pip3而不是pip)
sudo apt install python3-pip

# 3.安裝pillow(PIL)
pip3 install pillow 

# 4.安裝TFlite
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0-cp36-cp36m-linux_x86_64.whl
```
<br>

> ## 3.  下載Label_image專案的必要資源
- 專案github連結
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python

- 要下載的檔案包括:
    1. pre-trained model(mobilenet_v1_1.0_224)
    2. 測試圖片(grace_hopper.bmp)
    3. label的文字檔(labels.txt)
    
- 操作指令如下: 
```
# 1.Get photo (取得圖片)
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp > /tmp/grace_hopper.bmp
# 2.Get model (取得模型)
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp
# 3.Get labels (取得標記檔案)
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt

mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```
###### 圖片下載後如下所示: 
![](https://i.imgur.com/kMk8tvd.png)

<br>

> ## 4. 下載label_image.py
由於我們只想要用TFlite來推論，不需要完整的Tensorflow，所以我們在前面就只有下載TFlite的部分。
而Label_image這個專案在google的github上並不是專門為TFlite準備，所以我們要稍微修改程式碼。
這邊可以讓各位直接下載修改好的程式碼到 **桌面** ，請點這個[連結](https://drive.google.com/open?id=1x-sHSo_69lY9oUVEWp6OoY92wBN9OJl4)，或者從前面參考連結那邊的[June的github專案位置](https://github.com/June103310110/MachineLearning/tree/master/win10_buntu_test/Label_image)裡面下載。

### 重要: 如何把label_image.py移動到home
因為我們使用的ubuntu是win10的子系統，所以還是可以從win10的系統中獲取資源。
```
# 移動到win10的桌面
cd /mnt/c/Users/June/Desktop/

# 把label_image.py複製回ubuntu系統的home位置
cp label_image.py ~
```

#### Note: 如果要自己修改看看可以參考[google的專案連結](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python)

<br>

> ## 5. 推論模型
由於我們python版本是3.6.9，所以是使用`python3`這個指令來執行程式。

##### 執行的程式如下:

```
python3 label_image.py   --model_file /tmp/mobilenet_v1_1.0_224.tflite   --label_file /tmp/labels.txt   --image /tmp/grace_hopper.bmp
```

##### 執行完的結果如下，可以看到他成功推論出陸軍制服的選項
![](https://i.imgur.com/ByW3gxX.png)

![](https://i.imgur.com/nmC9Znk.png =200x200)

<br>

> ## 結論與補充: 
我們也可以把其他圖片放進去，他大致上的物件都還是能辨認出來的。

如果要用其他圖片來推論，我們可以把圖片丟到/tmp的資料夾下，我們可以執行下面的指令。
```
# 首先cd到桌面位置
cd /mnt/c/Users/你的使用者名字/Desktop/

# 假設貓的圖片名稱為123.jpg
cp 123.jpg /tmp

# 接下來推論的程式碼要改成
python3 label_image.py   --model_file /tmp/mobilenet_v1_1.0_224.tflite   --label_file /tmp/labels.txt   --image /tmp/123.jpg
```

#### Note: 如果想要把ubuntu系統裡的東西拿出來，也可以透過cp指令把檔案移動到 `/mnt/c/Users/你的使用者名字/Desktop/`
![](https://i.imgur.com/pNdo6cI.png)
![](https://i.imgur.com/xFPpznD.png =200x200)

#### 補充: ubuntu系統的根目錄在win10下的這個位置 
```
C:\Users\你的使用者名字\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs
```

##### 例如: 
貓: 

![](https://i.imgur.com/CfYxKkG.jpg)

![](https://i.imgur.com/bmU6ACk.png)

狗: 

![](https://i.imgur.com/2d7D3Jt.jpg)

![](https://i.imgur.com/kWggu0b.png)
