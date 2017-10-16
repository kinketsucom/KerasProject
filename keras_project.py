import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,  Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np# 元となる画像の読み込み

# opencv画像入力用
from IPython.display import display, Image

def display_cv_image(image, format='.JPG'):
    decoded_bytes = cv2.imencode(format, image)[1].tobytes()
    display(Image(data=decoded_bytes))

def readyForGuess(img_name):
    img = cv2.imread( "./img/"+img_name+".jpg", cv2.IMREAD_GRAYSCALE)
    width = 100
    size = (width,width)
    #Create Small Size Image
    length=size[0]*size[1]
    img = cv2.resize(img, size)
    img_dat = img/255
    img_dat = img_dat.reshape(1,length)
    return img_dat

# 元となる画像の読み込み
bef_img = cv2.imread( './img/bef.JPG', cv2.IMREAD_GRAYSCALE)
aft_img = cv2.imread('./img/aft.JPG', cv2.IMREAD_GRAYSCALE)

width = 40
size = (3*width,4*width)
#Create Small Size Image
bef_img = cv2.resize(bef_img, size)
aft_img = cv2.resize(aft_img, size)

bef_dat = bef_img/255
aft_dat = aft_img/255
hoge = bef_dat
length=size[0]*size[1]
bef_dat = bef_dat.reshape(1,length)
aft_dat = aft_dat.reshape(1,length)

NN = Sequential() #空のニューラルネット
NN.add(Dense(length, input_dim=length))
NN.add(Activation('relu'))
NN.add(Dense(length))
NN.summary()
NN.compile(optimizer='adam',loss='mse',metrics = ['accuracy'])

NN.fit(aft_dat,bef_dat, epochs=10)


img_name = "aft" #ファイル名のみ
guess_dat = readyForGuess(img_name)
result = NN.predict(guess_dat)


#正規化準備
result_max = result.max()
result_min = result.min()
# 0-255に正規化
result = (result - result_min) * (255 - 0) / (result_max - result_min)

#intに変換する.グレースケールはintだから
result = np.floor(result)
result = result.astype(np.int)
result = result.reshape(size[0],-1)

cv2.imwrite("./img/output.jpg", result)
# 元となる画像の読み込み
img = cv2.imread("./img/output.jpg")
print("推測画像")
display_cv_image(img)

print("これから")
display_cv_image(aft_img)
print("これを推測")
display_cv_image(bef_img)
