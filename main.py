#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.utils import np_utils
%matplotlib inline

#------------変数宣言------------
img_size = 32
#csvのファイルパスを記載
csv_filepath = 'filename.csv'
#水増し処理の種類を記載
img_add = 9
#------------処理ここから------------

#csvからファイルパスを読み込む
csv_file = pd.read_csv(csv_filepath, encoding="UTF-8")#もしくわUTF-8

#csvに記載しているファイルパス、ラベル、総数を格納
fnames = csv_file['filepath']
labels = csv_file['label']
fnames_total = len(fnames)

#データセットを格納する変数の初期化
img_train = np.zeros((fnames_total,img_size,img_size,3))

#データセットの格納
#前処理する場合はここでやる
for i,fname in enumerate(fnames):
    #解像度を変えて、画像を読み込む
    temp_img = load_img(fname, target_size=(img_size,img_size))

    #画像を配列に変換
    img_train[i] = img_to_array(temp_img)


#%%
#ちゃんと読み込まれたか確認する
#画像をランダムに表示
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3,3,i+1)
    j = np.random.randint(fnames_total)
    print(j,labels[j])
    plt.imshow(img_train[j]/255)

#%%
#水増し処理を行う
#ImageDataGeneratorの生成
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
#    rescale=1. / 255,    #正規化
    rotation_range=180,    #回転
    vertical_flip=True,    #上下#反転
    horizontal_flip=True,    #左右反転
    brightness_range=[0.3, 1.0]    #明度調整 0だと暗く、1だと明るい
)
#ImageDataGeneratorを通す
gen = datagen.flow(img_train, batch_size=fnames_total)


#%%
print(img_train.shape)
print(labels.shape)
img_train_pro = list(img_train)
label_train_pro = list(labels)

for i in range(img_add):
    batches = next(gen)
    print(gen)  # (NumBatches, Height, Width, Channels) の4次元データを返す。
    img_train_pro = np.vstack((img_train_pro, batches))
print(img_train_pro.shape)
#print(label_train_pro.shape)

#正規化する
img_train_pro = img_train_pro/255
#正解ラベルをOne-Hot表現に変換
#label_train = np_utils.to_categorical(labels,10)

#%%
#ちゃんと読み込まれたか確認する
#画像をランダムに表示
plt.figure(figsize=(16, 16))
for i in range(36):
    plt.subplot(6,6,i+1)
#    j = np.random.randint(fnames_total*(img_add+1))
    j = 940+i
    print(j)
    plt.imshow(img_train_pro[j])



#%%
#モデルを構築
model=Sequential()

model.add(Conv2D(img_size,(3,3),padding='same',input_shape=(img_size,img_size,3)))
model.add(Activation('relu'))
model.add(Conv2D(img_size,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(img_train,label_train,batch_size=128,nb_epoch=20,verbose=1,validation_split=0.1)

#モデルと重みを保存
json_string=model.to_json()
open('Anime_cnn.json',"w").write(json_string)
model.save_weights('Anime_cnn.h5')

#モデルの表示
model.summary()

#%%
