#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

%matplotlib inline

#------------変数宣言------------
#画像の解像度
image_size = 256
#バッチサイズ
batch_size = 8
#学習データの格納先
train_dir = '.\\animeface-character-dataset\\thumb_ten\\train'
test_dir = '.\\animeface-character-dataset\\thumb_ten\\test'
#------------処理ここから------------

#ジュネレータの生成
train_datagen = ImageDataGenerator(
    rescale=(1./255), 
    vertical_flip=True,     #上下反転
    horizontal_flip=True,   #左右反転
    rotation_range=180,       #回転
    brightness_range=[0.3, 1.0] # 明度の変更
)
test_datagen = ImageDataGenerator(rescale=1./255)

#ジュネレータに読み込ませるディレクトリの指定
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical"
)

num_categories=10

#モデルの生成
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_categories, activation="sigmoid"))

adam = Adam(lr=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#チェックポイントの生成
checkpoint_cb = ModelCheckpoint(".\\snapshot\\{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)

#num_train_images = 946
num_train_images = 853
num_test_images = 93

history = model.fit_generator(
    train_generator,
    steps_per_epoch=128,
    epochs=300,
    validation_data=test_generator,
    validation_steps=batch_size,
    callbacks=[checkpoint_cb]
)

# 精度の履歴のプロット
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()

# 損失の履歴をプロット
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'], loc='lower right')
plt.show()

#%%
