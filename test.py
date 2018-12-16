#%%
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

#画像読み込み
#temp_img=load_img(".\\test_image\\akiyama.jpg",target_size=(32,32))
temp_img=load_img(".\\test_image\\akiyama4.jpg",target_size=(32,32))

#画像を配列に変換し0-1で正規化
temp_img_array=img_to_array(temp_img)
temp_img_array=temp_img_array.astype('float32')/255.0
temp_img_array=temp_img_array.reshape((1,32,32,3))

#学習済みのモデルと重みを読み込む
json_string=open('Anime_cnn.json').read()
model=model_from_json(json_string)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights('Anime_cnn.h5')

#モデルを表示
model.summary()

#画像を予想
img_pred=model.predict_classes(temp_img_array)
print('\npredict_classes=',img_pred)

plt.imshow(temp_img)
plt.title('pred:{}'.format(img_pred))
plt.show()