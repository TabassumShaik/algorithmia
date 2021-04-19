import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model

import Algorithmia

client = Algorithmia.client()
model_file_path = "data://<USERNAME>/digits_recognition/digits-classifier.joblib"

def upload():
    img_path = (
                client.algo("util/SmartImageDownloader/")
                .pipe(query["url"])
                .result["savePath"][0]
            )
            img_fpath = client.file(img_path).getFile().name
            return img_fpath


def predict_cls():
    path=upload()
    MODEL_WEIGHTS = 'static/model/vgg_50epochs.h5'
    reconstructed_model=keras.models.load_model(MODEL_WEIGHTS)
    img=tf.keras.preprocessing.image.load_img(path)
    new_img=img.resize((224,224))
    input_arr=tf.keras.preprocessing.image.img_to_array(new_img)
    input_arr=np.array([input_arr])
    result=reconstructed_model.predict(input_arr)
    if result[0][0]>=0.5:
        res="Cancer"
    else:
        res="No Cancer"
    return (res,result[0][0])


if __name__=="__main__":
    predict_cls()