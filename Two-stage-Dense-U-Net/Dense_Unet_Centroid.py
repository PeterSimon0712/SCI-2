from __future__ import print_function
"""Dense_Unet_Centroid.py: 2D-Dense-U-Net for localization."""

__author__      = "Peter"

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import cv2
import pydot


#自动根据GPU个数，选择CPU建立读取图片的线程数
auto = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 1
BUFFER_SIZE = 300
EPOCHS = 30

project_name = '2D-Dense-Unet-Centroid'
img_rows = 128
img_cols = 128
img_depth = 9
smooth = 1.

def read_npy(path):
    
    case = tf.io.read_file(path)
    case = tf.io.decode_raw(case,tf.uint8)
    data=tf.reshape(case,[img_rows,img_cols,img_depth]) #拼接成你要的维度
    
    return data

def read_png_label(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img

# 定义归一化函数
def normal(case, label):
    case = tf.cast(case, tf.float32)/127.5 - 1
    label = tf.cast(label, tf.int32)/255
    return case, label

def load_image_train(img_path, mask_path):
    img = read_npy(img_path)
    mask = read_png_label(mask_path)
    
    # img, mask = crop_img(img, mask)
    
    # if tf.random.uniform(())>0.5:
    # img = tf.image.flip_left_right(img)
    # mask = tf.image.flip_left_right(mask)
    img, mask = normal(img, mask)
    
    return img, mask

def load_image_val(img_path, mask_path):
    img = read_npy(img_path)
    mask = read_png_label(mask_path)
    # img = tf.image.resize(img, (256, 256))
    # mask = tf.image.resize(mask, (256, 256))
    img, mask = normal(img, mask)
    
    return img, mask


# 创建2D-Dense-U-Net模型，训练并预测

import os
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import datetime

np.random.seed(1337)
import tensorflow as tf
tf.random.set_seed(1337)

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model

K.set_image_data_format('channels_last')

def get_unet():
    inputs = Input((img_rows, img_cols, img_depth))
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=3)
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)

    conv21 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=3)
    conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

    conv31 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=3)
    conv32 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)

    conv41 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=3)
    conv42 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)

    conv51 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=3)
    conv52 = Conv2D(512, (3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=3)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
    conv61 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=3)
    conv62 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=3)


    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc62), conv32], axis=3)
    conv71 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=3)
    conv72 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
    conv81 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=3)
    conv82 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
    conv91 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=3)
    conv92 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=3)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.summary()
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# train
def train_2D(dataset_train,dataset_val,EPOCHS,STEPS_PER_EPOCH,VALIDATION_STEPS):
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S") + '.h5'), monitor='val_loss', save_best_only=True)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir,  project_name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S") + '.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    # model.fit(dataset_train, batch_size=1, epochs=50, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])
    history = model.fit(dataset_train,
                        epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=dataset_val,callbacks=[model_checkpoint, csv_logger])

    print('-'*30)
    print('Training finished')
    print('-'*30)

# Predict: 预测所有椎体并保存为png文件到相应文件夹，用于后续结果评估
# weight_dir: 权重路径
# dataset_test: 测试数据集
# outPath: 保存预测后png文件夹路径
import cv2
def predict_2D(weight_dir,dataset_test,outPath):
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    
    model = get_unet()
    model.load_weights(weight_dir)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    VertebralNth = 0
    for image, mask in dataset_test.as_numpy_iterator():
        # print(image.shape,mask.shape)
        pred_mask = model.predict(image, batch_size=1, verbose=1)
        # pred_mask = tf.argmax(pred_mask, axis=-1)
        # pred_mask = pred_mask[..., tf.newaxis]

        pred_mask = np.squeeze(pred_mask, axis=0)
        image = np.squeeze(image, axis=0)
        mask = np.squeeze(mask, axis=0)

        image = ((image+1)*127.5).astype(np.uint8)
        mask = (mask*255.).astype(np.uint8)
        # pred_mask = np.around(pred_mask, decimals=0)
        pred_mask = (pred_mask*255.).astype(np.uint8)

        if not os.path.exists(outPath):
            os.makedirs(outPath)
        
        img = pred_mask
        cv2.imwrite(os.path.join(outPath,str('%03d' % (VertebralNth))+'.png'),img)
        
        # Savenii(pred_mask,outPath + str('%03d' % VertebralNth) + ".nii")
        VertebralNth+=1
        
    print('-'*30)
    print('Prediction finished')
    print('-'*30)
    # 返回最后一个椎体数据
    return image,mask,pred_mask

## dataset_train_build
def dataset_train_build(casesPath,labelsPath):
    cases_train = sorted(glob(casesPath))
    labels_train = sorted(glob(labelsPath))
    
    # 检查顺序是否对应
    print(cases_train[-3:],labels_train[-3:])
    
    data_train = tf.data.Dataset.from_tensor_slices((cases_train, labels_train))
    dataset_train = data_train.map(load_image_train, num_parallel_calls = auto)
    dataset_train = dataset_train.cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(auto)
    
    count_train = len(cases_train)
    return dataset_train,count_train

## dataset_val_build
def dataset_val_build(casesPath,labelsPath):
    cases_val = sorted(glob(casesPath))
    labels_val = sorted(glob(labelsPath))
    
    # 检查顺序是否对应
    print(cases_val[-3:],labels_val[-3:])
    
    data_val = tf.data.Dataset.from_tensor_slices((cases_val, labels_val))
    dataset_val = data_val.map(load_image_val, num_parallel_calls = auto)
    dataset_val = dataset_val.cache().batch(BATCH_SIZE).prefetch(auto)
    
    count_val = len(cases_val)
    return dataset_val,count_val


## dataset_test_build
def dataset_test_build(casesPath,labelsPath):
    casestest = sorted(glob(casesPath))
    labelstest = sorted(glob(labelsPath))
    
    # 检查顺序是否对应
    print(casestest[-3:],labelstest[-3:])
    testset = tf.data.Dataset.from_tensor_slices((casestest, labelstest))

    dataset_test = testset.map(load_image_val,num_parallel_calls = auto)
    dataset_test = dataset_test.cache().batch(BATCH_SIZE).prefetch(auto)
    return dataset_test