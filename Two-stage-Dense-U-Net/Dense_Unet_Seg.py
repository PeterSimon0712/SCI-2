from __future__ import print_function
"""Dense_Unet_Seg.py: 3D-Dense-U-Net for segmentation."""

__author__      = "Peter"

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

import SimpleITK as sitk
import xml.etree.ElementTree as ET
import pandas as pd

# 设置用于训练的GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#自动根据GPU个数，选择CPU建立读取图片的线程数
auto = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 1
BUFFER_SIZE = 300
EPOCHS = 50

project_name = '3D-Dense-Unet-Ins'
img_rows = 128
img_cols = 112
img_depth = 80
smooth = 1.

# Path of evaluate the results of segmentation
ToolPath = '/home/shu/Github/EvaluateSegmentation/builds/Ubuntu/EvaluateSegmentation'

#在cases的范围内创建乱序
def permutation(cases,labels,seed):
    np.random.seed(seed)
    index = np.random.permutation(len(cases))
    print(index)
    #使用同一个乱序进行排序，cases和labels继续一一对应
    cases = np.array(cases)[index]
    labels = np.array(labels)[index]
    print(cases[-5:])
    print(labels[-5:])
    return cases,labels

def read_npy(path):
    
    case = tf.io.read_file(path)
    case = tf.io.decode_raw(case,tf.uint8)
    data = tf.reshape(case,[img_depth,img_rows,img_cols]) #拼接成你要的维度
    
    return data

# 定义归一化函数
def normal(case, label):
    case = tf.cast(case, tf.float32)/127.5 - 1
    label = tf.cast(label, tf.int32)/255
    return case, label

def load_image_train(case_path, label_path):
    case = read_npy(case_path)
    label = read_npy(label_path)
    
#     img, mask = crop_img(img, mask)
    
#     if tf.random.uniform(())>0.5:
#         img = tf.image.flip_left_right(img)
#         mask = tf.image.flip_left_right(mask)
    case, label = normal(case, label)
    
    return case, label

def load_image_val(case_path, label_path):
    case = read_npy(case_path)
    label = read_npy(label_path)
    
#     img = tf.image.resize(img, (256, 256))
#     mask = tf.image.resize(mask, (256, 256))
    
    case, label = normal(case, label)
    
    return case, label

# 创建模型：3D-Dense-U-Net

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
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model

K.set_image_data_format('channels_last')

def get_unet():
    inputs = Input((img_depth, img_rows, img_cols, 1))
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)
    
    conv41 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=4)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=4)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc62), conv32], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    # model.summary(positions = [.3, .6, .7, 1.])
    
    # # Open the file
    # with open('3D model report.txt','w') as fh:
    # # Pass the file handle in as a lambda function to make it callable
    # model.summary(positions = [.3, .6, .7, 1.],print_fn=lambda x: fh.write(x + '\n'))
    
    # plot_model(model, to_file='3D model.png')

    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# train
def train(dataset_train,dataset_val,EPOCHS,STEPS_PER_EPOCH,VALIDATION_STEPS):
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
    csv_logger = CSVLogger(os.path.join(log_dir,  project_name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S") + '.csv'), separator=',', append=False)
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

# predict: 预测所有椎体并保存为nii文件到相应文件夹，用于后续结果评估
# weight_dir: 权重路径
# dataset_test: 测试数据集
# outPath: 保存预测后nii文件夹路径

def predict(weight_dir,dataset_test,outPath,Thres):
    
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
        print(image.shape,mask.shape)
        pred_mask = model.predict(image, batch_size=1, verbose=1)
        # pred_mask = tf.argmax(pred_mask, axis=-1)
        # pred_mask = pred_mask[..., tf.newaxis]

        pred_mask = np.squeeze(pred_mask, axis=0)
        image = np.squeeze(image, axis=0)
        mask = np.squeeze(mask, axis=0)

        image = ((image+1)*127.5).astype(np.uint8)
        mask = (mask*255.).astype(np.uint8)
        # 原来设置 0.5/0.5分界
        # 现在设置 0.9/0.1分界
        # pred_mask = np.around(pred_mask, decimals=0)
        # 根据椎体位置，确定分割阈值
        # if VertebralNth < 10:
        #    pred_mask = np.around(pred_mask, decimals=0)
        # else:
        #    pred_mask[pred_mask>=0.9] = 1
        #    pred_mask[pred_mask<0.9] = 0

        pred_mask[pred_mask>=Thres] = 1
        pred_mask[pred_mask<Thres] = 0
            
        pred_mask = (pred_mask*255.).astype(np.uint8)
        
        pred_mask_R = RemoveSmallObject(pred_mask)
        
        outPathnorm = outPath + 'norm/'
        if not os.path.exists(outPathnorm):
            os.makedirs(outPathnorm)
            
        outPathrmvsml = outPath + 'rmvsml/'
        if not os.path.exists(outPathrmvsml):
            os.makedirs(outPathrmvsml)
        
        Savenii(pred_mask,outPathnorm + str('%03d' % VertebralNth) + ".nii")
        Savenii(pred_mask_R,outPathrmvsml + str('%03d' % VertebralNth) + ".nii")
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
    
    #乱序
    cases_train,labels_train = permutation(cases_train,labels_train,2021)
    
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

def dataset_test_build(casesPath,labelsPath):
    casestest = sorted(glob(casesPath))
    labelstest = sorted(glob(labelsPath))
    # 检查顺序是否对应
    print(casestest[-3:],labelstest[-3:])
    testset = tf.data.Dataset.from_tensor_slices((casestest, labelstest))
    #自动根据GPU个数，选择CPU建立读取图片的线程数
    auto = tf.data.experimental.AUTOTUNE
    dataset_test = testset.map(load_image_val,num_parallel_calls = auto)
    dataset_test = dataset_test.cache().batch(BATCH_SIZE).prefetch(auto)
    return dataset_test

## 预测结果保存为nii文件
import SimpleITK as sitk
def Savenii(pred_mask,outPath):
    out = sitk.GetImageFromArray(pred_mask)
    out.SetSpacing([1,1,1])
    #out.SetOrigin(itkimage.GetOrigin())
    sitk.WriteImage(out,outPath)

## 去除小连通区域
from skimage import measure
from skimage import morphology

# npraw:0-background,255-segmented mask
def RemoveSmallObject(numpyImage):
    numpyImage[numpyImage==255] = 1
    imgprocess = numpyImage.astype(bool)
    e = morphology.remove_small_objects(imgprocess, 500,connectivity=50)
    imgResult = e.astype(np.uint8)
    imgResult[imgResult == 1] = 255
    return imgResult

# 评估分割结果，并输出到文件夹
def EvalSegResult(GTPath,PredPath,outPath,EvalDict):

    cmduse = ""
    for EvalItem in EvalDict:
        cmduse = cmduse + EvalItem + ","
    cmd = ToolPath + ' ' + GTPath + ' ' + PredPath + ' ' + '-xml' + ' ' + outPath + ' ' + '-use' + ' '+ cmduse
    result = os.system(cmd)
    # print(cmd)
    # print(result)
    return result

# 汇总评估结果    
def GetEvalResult(xmlPath,EvalDict):
    tree = ET.ElementTree(file=xmlPath)
    for EvalItem in EvalDict:
        for elem in tree.iterfind('metrics/' + EvalItem):
            EvalDict[EvalItem] = float(elem.get('value'))
    return EvalDict

# 根据文件夹，评估测试结果
def EvalResults(GTFolder,PredFolder,EvalDict):
    GTPaths = sorted(glob(GTFolder + '/*.nii'))
    PredPaths = sorted(glob(PredFolder + '/*.nii'))
    print(len(GTPaths),len(PredPaths))
    
    EvalResults = pd.DataFrame(data=None,columns = EvalDict)
    for j in range(0,len(GTPaths),1):
        
        outPath = PredFolder + '/Result/'
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        xmlPath = outPath + str('%03d' % j) + '.xml'
        # print(xmlPath)
        
        EvalSegResult(GTPaths[j],PredPaths[j],xmlPath,EvalDict)
        EvalDict = GetEvalResult(xmlPath,EvalDict)
        dfEval=pd.DataFrame([EvalDict])
        EvalResults = EvalResults.append(dfEval)
        
    print(EvalResults)
    EvalResults.to_csv(PredFolder + '/Result/ResultALL' + '.csv')
    
    return EvalResults
