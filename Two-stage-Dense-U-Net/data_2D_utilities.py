"""data_2D_utilities.py: data preparation for 2D-Dense-U-Net."""

__author__      = "Peter"


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd

import SimpleITK as sitk
import csv
from glob import glob #用它可以查找符合自己目的的文件
from PIL import Image
import cv2
import math
from scipy.spatial.distance import pdist
import copy

#import itk
#from itkwidgets import view

## load_itk
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x(Formerly: x y z)
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


## normalization
#替换cv2.normalize,保证不同图像像素值映射的一致性。
#Dataset-2数据集：0对应0，255对应4095；
def normalization(hu_value, hu_min, hu_max):
    normal_value = np.uint8((hu_value - hu_min) / (hu_max - hu_min)*255)
    return normal_value

#替换cv2.normalize。
#椎体中心距离标签归一化处理，实际10个数据集最大距离为59，为保留余量最大设置为hu_max；
#图像为0的保持为0，图像有值的，找出本章图像中最大最小值，然后映射0到255，保证越靠近椎体的像素值越大；
def normalizationC(hu_value, hu_min, hu_max):
    if hu_value.max()> 3:
        hu_nonzero= np.nonzero(hu_value)
        temp = hu_value[hu_nonzero]
        minval = np.min(temp)
        maxval = np.max(temp)
        #print(minval)
        #print(maxval)
        hu_value[hu_value==0] = maxval
        #hu_value[hu_value <= 0] = minval
        hu_value = maxval + minval - hu_value
        normal_value = np.uint8((hu_value - minval) / (maxval - minval)*255)
    else:
            normal_value = hu_value
    return normal_value


#替换cv2.normalize,保证不同图像像素值映射的一致性。
#椎体中心距离标签归一化处理，实际10个数据集最大距离为59，为保留余量最大设置为hu_max；
#图像为0的保持为0，图像有值的，处理为“hu_max-当前像素值”，并归一化到255，保证越靠近椎体的像素值越大；
def normalizationCALL(hu_value, hu_min, hu_max):
    hu_value[hu_value==0] = hu_max
    #hu_value[hu_value <= 0] = minval
    hu_value = hu_max + hu_min - hu_value
    normal_value = np.uint8((hu_value - hu_min) / (hu_max - hu_min)*255)
    return normal_value

## mhd2jpg 和 mhd2png
def mhd2jpg(mhdPath,outFolder):#,windowsCenter,windowsSize
    
    """
    The function can output a group of jpg files by a specified mhd file.
    Args:
        mhdPath:mhd file path.
        outfolder:The folder that the jpg files are saved.
        windowsCenter:the CT windows center.
        windowsSize:the CT windows size.
    Return:void

    """
    image = sitk.ReadImage(mhdPath)
    img_data = sitk.GetArrayFromImage(image)
#     max_norm = img_data.max()
#     min_norm = img_data.min()
    #channel根据要提取的面设置，0-横断面，1-冠状面，2-矢状面
    channel = img_data.shape[2]

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)


    #low = windowsCenter-windowsSize/2
    #high = windowsCenter+windowsSize/2

    for s in range(channel):
        slicer = img_data[s,:,:] #横断面获取
        #slicer = img_data[:,s,:] #冠状面获取
        #slicer = img_data[...,s] #矢状面获取
        #slicer[slicer<low] = low
        #slicer[slicer>high] = high
        #slicer = slicer-low
        img = normalization(slicer,0,4095)
        cv2.imwrite(os.path.join(outFolder,str('%03d' % s)+'.jpg'),img)

def mhd2png(mhdPath,outFolder,Threshold):#,windowsCenter,windowsSize
    
    """
    The function can output a group of jpg files by a specified mhd file.
    Args:
        mhdPath:mhd file path.
        outfolder:The folder that the jpg files are saved.
        windowsCenter:the CT windows center.
        windowsSize:the CT windows size.
    Return:void

    """
    image = sitk.ReadImage(mhdPath)
    img_data = sitk.GetArrayFromImage(image)
#     max_norm = img_data.max()
#     min_norm = img_data.min()
    #channel根据要提取的面设置，0-横断面，1-冠状面，2-矢状面
    channel = img_data.shape[0]

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)


    #low = windowsCenter-windowsSize/2
    #high = windowsCenter+windowsSize/2

    #3D的数据获取时以16的倍数记录，后续的切片不包含有效椎骨,所以不要，保证对齐;
    #2D的数据不需要以16为倍数，故注释掉
    #channel = (channel//16)*16
    for s in range(channel):
        slicer = img_data[s,:,:] #横断面获取
        #slicer = img_data[:,s,:] #冠状面获取
        #slicer = img_data[...,s] #矢状面获取
        #slicer[slicer<low] = low
        #slicer[slicer>high] = high
        #slicer = slicer-low
        img = normalization(slicer,0,4095)
        if Threshold:
            (T,img) = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(outFolder,str('%03d' % s)+'.png'),img)


## mhd2pngR(wiht resize)
def mhd2pngR(mhdPath,outFolder,Threshold,img_col,img_row,normmin,normmax):#,windowsCenter,windowsSize
    
    """
    The function can output a group of png files by a specified mhd file.
    Args:
        mhdPath:mhd file path.
        outfolder:The folder that the png files are saved.
        Threshold:
        img_col: column after resize
        img_row: row after resize
        nornmin: min for img normalization
        nornmax: max for img normalization
    Return:void

    """
    image = sitk.ReadImage(mhdPath)
    img_data = sitk.GetArrayFromImage(image)
    img_shape = img_data.shape
    max_norm = img_data.max()
    min_norm = img_data.min()
    #channel根据要提取的面设置，0-横断面，1-冠状面，2-矢状面
    channel = img_data.shape[0]

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    #3D的数据获取时以16的倍数记录，后续的切片不包含有效椎骨,所以不要，保证对齐;
    #2D的数据不需要以16为倍数，故注释掉
    #channel = (channel//16)*16
    for s in range(channel):
        slicer = img_data[s,:,:] #横断面获取
        #slicer = img_data[:,s,:] #冠状面获取
        #slicer = img_data[...,s] #矢状面获取
        #slicer[slicer<low] = low
        #slicer[slicer>high] = high
        #slicer = slicer-low
        img = normalization(slicer,normmin,normmax)
        imgR = cv2.resize(img,dsize=None,fx=img_col/img_shape[2],fy=img_row/img_shape[1],interpolation=cv2.INTER_LINEAR)
        if Threshold:
            (T,img) = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(outFolder,str('%03d' % s)+'.png'),imgR)

## mhd2pngC(wiht resize for centroid Dense)
def mhd2pngC(mhdPath,outFolder,Threshold,img_col,img_row):#,windowsCenter,windowsSize
    
    """
    The function can output a group of jpg files by a specified mhd file.
    Args:
        mhdPath:mhd file path.
        outfolder:The folder that the jpg files are saved.
        windowsCenter:the CT windows center.
        windowsSize:the CT windows size.
    Return:void

    """
    image = sitk.ReadImage(mhdPath)
    img_data = sitk.GetArrayFromImage(image)
    img_shape = img_data.shape
#     max_norm = img_data.max()
#     min_norm = img_data.min()
    #channel根据要提取的面设置，0-横断面，1-冠状面，2-矢状面
    channel = img_data.shape[0]

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)


    #low = windowsCenter-windowsSize/2
    #high = windowsCenter+windowsSize/2

    #3D的数据获取时以16的倍数记录，后续的切片不包含有效椎骨,所以不要，保证对齐;
    #2D的数据不需要以16为倍数，故注释掉
    #channel = (channel//16)*16
    for s in range(channel):
        slicer = img_data[s,:,:] #横断面获取
        #slicer = img_data[:,s,:] #冠状面获取
        #slicer = img_data[...,s] #矢状面获取
        #slicer[slicer<low] = low
        #slicer[slicer>high] = high
        #slicer = slicer-low
        imgR = cv2.resize(slicer,dsize=None,fx=img_col/img_shape[2],fy=img_row/img_shape[1],interpolation=cv2.INTER_NEAREST)
        img = normalizationCALL(imgR,0,50)
        if Threshold:
            (T,img) = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(outFolder,str('%03d' % s)+'.png'),img)


## mhd2pngCZEnhance(wiht resize for centroid Dense0412)
def mhd2pngCZEnhance(mhdPath,outFolder,Threshold,img_col,img_row):#,windowsCenter,windowsSize
    
    """
    The function can output a group of jpg files by a specified mhd file.
    Args:
        mhdPath:mhd file path.
        outfolder:The folder that the jpg files are saved.
        windowsCenter:the CT windows center.
        windowsSize:the CT windows size.
    Return:void

    """
    image = sitk.ReadImage(mhdPath)
    img_data = sitk.GetArrayFromImage(image)
    print(np.max(img_data))
    img_shape = img_data.shape
#     max_norm = img_data.max()
#     min_norm = img_data.min()
    #channel根据要提取的面设置，0-横断面，1-冠状面，2-矢状面
    channel = img_data.shape[0]

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)


    #low = windowsCenter-windowsSize/2
    #high = windowsCenter+windowsSize/2

    #3D的数据获取时以16的倍数记录，后续的切片不包含有效椎骨,所以不要，保证对齐;
    #2D的数据不需要以16为倍数，故注释掉
    #channel = (channel//16)*16
    for s in range(channel):
        slicer = img_data[s,:,:] #横断面获取
        #slicer = img_data[:,s,:] #冠状面获取
        #slicer = img_data[...,s] #矢状面获取
        #slicer[slicer<low] = low
        #slicer[slicer>high] = high
        #slicer = slicer-low
        imgR = cv2.resize(slicer,dsize=None,fx=img_col/img_shape[2],fy=img_row/img_shape[1],interpolation=cv2.INTER_NEAREST)
        img = normalization(imgR,0,50)
        if Threshold:
            (T,img) = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(outFolder,str('%03d' % s)+'.png'),img)

        
# 将png以2k+1形式打包成bin文件
def read_png2bin(path,k):

    # Read first png as center png
    img = cv2.imread(path,0)

    # Get png from before and after, then fill as a 2k+1 tf array
    imggroup = np.expand_dims(img, axis=2)
    imgZero = np.zeros_like(img)
    imgZero = np.expand_dims(imgZero, axis=2)

    casepath = os.path.split(path)
    pathNO = int(os.path.splitext(casepath[1])[0])
    for i in range(1,k+1,1):
        pathnew = os.path.join(casepath[0],str("%03d" % (pathNO - i) +".png"))
    #         print(pathnew)
        if os.path.exists(pathnew):
            img = cv2.imread(pathnew,0)
            img = np.expand_dims(img, axis=2)
            imggroup = np.concatenate([img,imggroup], axis=2)
        else:
            imggroup = np.concatenate([imgZero,imggroup], axis=2)
    for i in range(1,k+1,1):
        pathnew = os.path.join(casepath[0],str("%03d" % (pathNO + i) +".png"))
    #         print(pathnew)
        if os.path.exists(pathnew):
            img = cv2.imread(pathnew,0)
            img = np.expand_dims(img, axis=2)
            imggroup = np.concatenate([imggroup,img], axis=2)
        else:
            imggroup = np.concatenate([imggroup,imgZero], axis=2)
    outPath = os.path.join(casepath[0], str('%03d' % pathNO) + ".bin")
    imggroup.tofile(outPath)
    
    return imggroup

# -*- coding: utf-8 -*-  
# 获取所有目录及其子目录下的mhd文件

def listdir(path,Extension):
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1]==Extension:  
            list_name.append(file_path)
    return list_name


## DenseLabel：距离Dense标签生成
def DenseLabel(filelabel,filecentroid,fileDense):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filelabel)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x(Formerly: x y z)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    # Read the spacing along each dimension
    npSpacing = np.array(list(itkimage.GetSpacing()))

    Num_Vertebra = np.unique(numpyImage)

    caseCentroids = np.loadtxt(filecentroid,delimiter = ',')
     
    #计算所有椎体上的像素点以及与它对应所在椎体中心点的距离，并替换该像素点原有的像素值
    FlagPrint = 0
    for z in range(0,numpyImage.shape[0],1):
        for y in range(0,numpyImage.shape[1],1):
            for x in range(0,numpyImage.shape[2],1):
                if numpyImage[z,y,x] != 0:
                    VertebraNth = 0
                    for centroid_Num in Num_Vertebra[1:]:
                        if numpyImage[z,y,x] == centroid_Num:
                            coordIJK = [x,y,z]
                            coord = coordIJK * npSpacing
                            centroidNth = caseCentroids[VertebraNth] * npSpacing
                            X = np.vstack([coord,centroidNth])
                            dist = int(pdist(X))
                            if FlagPrint == 0:
                                if z%100==0:
                                    print(coordIJK,coord,centroidNth,dist)
                                    FlagPrint += 1
                            if z%100==1:
                                FlagPrint = 0
                            itkimage.SetPixel(x,y,z,dist)

                        VertebraNth = VertebraNth + 1
    path_Dense = os.path.dirname(fileDense)
    if not os.path.exists(path_Dense):
        os.makedirs(path_Dense)
    
    sitk.WriteImage(itkimage,fileDense)

## DenseLabel：距离Dense标签生成，Z方向作了区分，使椎骨中心所在slice更突出
def DenseLabelZEnhance(filelabel,filecentroid,fileDense):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filelabel)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x(Formerly: x y z)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    # Read the spacing along each dimension
    npSpacing = np.array(list(itkimage.GetSpacing()))

    Num_Vertebra = np.unique(numpyImage)

    caseCentroids = np.loadtxt(filecentroid,delimiter = ',')
    
    # Get height of each vertebrae
    Height_Vertebra = []
    for m in Num_Vertebra[1:]:
        Height = 0
        for n in range(numpyImage.shape[0]):
            if m == np.max(numpyImage[n,:,:]):
                Height += 1
        Height_Vertebra.append(Height)
    # 计算所有椎体上像素值不为0的像素点与它对应所在椎体中心点的距离，并替换该像素点原有的像素值
    # 在z轴方向上添加增强，以更加明显的区分z轴方向的椎体
    FlagPrint = 0
    for z in range(0,numpyImage.shape[0],1):
        for y in range(0,numpyImage.shape[1],1):
            for x in range(0,numpyImage.shape[2],1):
                if numpyImage[z,y,x] != 0:
                    VertebraNth = 0
                    for centroid_Num in Num_Vertebra[1:]:
                        if numpyImage[z,y,x] == centroid_Num:
                            coordIJK = [x,y,z]
                            coord = coordIJK * npSpacing
                            centroidNth = caseCentroids[VertebraNth] * npSpacing
                            X = np.vstack([coord,centroidNth])
                            dist_raw = pdist(X)
                            dist_flip = 50 - dist_raw
                            if dist_flip > 50:
                                dist_flip = 50
                            if dist_flip < 0:
                                dist_flip = 0
                            dist_slice = np.abs((z - caseCentroids[VertebraNth,2]))
                            HeihtVertebra = Height_Vertebra[VertebraNth]
                            if dist_slice > HeihtVertebra/2:
                                dist_slice = HeihtVertebra/2
                            
                            FactorZ = 1 - np.tan((dist_slice/HeihtVertebra)*(math.pi/2))
                            
                            dist = int(dist_flip*FactorZ)
                            if FlagPrint == 0:
                                if z%100==0:
                                    print(coordIJK,coord,centroidNth,dist)
                                    FlagPrint += 1
                            if z%100==1:
                                FlagPrint = 0
                            itkimage.SetPixel(x,y,z,dist)

                        VertebraNth = VertebraNth + 1
    
    path_Dense = os.path.dirname(fileDense)
    if not os.path.exists(path_Dense):
        os.makedirs(path_Dense)
    
    sitk.WriteImage(itkimage,fileDense)