"""result_2D_utilities.py: result process i.e. aggragate the vertebral centroids after the prediction of  2D-Dense-U-Net."""

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
import pylab as pl
import scipy.signal as signal
#import itk
#from itkwidgets import view


# 预测后的图片，提取椎体中心点
## 距离椎体中心点最近的横断面提取，即提取预测的各个椎体中心点的z坐标
### GetCentroidZ

##根据预测的图片所在的位置，提取椎体中心点z轴坐标
def GetCentroidZ(PredPNGPaths):
    maxList = []
    ## 获取每个预测图片的最大像素值，即该图片中距离椎体中心最近的点
    for pngPath in PredPNGPaths:
        img = cv2.imread(pngPath)
        maxC = img.max()
#         means,stddev=cv2.meanStdDev(img)
#         maxList.append(means[0][0])
#         npimg = np.array(img)
#         npimgvalid = npimg[npimg>10]
#         if len(npimgvalid)!=0:
#             meanC = maxC*len(npimgvalid)
#         else:meanC=0
        maxList.append(maxC)
#     print(maxList)
    ## SG滤波maxList曲线，并获取每个椎体距离椎体中心最近的横断面，即z值
    x=np.array(maxList)
    xf = signal.savgol_filter(x,7,1)
    
    maxval = xf[signal.find_peaks(xf, height=180,distance=10,prominence=5,width=1)[0]]
    index_max = signal.find_peaks(xf, height=180,distance=10,prominence=5,width=1)[0]
    # 排除不需要的元素,距离Dense小于100的值
#     idx = 0
#     delete_num = 0
#     for i in maxval:
#         if i < 180:
#             maxval = np.delete(maxval,idx-delete_num)
#             index_max = np.delete(index_max,idx-delete_num)
#             delete_num += 1
#         idx+=1
#         print(i,idx)
    
    plt.figure(figsize=(16,4))
    plt.plot(np.arange(len(xf)),xf)

    print(maxval)
    print(index_max)

    plt.plot(index_max,maxval,'o')
    plt.show()
    
    return x,xf,maxval,index_max


### GetCentroidZEnhance
##根据预测的图片所在的位置，提取椎体中心点z轴坐标
def GetCentroidZEnhance(PredPNGPaths):
    maxList = []
    ## 获取每个预测图片的最大像素值，即该图片中距离椎体中心最近的点
    for pngPath in PredPNGPaths:
        img = cv2.imread(pngPath)
        maxC = img.max()
#         means,stddev=cv2.meanStdDev(img)
#         maxList.append(means[0][0])
#         npimg = np.array(img)
#         npimgvalid = npimg[npimg>10]
#         if len(npimgvalid)!=0:
#             meanC = maxC*len(npimgvalid)
#         else:meanC=0
        maxList.append(maxC)
#    np.savetxt(rootPath+"/maxList.csv", maxList, delimiter=',')
#     print(maxList)
    ## SG滤波maxList曲线，并获取每个椎体距离椎体中心最近的横断面，即z值
    x=np.array(maxList)
    xf = signal.savgol_filter(x,7,1)
    
    maxval = xf[signal.find_peaks(xf, height=80,distance=10,prominence=5,width=1)[0]]
    index_max = signal.find_peaks(xf, height=80,distance=10,prominence=5,width=1)[0]
    # 排除不需要的元素,距离Dense小于100的值
#     idx = 0
#     delete_num = 0
#     for i in maxval:
#         if i < 180:
#             maxval = np.delete(maxval,idx-delete_num)
#             index_max = np.delete(index_max,idx-delete_num)
#             delete_num += 1
#         idx+=1
#         print(i,idx)
    
    plt.figure(figsize=(16,4))
    plt.plot(np.arange(len(xf)),xf,label='max')
    plt.legend()
    print(maxval)
    print(index_max)

    plt.plot(index_max,maxval,'o',label='z')
    plt.tick_params(labelsize=22)
    plt.legend()
    plt.savefig(fname="name.svg",format="svg",transparent = True)
    plt.show()
    
    return x,xf,maxval,index_max

## 获取横断面的质心，对应椎体中心点的XY坐标
### GetCentroidXY
def GetCentroidXY(PredPNGPaths,CentroidZs):
    CentroidXs = []
    CentroidYs = []
    for img_index in CentroidZs:
        img_max = cv2.imread(PredPNGPaths[img_index])
        imgray = cv2.cvtColor(img_max, cv2.COLOR_BGR2GRAY)
        (T,imgthres) = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY)
#         contours, hierachy = cv2.findContours(imgthres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	#findContours函数用于找出边界点
#         area = []
#         for i in contours:
#             area.append(cv2.contourArea(i))
#         nparea = np.array(area)
#         indexmaxcontour  = nparea.argmax()
#         m = cv2.moments(contours[indexmaxcontour])
        m = cv2.moments(imgthres)
        cx = int(m["m10"] / m["m00"])*4  #128->512
        cy = int(m["m01"] / m["m00"])*4  #128->512
        CentroidXs.append(cx)
        CentroidYs.append(cy)
    return np.array(CentroidXs),np.array(CentroidYs)  

### GetCentroidXY_lsq
from numpy import *
#! python
#  == METHOD 2 ==
from scipy      import optimize

def GetCentroidXY_lsq(PredPNGPaths,CentroidZs):
    CentroidXs = []
    CentroidYs = []
    for img_index in CentroidZs:
        img_max = cv2.imread(PredPNGPaths[img_index])
        imgray = cv2.cvtColor(img_max, cv2.COLOR_BGR2GRAY)
        
        imgray[imgray<50]=0
        
        minval = np.min(imgray[np.nonzero(imgray)])
        maxval = np.max(imgray[np.nonzero(imgray)])
        meanval = (int(minval)*0.2+int(maxval)*0.8)
        
        imgray[imgray>meanval+5]=0
        imgray[imgray<meanval-5]=0
        choice = np.where(imgray!=0)
        
        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        y=choice[0]
        x=choice[1]

        x_m = mean(x)
        y_m = mean(y)
#         print(x_m,y_m,x,y)
        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2       = calc_R(*center_2)
        R_2        = Ri_2.mean()
        residu_2   = sum((Ri_2 - R_2)**2)
        
        CentroidXs.append(xc_2*4)
        CentroidYs.append(yc_2*4)
        cv2.circle(imgray,(int(xc_2),int(yc_2)),2,(255,0,255),1)
        plt.imshow(imgray)
    return np.array(CentroidXs),np.array(CentroidYs)  

### GetCentroidXY_lsq_hist
from numpy import *
#! python
#  == METHOD 2 ==
from scipy      import optimize

def GetCentroidXY_lsq_hist(PredPNGPaths,CentroidZs):
    CentroidXs = []
    CentroidYs = []
    for img_index in CentroidZs:
        img_max = cv2.imread(PredPNGPaths[img_index])
        imgray = cv2.cvtColor(img_max, cv2.COLOR_BGR2GRAY)
        
        imgray[imgray<10]=0
        imgnonzero = imgray[np.nonzero(imgray)]
        hist, bins = np.histogram(imgnonzero.ravel(), 20, density = True)
        mu = np.mean(bins) #计算均值
        sigma = np.std(bins)
        # add a 'best fit' line
        y = norm.pdf(bins, mu, sigma)
        meanval = bins[y.argmax()]
        print(meanval)
             
        imgray[imgray>meanval+5]=0
        imgray[imgray<meanval-5]=0
    
        choice = np.where(imgray!=0)
        
        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        y=choice[0]
        x=choice[1]

        x_m = mean(x)
        y_m = mean(y)
#         print(x_m,y_m,x,y)
        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2       = calc_R(*center_2)
        R_2        = Ri_2.mean()
        residu_2   = sum((Ri_2 - R_2)**2)
        
        CentroidXs.append(xc_2*4)
        CentroidYs.append(yc_2*4)
        cv2.circle(imgray,(int(xc_2),int(yc_2)),2,(255,0,255),1)
        plt.imshow(imgray)
    return np.array(CentroidXs),np.array(CentroidYs)

### GetCentroidXY_lsq_mean

from numpy import *
#! python
#  == METHOD 2 ==
from scipy      import optimize

def GetCentroidXY_lsq_mean(PredPNGPaths,CentroidZs,imagePath,Thres):
    CentroidXs = []
    CentroidYs = []
    count4show = 0
    for img_index in CentroidZs:
        x_c_list = []
        y_c_list = []
        for i in range(2,6,1):
            img_max = cv2.imread(PredPNGPaths[img_index])

            img_size = len(img_max)
            imgray = cv2.cvtColor(img_max, cv2.COLOR_BGR2GRAY)

            imgray[imgray<Thres]=0 #泛化能力测试时选择80，Dataset2-Jack选择80
            imgthres = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
            imgfinal = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
            
            if count4show == 0:
                if i ==2:
                    plt.xticks(())
                    plt.yticks(())
                    plt.imshow(img_max)
                    plt.show()

                    plt.tick_params(labelsize=15)
                    plt.xticks(())
                    plt.yticks(())
                    plt.imshow(imgthres)
                    plt.show()

            minval = np.min(imgray[np.nonzero(imgray)])
            maxval = np.max(imgray[np.nonzero(imgray)])
            
            print(minval,maxval)
            
            meanval = (int(minval)*0.1*i + int(maxval)*0.1*(10-i))

            imgray[imgray>meanval+5]=0
            imgray[imgray<meanval-5]=0
            choice = np.where(imgray!=0)

            def calc_R(xc, yc):
                """ calculate the distance of each 2D points from the center (xc, yc) """
                return sqrt((x-xc)**2 + (y-yc)**2)

            def f_2(c):
                """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
                Ri = calc_R(*c)
                return Ri - Ri.mean()

            y=choice[0]
            x=choice[1]

            x_m = mean(x)
            y_m = mean(y)
            # print(x_m,y_m,x,y)
            center_estimate = x_m, y_m
            center_2, ier = optimize.leastsq(f_2, center_estimate)

            xc_2, yc_2 = center_2
            Ri_2       = calc_R(*center_2)
            R_2        = Ri_2.mean()
            residu_2   = sum((Ri_2 - R_2)**2)
            x_c_list.append(xc_2)
            y_c_list.append(yc_2)
            
            if count4show == 0:
                plt.tick_params(labelsize=15)
                plt.xticks(())
                plt.yticks(())
                plt.subplot(131)
                plt.imshow(imgray)

                imgray[imgray>0]=255
                imgray1 = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
                img_circle = MaskOverlop(imgthres,imgray1,(255,255,0))
                plt.tick_params(labelsize=15)
                plt.xticks(())
                plt.yticks(())
                plt.subplot(132)
                plt.imshow(img_circle)

                cv2.circle(img_circle,(int(xc_2),int(yc_2)),2,(255,0,255),-1)
                plt.tick_params(labelsize=15)
                plt.xticks(())
                plt.yticks(())
                plt.subplot(133)
                plt.imshow(img_circle)
                plt.savefig(fname="name"+str(i)+".svg",format="svg",transparent = True)
                plt.show()
            
        xc_2 = mean(x_c_list)
        yc_2 = mean(y_c_list)
        
        itkimage = sitk.ReadImage(imagePath)
        Sizeimage = itkimage.GetSize()
        ScaleX = Sizeimage[0]/img_size
        ScaleY = Sizeimage[1]/img_size
        
        CentroidXs.append(xc_2*ScaleX)  #根据1*1*1之后的SizeX/img_size，得到ScaleX
        CentroidYs.append(yc_2*ScaleY)  #根据1*1*1之后的SizeY/img_size，得到ScaleY
        
        if count4show == 0:        
            cv2.circle(imgfinal,(int(xc_2),int(yc_2)),2,(255,128,0),-1)
            plt.tick_params(labelsize=15)
            plt.xticks(())
            plt.yticks(())
            plt.imshow(imgfinal)
            plt.show()
        count4show+=1
    return np.array(CentroidXs),np.array(CentroidYs)  

def MaskOverlop(scr,mask,color):
    rows,cols,channels = mask.shape
    for i in range(0,rows,1):
        for j in range(0,cols,1):
            if mask[i,j,0]!=0:
                scr[i,j,0]=color[0]
                scr[i,j,1]=color[1]
                scr[i,j,2]=color[2]
    return scr
#     plt.imshow(bgr3,cmap='gray')
#     plt.show()

## 获取预测的所有椎体中心点坐标
# predsPath: 预测的png图片所在的路径
# IJKPath: 输出的IJK坐标保存路径
# imagePath: 重采样为1x1x1的文件路径
# Thres: 去除阈值低于某个值的部分
def GetCentroids(predsPath,IJKPath,imagePath,Thres):
    pngPaths = sorted(glob(predsPath))
    xZ,xZf,CentroidZmaxval,CentroidZs = GetCentroidZEnhance(pngPaths)
    CentroidXs,CentroidYs = GetCentroidXY_lsq_mean(pngPaths,CentroidZs,imagePath,Thres)
    Centroids = np.column_stack((CentroidXs,CentroidYs,CentroidZs))
    
    path_IJK = os.path.dirname(IJKPath)
    if not os.path.exists(path_IJK):
        os.makedirs(path_IJK)
    np.savetxt(IJKPath, Centroids, delimiter=',')
    return len(Centroids)

#GetLE: 获取预测中心和标注中心的Location error
from scipy.spatial.distance import pdist
'''
Centroids: csv files list of GT centroids
Pred_Centroids: csv files list of predicted centroids
'''

def GetLE(Centroids, Pred_Centroids):
    
    i = 0
    LE_lists = []
    mean_lists = []
    std_lists = []
    for Centroid in Centroids:
        caseCentroid = np.loadtxt(Centroid,delimiter = ',')
        predCentroid = np.loadtxt(Pred_Centroids[i],delimiter = ',')

        filename = re.sub(r'.csv','',os.path.split(Pred_Centroids[i])[1])
        file_LE = os.path.split(Pred_Centroids[i])[0] + '/LE/' + filename + '_LE.csv'
        
        path_LE = os.path.dirname(file_LE)
        if not os.path.exists(path_LE):
            os.makedirs(path_LE)
        
        j = 0
        dist_raw_list = []
        for eachcentroid in caseCentroid:

            X = np.vstack([eachcentroid,predCentroid[j]])
            dist_raw = pdist(X)
            dist_raw_list.append(dist_raw[0])
            j+=1

        np.savetxt(file_LE, np.array(dist_raw_list), delimiter=',')
        LE_lists.append(dist_raw_list)
        mean_LE = np.mean(dist_raw_list)
        std_LE  = np.std(dist_raw_list)
        mean_lists.append(mean_LE)
        std_lists.append(std_LE)
        
        i+=1
    
    mean_LEs = np.mean(mean_lists)
    std_LEs = np.std(std_lists)
    return LE_lists,mean_lists,std_lists,mean_LEs,std_LEs

