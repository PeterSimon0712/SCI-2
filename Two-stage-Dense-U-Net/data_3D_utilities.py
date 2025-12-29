"""data_3D_utilities.py: data preparation and result process for 3D-Dense-U-Net."""
"""Get ROI and RecoverROI"""

__author__      = "Peter"

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import imageio
import elasticdeform.tf as etf
import tensorflow as tf

#BoxSize
BoxSize = np.array([80,128,112],dtype='int16') #z,y,x

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


#替换cv2.normalize,保证不同图像像素值映射的一致性。
#Dataset-2数据集：0对应0，255对应4095；
def normalization(hu_value, hu_min, hu_max):
    normal_value = np.uint8((hu_value - hu_min) / (hu_max - hu_min)*255)
    return normal_value

# 重采样
def ImageResample(initfile, resmplfile, new_spacing, is_label = True):
    '''
    initfile: path of the file for resample
    resmplfile: output path of the resampled file
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    sitk_image = sitk.ReadImage(initfile)
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    # new_spacing = np.array([spacing[0], spacing[1], 10]) #切片间距离变为10mm
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = np.array([float(s) for s in new_spacing_refine]).tolist()
    new_size = np.array([int(s) for s in new_size]).tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)
    
    #线性差值或者邻近差值
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    
    path_resmpl = os.path.dirname(resmplfile)
    if not os.path.exists(path_resmpl):
        os.makedirs(path_resmpl)
    sitk.WriteImage(newimage, resmplfile)
    
    return newimage

# 提取标注的有效区域，验证横断面提取被标注的腰椎有效区域
def ImageSpilt(casefile, labelfile, caseout, labelout, exNum):
    '''
    casefile: path of the file for original
    labelfile: path of the file for original
    caseout: output path of the extracted case file
    labelout: output path of the extracted label file
    iskeepend: determine whether keep the end of the slice
    '''
    
    case_image = sitk.ReadImage(casefile)
    label_image = sitk.ReadImage(labelfile)

    np_case = sitk.GetArrayFromImage(case_image)
    np_label = sitk.GetArrayFromImage(label_image)

    spacing = case_image.GetSpacing()

    #channel根据要提取的面设置，0-横断面，1-冠状面，2-矢状面
    channel = np_label.shape[0]
    list_max = []
    for s in range(channel):
        slicer = np_label[s,:,:] #横断面获取
        list_max.append(slicer.max())

    # 获得label有效标注的范围
    np_list_max = np.array(list_max)
    idx_start = (np_list_max!=0).argmax(axis=0)
    idx_start = idx_start - exNum
    if idx_start < 0:
        idx_start = 0
    
    np_list_max = np.flip(np_list_max)
    idx_end = np_list_max.size - (np_list_max!=0).argmax(axis=0)
    idx_end = idx_end + exNum
    if idx_end > np_list_max.size:
        idx_end = np_list_max.size
        
    caseout_image = sitk.GetImageFromArray(np_case[idx_start:idx_end,:,:])
    labelout_image = sitk.GetImageFromArray(np_label[idx_start:idx_end,:,:])
    
    path_caseout = os.path.dirname(caseout)
    if not os.path.exists(path_caseout):
        os.makedirs(path_caseout)
    path_labelout = os.path.dirname(labelout)
    if not os.path.exists(path_labelout):
        os.makedirs(path_labelout)    
    
    sitk.WriteImage(caseout_image, caseout)
    sitk.WriteImage(labelout_image, labelout)
    
    return caseout_image, labelout_image

# 根据椎体中心点坐标及BoxSize获取ROI范围
def GetROIbyCentroid(npImage,centroid,BoxSize):
    dim = np.zeros((3,2),dtype='int16')  # bouding box 和 z,y,x轴的交点
    FlagFill = np.zeros((3,2),dtype='int8') # z,y,x
    
    for i in range(0,3,1):
        dim[i,0] = int(centroid[i] - BoxSize[i]/2)
        if 0 <= dim[i,0] < npImage.shape[i]:
            FlagFill[i,0] = 0
        elif dim[i,0] < 0:
            FlagFill[i,0] = -1
            dim[i,0] = 0
        elif dim[i,0] >= npImage.shape[i]:
            FlagFill[i,0] = 1
            dim[i,0] = npImage.shape[i]-1

        dim[i,1] = int(centroid[i] + BoxSize[i]/2)
        if 0 <= dim[i,1] < npImage.shape[i]:
            FlagFill[i,1] = 0
        elif dim[i,1] < 0:
            FlagFill[i,1] = -1
            dim[i,1] = 0
        elif dim[i,1] >= npImage.shape[i]:
            FlagFill[i,1] = 1
            dim[i,1] = npImage.shape[i]-1
    # print(np.unique(npImage),npImage.shape)
    return dim,FlagFill

# 获取imageROI，并根据FlagFill判断是否填补0
# 只判断截取左侧坐标小于0，及右侧坐标大于图像最大索引值的情况
def GetimageROI(npImage,dim,FlagFill,labelPixel,islabel):
    imageROI = npImage[dim[0,0]:dim[0,1], dim[1,0]:dim[1,1], dim[2,0]:dim[2,1]]
    for i in range(0,3,1):
        if FlagFill[i,0]!=0 or FlagFill[i,1]!=0:
            if i==0:
                Fill = np.zeros((BoxSize[0]-imageROI.shape[0],imageROI.shape[1],imageROI.shape[2]),dtype='int16')
            elif i==1:
                Fill = np.zeros((imageROI.shape[0],BoxSize[1]-imageROI.shape[1],imageROI.shape[2]),dtype='int16')
            elif i==2:
                Fill = np.zeros((imageROI.shape[0],imageROI.shape[1],BoxSize[2]-imageROI.shape[2]),dtype='int16')
            if FlagFill[i,0] < 0:
                imageROI = np.concatenate((Fill,imageROI),axis=i)
            if FlagFill[i,1] > 0:
                imageROI = np.concatenate((imageROI,Fill),axis=i)
    
    #     # 如果是标签数据，判断最多的标签值，作为这个ROI内椎体的label，其余像素全部为0
    #     sumelements=[]
    #     if islabel:
    #         elements = np.unique(imageROI)
    #         for element in elements:
    #             if element!=0:
    #                 sumelement = np.sum(imageROI == element)
    #                 sumelements.append(sumelement)
    #         npsumelements = np.array(sumelements)
    #         maxsum = npsumelements.max()
    #         maxsumidx = np.where(npsumelements == maxsum)
    #         print(elements,maxsum,maxsumidx)
    #         imageROI[imageROI!=elements[maxsumidx]]=0
    #     # 如果是标签数据，判断最靠近椎体中心的坐标像素值，作为这个ROI内椎体的label，其余像素全部为0
    #     # 未编写

    # 如果是标签数据，把该椎体中心对应椎体的像素值保留，其余像素全部为0
    if islabel:
        labelsROI = np.unique(imageROI)
        # 打印观察ROI区域内有多少不同的椎体
        # print(labelsROI,len(labelsROI),labelPixel,np.unique(npImage),npImage.shape)
        imageROIValid = np.where(imageROI!=labelPixel,0,imageROI)
    else:
        imageROIValid = imageROI
    return imageROIValid

# 二值化图像,直接找出像素最多的那个值
def FindPixelMaxNum(ImgSlice):
    temp = 0
    LabelPixelval = 0
    LabelPixel = 0
    labelsSlice = np.unique(ImgSlice)
    for i in labelsSlice:
        if i!=0:
            temp = len(ImgSlice[ImgSlice==i])
            if temp > LabelPixelval:
                LabelPixelval = temp
                LabelPixel = i
    return LabelPixel

# casePath:  未分割图片路径
# outPath:   每段椎体保存路径
# centroids: 椎体中心点坐标（x,y,z）
# BoxSize:   ROI大小(z,y,x)
# islabel:  case: False, label: True

def FindROI3D(casePath,outPath,centroids,normmin,normmax,islabel,Savenp = True,Savemhd = True,Savenii = True):
    
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(casePath)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x(Formerly: x y z)
    npImage = sitk.GetArrayFromImage(itkimage)
    min_norm = npImage.min()
    # print(numpyImage.shape)
    
    labelPixel = 0
    if islabel:
        labelsAll = np.unique(npImage)
        print(labelsAll)
    
    i = 0
    for centroid in centroids:
        # [x,y,z]->[z,y,x]，Convert centroids coordinates to [1,1,1]
        centroid = np.array(np.flipud(centroid),dtype='int16')

        dim,FlagFill = GetROIbyCentroid(npImage,centroid,BoxSize)
        # print(dim)
        if islabel:
            # 统计z轴坐标对应的图片像素值分布，并获得最接近中心的像素值,即除0外值最多的像素值
            ImgSlice = npImage[centroid[0],:,:]
            labelPixel = FindPixelMaxNum(ImgSlice)
            print(labelPixel)
            # labelPixel = labelsAll[i+1]
        else:
            labelPixel = 0
            # print(labelPixel)
        imageROI = GetimageROI(npImage,dim,FlagFill,labelPixel,islabel)

        # print(centroid)
        if islabel:
            imageROInorm = normalization(imageROI,0,labelPixel)
        else:
            imageROInorm = normalization(imageROI,normmin,normmax)
        
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        
        # save as numpy array
        if Savenp:
            imageROInorm.tofile(outPath + str('%03d' % i) + ".bin")
        
        if Savemhd:
            out = sitk.GetImageFromArray(imageROInorm)
            out.SetSpacing(itkimage.GetSpacing())
            #out.SetOrigin(itkimage.GetOrigin())
            sitk.WriteImage(out,outPath + str('%03d' % i) + ".mhd")
            
        if Savenii:
            out = sitk.GetImageFromArray(imageROInorm)
            out.SetSpacing(itkimage.GetSpacing())
            #out.SetOrigin(itkimage.GetOrigin())
            sitk.WriteImage(out,outPath + str('%03d' % i) + ".nii")
        
        # i +=1 必须在最后一行，保证前面所有处理的i值对应的都是同一个    
        i +=1

# 椎体重组

# 根据FlagFill把npImageROI截取到dim同样的大小，与生成imageROI相反.
def GetnpImageROIbydim(npImageROI,BoxSize,FlagFill,dim):
    for i in range(2,-1,-1):
        if FlagFill[i,0]!=0 or FlagFill[i,1]!=0:
            if i==2:
                if FlagFill[i,0] < 0:
                    npImageROI = npImageROI[:,:,(BoxSize[i]-dim[i,1]):BoxSize[i]]
                if FlagFill[i,1] > 0:
                    npImageROI = npImageROI[:,:,0:(dim[i,1]-dim[i,0])]
            elif i==1:
                if FlagFill[i,0] < 0:
                    npImageROI = npImageROI[:,(BoxSize[i]-dim[i,1]):BoxSize[i],:]
                if FlagFill[i,1] > 0:
                    npImageROI = npImageROI[:,0:(dim[i,1]-dim[i,0]),:]
            elif i==0:
                if FlagFill[i,0] < 0:
                    npImageROI = npImageROI[(BoxSize[i]-dim[i,1]):BoxSize[i],:,:]
                if FlagFill[i,1] > 0:
                    npImageROI = npImageROI[0:(dim[i,1]-dim[i,0]),:,:]
    return npImageROI

# Dataset-2数据集：0对应0，255对应4095；
# 将训练结束，0-255的数据恢复到0-4095；
def normalRecovery(hu_value, NthVertebra):
    normal_value = np.uint16((hu_value)/255*NthVertebra)
    return normal_value

def npImageNewFill(npImageNew,npImageROI,dim):
    for i in range(dim[0,0],dim[0,1],1):
        for j in range(dim[1,0],dim[1,1],1):
            for k in range(dim[2,0],dim[2,1],1):
                if npImageROI[i-dim[0,0],j-dim[1,0],k-dim[2,0]] != 0:
                    npImageNew[i,j,k] = npImageROI[i-dim[0,0],j-dim[1,0],k-dim[2,0]]
    return npImageNew

# 将各个椎体ROI组合成完整的脊柱
# casePath:  未分割图片的路径
# ROIPath:   已分割每个椎体的路径
# outPath:   保存组合完成的路径
# centroids: 椎体中心点坐标（x,y,z）
# BoxSize:   ROI大小(z,y,x)
# Savemhd:   是否保存为mhd文件

def RecoverROI(casePath,ROIPath,outPath,centroids,BoxSize,Savemhd):
    
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(casePath)
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    # 重采样到1*1*1
    newspacing = np.array([1,1,1])
    itkimageResample = ImageResample(itkimage,newspacing,False)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x(Formerly: x y z)
    npImage = sitk.GetArrayFromImage(itkimageResample)
        
    # 创建一个新的空的图像数据
    npImageNew = np.zeros(npImage.shape)
    
    i = 0
    for centroid in centroids:
        
        centroid = np.array(np.flipud(centroid)*spacing,dtype='int16') # [x,y,z]->[z,y,x]，Convert centroids coordinates to [1,1,1]

        dim,FlagFill = GetROIbyCentroid(npImage,centroid,BoxSize)
        
        # 加载第几个椎体中心点对应的已分割椎体
        npImageROI = np.load(ROIPath + str('%03d' % i) + ".npy")
        
        # 得到对应dim大小的npImageROI
        npImageROI = GetnpImageROIbydim(npImageROI,BoxSize,FlagFill,dim)
        
        # 得到该中心点对应椎体标签
        npImageROI = normalRecovery(npImageROI, 200+i*100)
        
        print(i,dim,FlagFill,npImageROI.shape)
        # 将npImageROI恢复到原来图像
        npImageNew = npImageNewFill(npImageNew,npImageROI,dim)
        
        # i +=1 必须在最后一行，保证前面所有处理的i值对应的都是同一个    
        i +=1
        
    # 重采样到原始分辨率
    itkimagelow = sitk.GetImageFromArray(npImageNew)
    itkimagelow.SetSpacing([1,1,1])
    itkimagehigh = ImageResample(itkimagelow,itkimage.GetSpacing(),True)
    
    if Savemhd:
        #out = sitk.GetImageFromArray(npImageNew)
        #out.SetSpacing(itkimageResample.GetSpacing())
        #out.SetOrigin(itkimage.GetOrigin())
        sitk.WriteImage(itkimagehigh,outPath + str('%03d' % i) + ".mhd")   

## 数据增强

### etfdeform
# 弹性变形
# X,Y 为tensor数据
def etfdeform(X,Y):
    displacement_val = np.random.randn(3, 3, 3, 3) * 5
    displacement = tf.Variable(displacement_val)
    [X_deformed,Y_deformed]=etf.deform_grid([X,Y],displacement,order=[0,0])
    return X_deformed,Y_deformed

### Add noisy
# Parameters
# ----------
# image : ndarray
#     Input image data. Will be converted to float.
# mode : str
#     One of the following strings, selecting the type of noise to add:

#     'gauss'     Gaussian-distributed additive noise.
#     'poisson'   Poisson-distributed noise generated from the data.
#     's&p'       Replaces random pixels with 0 or 1.
#     'speckle'   Multiplicative noise using out = image + n*image,where
#                 n is uniform noise with specified mean & variance.

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
#         sigma = var**0.5
        sigma = 10
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = np.uint8(image + gauss)
        noisy[noisy<0] = 0
        noisy[noisy>255] = 255
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

