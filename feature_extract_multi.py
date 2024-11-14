import torch
import numpy as np
import xlrd
import os
import nrrd
import six
import csv
import SimpleITK as sitk
from radiomics import featureextractor
 
 
# def count_file_number(filepath, filetype):
#     count = 0  # 初始化计数器
#     # 遍历指定路径下的所有文件
#     for root, dirname, filenames in os.walk(filepath):
#         for filename in filenames:
#             # 检查文件扩展名是否匹配指定类型
#             if os.path.splitext(filename)[1] == filetype:
#                 count += 1  # 增加计数器
#     return count, filenames  # 返回文件数量和文件名列表

def count_file_number(directory, type):
    dicom_files = []
    counter = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(type):  # 过滤DICOM文件扩展名
                dicom_files.append(os.path.join(root, file))
                counter +=1
                # print("DICOM files found:", dicom_files)
    return counter, dicom_files


# def dcmseriesread(dicom_names):
#     readerC = sitk.ImageSeriesReader()  # 创建图像序列读取器对象
#     # dicom_names = readerC.GetGDCMSeriesFileNames(dicompath)  # 获取指定路径下的DICOM文件名
#     readerC.SetFileNames(dicom_names)  # 设置读取的文件名
#     readerC.MetaDataDictionaryArrayUpdateOn()  # 启用元数据字典数组更新
#     readerC.LoadPrivateTagsOn()  # 启用私有标签加载
#     dicomImage = readerC.Execute()  # 执行读取操作，返回DICOM图像对象
#     return dicomImage  # 返回读取的图像

def dcmseriesread(dicom_names):
    readerC = sitk.ImageSeriesReader()  
    readerC.SetFileNames(dicom_names)
    readerC.MetaDataDictionaryArrayUpdateOn()  
    readerC.LoadPrivateTagsOn()
    try:
        dicomImage = readerC.Execute()  # 尝试执行读取
        print("Image successfully loaded.")
        return dicomImage  # 返回读取的图像
    except RuntimeError as e:
        print("Error loading image:", e)
        # index = [0, 0, 0]  # 从图像的起始点开始
        # sample_image = sitk.ReadImage(dicom_names[0])
        # size = list(sample_image.GetSize())  # 获取图像的尺寸，自动获取实际大小
        # print(size)
        if subID[i] not in errorId:
            errorId.append(str(subID[i]))
        # 设置读取区域（仅在某些 ITK 版本中有效）
        # requested_region = sitk.ImageRegion(3, index, size)
        # readerC.SetRequestedRegion(requested_region)
        # dicomImage = readerC.Execute()  # 尝试执行读取
        return None

def radiomics_feature_extractor(image, mask):
    settings = {}  # 创建一个空字典以存储特征提取设置
    settings['binWidth'] = 25  # 设置直方图的宽度
    settings['resampledPixelSpacing'] = None  # 设置重采样像素间距，None表示不进行重采样
    settings['interpolator'] = sitk.sitkBSpline  # 设置插值方法为B样条插值
    settings['correctMask'] = True  # 启用掩膜校正
    settings['geometryTolerance'] = 1  # 设置几何容差

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  # 创建特征提取器对象，并传入设置
    # extractor.enableAllImageTypes()  # 启用所有图像类型
    # extractor.enableFeatureClassByName('shape', 'texture')  # 可选择启用形状和纹理特征
    extractor.enableImageTypeByName('Original')  # 启用原始图像特征
    extractor.enableImageTypeByName('Wavelet')   # 启用小波分解图像特征
    # extractor.enableFeatureClassByName('firstorder')  # 可选择启用第一类特征
    try:
        featureVector = extractor.execute(image, mask, label = 2)  # 执行特征提取，返回特征向量
    except:
        featureVector = extractor.execute(image, mask, label = 1)
    radiomicsList = []  # 创建空列表以存储特征值
    header = []  # 创建空列表以存储特征名称
    for key, val in six.iteritems(featureVector):  # 遍历特征向量中的键值对
        if not key.startswith('diagnostics'):  # 过滤掉以'diagnostics'开头的特征
            header.append(key)  # 将特征名称添加到header列表
            radiomicsList.append(str(val))  # 将特征值转换为字符串并添加到radiomicsList列表
    return header, radiomicsList  # 返回特征名称和特征值列表

# path = '/root/autodl-tmp/mark/negative'  # 定义项目路径
# # SKindex = xlrd.open_workbook(os.path.join(path, 'index.xlsx')).sheets()[0]  # 读取Excel索引文件
# # subIDtemp = np.array(SKindex.col_values(0))[12:]  # 获取第一列的值，从第13行开始
# # subID = [x[:-2].zfill(6) for x in subIDtemp]  # 对ID进行处理，去掉最后两位并填充为6位
# # print(subID)  # 打印处理后的ID（已注释掉）
# # subID = ['1231351', '1683488', '2081495', '2828100', '3058822', '3725504', '4299051', '4337158', '5365855', '5461857', '5462573', '5470470', '5496720', '5510883', '5535780', '5659272']
# subID = ['5510883', '5535780', '5659272']
# counter = 0  # 初始化计数器

# for i in range(len(subID)):  # 遍历所有患者ID
#     patientPath = os.path.join(path, subID[i])  # 定义患者图像文件路径
#     maskPath = os.path.join(path, subID[i])  # 定义掩膜文件路径
#     dicomSlices, dicomNames = count_file_number(patientPath, '.dcm')  # 统计DICOM文件数量
#     originalImage = dcmseriesread(dicomNames)  # 读取原始DICOM图像
#     maskNumber, maskNames = count_file_number(maskPath, '.nrrd')  # 统计NRRD掩膜文件数量

#     for maskName in maskNames:  # 遍历所有掩膜文件名
#         print([str(subID[i]), maskName, 'processing....'])  # 打印当前处理的信息
#         maskMatrix, options = nrrd.read(maskName)  # 读取NRRD掩膜文件
#         maskSlices = maskMatrix.shape[-1]  # 获取掩膜的切片数量

#         if maskSlices == dicomSlices:  # 检查掩膜切片数量是否与DICOM切片数量一致
#             maskImage = sitk.ReadImage(maskName)  # 读取掩膜图像
#             maskImage.SetOrigin(originalImage.GetOrigin())
#             header, radiomicsList = radiomics_feature_extractor(originalImage, maskImage)  # 提取放射组学特征
#             with open('data/ct_features.csv', 'a', newline='') as outcsv:  # 打开CSV文件以附加方式写入
#                 writer = csv.writer(outcsv)  # 创建CSV写入对象
#                 if counter == 0:  # 如果是第一次写入，写入表头
#                     writer.writerow(['patientID', 'maskName'] + header + ['label'])  # 写入表头
#                 writer.writerow([str(subID[i]), os.path.basename(maskName.rstrip("/\\"))] + radiomicsList + [0])  # 写入患者ID、掩膜名及特征值
#                 counter += 1  # 计数器加1

#         else:  # 如果掩膜切片数量与DICOM切片数量不一致
#             print("different")
#             with open("data/ct_features_error.csv", 'a', newline='') as outcsv:  # 打开CSV文件以附加方式写入
#                 writer = csv.writer(outcsv)  # 创建CSV写入对象
#                 writer.writerow([str(subID[i]), os.path.basename(maskName.rstrip("/\\"))])  # 记录患者ID和掩膜名（不提取特征）

path = 'autodl-tmp/mark/positive'  # 定义项目路径
# SKindex = xlrd.open_workbook(os.path.join(path, 'index.xlsx')).sheets()[0]  # 读取Excel索引文件
# subIDtemp = np.array(SKindex.col_values(0))[12:]  # 获取第一列的值，从第13行开始
# subID = [x[:-2].zfill(6) for x in subIDtemp]  # 对ID进行处理，去掉最后两位并填充为6位
# print(subID)  # 打印处理后的ID（已注释掉）
subID = ['1417907', '2081495', '2672227', '2749851', '2828100', '3428473', '3725504', '4175625', '4231455', '4299051', '4403085', '4472159', '4919128', '4954003', '5267110', '5422719', '5462573', '5475795', '5481029', '5487011', '5487738', '5497794', '5504833', '5535788', '5537399', '5632796', '5633900', '5633993', '5634984', '5645989', '5659388', '5663089', '5663617', '5667740']
counter = 0  # 初始化计数器
# global errorId 
# errorId = []
for i in range(len(subID)):  # 遍历所有患者ID
    patientPath = os.path.join(path, subID[i])  # 定义患者图像文件路径
    maskPath = os.path.join(path, subID[i])  # 定义掩膜文件路径
    dicomSlices, dicomNames = count_file_number(patientPath, '.dcm')  # 统计DICOM文件数量
    originalImage = dcmseriesread(dicomNames)  # 读取原始DICOM图像
    maskNumber, maskNames = count_file_number(maskPath, '.nrrd')  # 统计NRRD掩膜文件数量

    for maskName in maskNames:  # 遍历所有掩膜文件名
        print([str(subID[i]), maskName, 'processing....'])  # 打印当前处理的信息
        maskMatrix, options = nrrd.read(maskName)  # 读取NRRD掩膜文件
        maskSlices = maskMatrix.shape[-1]  # 获取掩膜的切片数量

        if maskSlices == dicomSlices:  # 检查掩膜切片数量是否与DICOM切片数量一致
            maskImage = sitk.ReadImage(maskName)  # 读取掩膜图像
            maskImage.SetOrigin(originalImage.GetOrigin())
            header, radiomicsList = radiomics_feature_extractor(originalImage, maskImage)  # 提取放射组学特征
            with open('autodl-tmp/data/ct_features.csv', 'a', newline='') as outcsv:  # 打开CSV文件以附加方式写入
                writer = csv.writer(outcsv)  # 创建CSV写入对象
                # if counter == 0:  # 如果是第一次写入，写入表头
                #     writer.writerow(['patientID', 'maskName'] + header + ['label'])  # 写入表头
                writer.writerow([str(subID[i]), os.path.basename(maskName.rstrip("/\\"))] + radiomicsList + [0])  # 写入患者ID、掩膜名及特征值
                counter += 1  # 计数器加1
        else:  # 如果掩膜切片数量与DICOM切片数量不一致
            print("different")
            # if subID[i] not in errorId:
            #     errorId.append(str(subID[i]))
            with open("autodl-tmp/data/ct_features_error.csv", 'a', newline='') as outcsv:  # 打开CSV文件以附加方式写入
                writer = csv.writer(outcsv)  # 创建CSV写入对象
                writer.writerow([str(subID[i]), os.path.basename(maskName.rstrip("/\\"))])  # 记录患者ID和掩膜名（不提取特征）

# warning: 4403085