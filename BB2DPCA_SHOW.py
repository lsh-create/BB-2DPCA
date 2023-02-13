# 参考：[1]孙凯,林强,陈良洁.基于分块双向2DPCA及ResNet的景象区域适配性分析[J].网络安全技术与应用,2022(05):46-47.
import numpy as np
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt

top_n = 60 # 排前面的特征向量

samples = []
for i in range(1,10):
    im = Image.open(r'D:\dataset\orl_faces\s15\\'+str(i)+'.pgm')
    #用图片的长和宽生成一个空数组
    im_data  = np.empty((im.size[1], im.size[0]))
    for j in range(im.size[1]):
        for k in range(im.size[0]):
            #对于图片的像素进行归一化处理
            R = im.getpixel((k, j))
            im_data[j,k] = R/255.0
    # 将归一化后的像素存储到新生成数组中
    samples.append(im_data)

# 所有样本
images = np.array(samples)

# 求均值
mean_img = images.mean(axis=0)

G = 0
M = images.shape[0]
# 对图像i
for i in range(0,M):
    img = images[i]
    # 对第r行
    for r in range(0,img.shape[0]):
        row = img[r,::]-mean_img[r,::]
        row = np.reshape(row, [row.shape[0],1])
        g = np.dot(row,row.T)
        G = G + g
# 得到协方差矩阵
G = G/M
# 特征值,特征向量
eigenvalue, featurevector = np.linalg.eig(G)
# 将特征值排序,返回排序后的索引
eig_sort_index = np.argsort(-eigenvalue)
eigenvalue[eig_sort_index] # 测试
fv_sort_row = featurevector[eig_sort_index, :] # 特征向量排序，行映射矩阵
fv_sort_row = fv_sort_row[:, 0:top_n] # 选取排前面的特征向量

G = 0
M = images.shape[0]
# 对图像i
for i in range(0,M):
    img = images[i]
    # 对第c列
    for c in range(0,img.shape[1]):
        col = img[::,c]-mean_img[::,c]
        col = np.reshape(col, [col.shape[0],1])
        g = np.dot(col,col.T)
        G = G + g
# 得到协方差矩阵
G = G/M
# 特征值,特征向量
eigenvalue, featurevector = np.linalg.eig(G)
# 将特征值排序,返回排序后的索引
eig_sort_index = np.argsort(-eigenvalue)
eigenvalue[eig_sort_index] # 测试
fv_sort_col = featurevector[:, eig_sort_index] # 特征向量排序，列映射矩阵
fv_sort_col = fv_sort_col[:, 0:top_n] # 选取排前面的特征向量

# 映射矩阵
C = np.dot(fv_sort_col.T, img)
C = np.dot(C, fv_sort_row)

# 矩阵重构
A_r = np.dot(fv_sort_col, C)
A_r = np.dot(A_r, fv_sort_row.T)

# 显示原图
plt.figure()
plt.imshow(img)

# 显示重构图像
plt.figure()
plt.imshow(A_r)





