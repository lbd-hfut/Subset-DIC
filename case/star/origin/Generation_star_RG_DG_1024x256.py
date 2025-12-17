# -*- coding: utf-8 -*-  
"""  
Created on Sun Jan 28 16:05:23 2024  
@author: leebda  
"""  
#%%  
import torch  
import torch.nn.functional as F  
import matplotlib.pyplot as plt  
import numpy as np  
from PIL import Image  
import cv2  
  
#%%  
# 加载图像并转换为灰度图  
ref_image = Image.open('./case/star/origin/star_original.bmp')  
ref_gray = ref_image.convert('L')  
RG = np.array(ref_gray)  
  
# 星状位移场  
H, L = RG.shape;  
x = np.arange(L)  
y = np.arange(H)  
X, Y = np.meshgrid(x, y)  
pmax = 120; pmin = 10  # 控制最左端的条纹数目 和 最右端的条纹数目  
pwave = pmin + X * (pmax - pmin) / L  
v1 = 0.5 * np.cos((Y - H / 2) * 2 * np.pi / (pwave))  
# 归一化矩阵到[-1, 1]范围  
min_val = np.min(v1)  
max_val = np.max(v1)  
v = 2 * (v1 - min_val) / (max_val - min_val) - 1  
  
y_size, x_size = RG.shape  
x_list = np.linspace(-1, 1, x_size)  
y_list = np.linspace(-1, 1, y_size)  
X, Y = np.meshgrid(x_list, y_list)  
  
# 创建插值后的新采样点  
displacement_field_u = np.zeros_like(v)  
displacement_field_v = v  
X_new = X - 2*displacement_field_u / x_size  
Y_new = Y - 2*displacement_field_v / y_size  
# 转换为PyTorch张量  
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)  
Y_new_tensor = torch.tensor(Y_new, dtype=torch.float32)  
RG_tensor = torch.tensor(RG, dtype=torch.float32)  
# 执行双线性插值  
interpolated_RG = F.grid_sample(RG_tensor.unsqueeze(0).unsqueeze(0),   
                                torch.stack((X_new_tensor, Y_new_tensor), dim=2).unsqueeze(0),   
                                mode='bilinear', align_corners=True)  
# 得到插值后的变形散斑图像  
DG = interpolated_RG[0, 0].view(256, -1).numpy()  
  
# 添加高斯噪声  
noise_mean = 0  
noise_std = 0
RG_noisy = RG + np.random.normal(noise_mean, noise_std, RG.shape)  
DG_noisy = DG + np.random.normal(noise_mean, noise_std, DG.shape)  
  
# 确保图像数据在有效范围内（0-255）  
RG_noisy = np.clip(RG_noisy, 0, 255).astype(np.uint8)  
DG_noisy = np.clip(DG_noisy, 0, 255).astype(np.uint8) 
ROI = np.ones_like(RG_noisy) * 255
  
#%% 画出散斑图像与星状位移场  
plt.figure(figsize=(8, 5))  
plt.subplot(2, 1, 1)  
plt.imshow(RG_noisy, cmap='gray')  
plt.axis('off')  # 关闭坐标轴，可选  
plt.title("Noisy Speckle Image")  
plt.subplot(2, 1, 2)  
plt.imshow(displacement_field_v, cmap='jet')  
plt.axis('off')  # 关闭坐标轴，可选  
plt.title(f"Pmax={pmax}; Pmin={pmin}")  
  
#%%  
# 保存图像  
cv2.imwrite(f'./case/star/001.bmp', RG_noisy)  
cv2.imwrite(f'./case/star/002.bmp', DG_noisy)  
cv2.imwrite(f'./case/star/003.bmp', ROI)  
#%%  
import scipy.io as sio  
sio.savemat('v.mat', {'v': v})