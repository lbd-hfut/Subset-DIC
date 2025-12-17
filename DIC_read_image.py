import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import label
from math import factorial
import gc
from tqdm import tqdm

class BufferManager:
    QKBQKT_ref = None
    QKBQKT_def = None
    fx = None
    fy = None
    refImg = None
    defImg = None
    mask = None
    mask_pad = None
    w_origin = None
    h_origin = None
    w_resize = None
    h_resize = None
    


class Img_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        train_root = config.input_dir
        image_files = np.array([x.path for x in os.scandir(train_root)
                             if (x.name.endswith(".bmp") or
                             x.name.endswith(".png") or 
                             x.name.endswith(".JPG") or 
                             x.name.endswith(".tiff"))
                             ])
        image_files.sort()
        
        self.rfimage_files = [image_files[0]]
        self.mask_files = [image_files[-1]]
        self.dfimage_files = image_files[1:-1]
        
    def __len__(self):
        return len(self.dfimage_files)
    
    def __getitem__(self, idx):
        # Open images
        df_image = self.open_image(self.dfimage_files[idx])
        BufferManager.defImg = df_image
        defImg_bcoef = self._form_bcoef(df_image, self.config)
        print(f"create QKBQKT_def{idx+1}:")
        BufferManager.QKBQKT_def = self._get_buffer_QK_B_QKT(defImg_bcoef)
        print(f"create QKBQKT_def{idx+1} over!")
        return df_image, defImg_bcoef
    
    def open_image(self,name):
        img = Image.open(name).convert('L')
        w, h = img.size
        if BufferManager.w_origin == None:
            BufferManager.w_origin = w
            BufferManager.h_origin = h
        if self.config.step > 1:
            img = img.resize((w // self.config.step, h // self.config.step), Image.BICUBIC)
        img = np.array(img)
        h, w = img.shape
        if BufferManager.w_resize == None:
            BufferManager.w_resize = w
            BufferManager.h_resize = h
        return img / 255
    
    def _get_refImg(self):
        refImg = self.open_image(self.rfimage_files[0])
        if BufferManager.refImg is None:
            BufferManager.refImg = refImg
        refImg_bcoef = self._form_bcoef(refImg, self.config)
        if BufferManager.QKBQKT_ref is None:
            print("create QKBQKT_ref:")
            BufferManager.QKBQKT_ref = self._get_buffer_QK_B_QKT(refImg_bcoef)
            print("create QKBQKT_ref over!")
        return refImg, refImg_bcoef
    
    def _get_roiRegion(self):
        mask_bin = self.open_image(self.mask_files[0]) > 0
        labeled, num_labels = label(mask_bin)
        if num_labels == 0:
            raise RuntimeError("Mask 中没有前景像素！")
        ROI_list = []
        ROI_pad_list = []
        for comp_id in range(1, num_labels + 1):
            roi_i = (labeled == comp_id)
            # 创建单连通域 ROI
            ROI_list.append(roi_i)
            roi_i_pad = np.pad(roi_i, pad_width=self.config.subset_half_size, mode='constant', constant_values=False)
            ROI_pad_list.append(roi_i_pad)
        if BufferManager.mask is None:
            BufferManager.mask = ROI_list
        if BufferManager.mask_pad is None:
            BufferManager.mask_pad = ROI_pad_list
        return ROI_list
    
    def beta5_nth(self, x, n=0):
        """
        Quintic B-spline β5(x) n-th derivative
        x: np.float64 or np.array
        n: 0~5
        """
        x = np.asarray(x, dtype=np.float32)

        def plus_power(y, p):
            return y**p * (y>0)

        coeffs = [1, -6, 15, -20, 15, -6, 1]
        shifts = [-3, -2, -1, 0, 1, 2, 3]

        result = np.zeros_like(x)
        for c, s in zip(coeffs, shifts):
            factor = factorial(5) // factorial(5-n)
            result += c * factor * plus_power(x + s, 5-n)
        return result / 120
    
    def _get_QK_QKdx_QKdxx(self):
        x_samples = np.array([-2, -1, 0, 1, 2, 3], dtype=np.float32)
        
        # 生成 QK（n = 0~5）
        QK = np.zeros((6, len(x_samples)))  # QK[n, :]
        for n in range(6):
            QK[n, :] = ((-1) ** n) * self.beta5_nth(x_samples, n=n) / factorial(n)
            
        # # 生成 QK_DX (一阶差分核) QK_DX[n-1,:] = -n * QK[n,:]
        # QK_DX = np.zeros((6, len(x_samples)))
        # for n in range(1, 6):
        #     QK_DX[n-1, :] = -n * QK[n, :]
            
        # # 生成 QK_DXX (二阶差分核) QK_DXX[n-2,:] = n*(n-1)*QK[n,:]
        # QK_DXX = np.zeros((6, len(x_samples)))
        # for n in range(2, 6):
        #     QK_DXX[n-2, :] = n * (n-1) * QK[n, :]
        
        self.QK = QK
        # self.QK_DX = QK_DX
        # self.QK_DXX = QK_DXX
    
    def _form_bcoef(self, img, config):
        """Replicate padding by 'border' pixels on each side."""
        if config.subset_half_size >= 3:
            border = config.subset_half_size
        else:
            border = 3
        # padding img
        plot_gs = np.pad(img, pad_width=border, mode='edge')
        h, w = plot_gs.shape
        if h < 5 or w < 5:
            raise ValueError("Array must be >= 5×5 or empty")
        plot_bcoef = np.zeros_like(plot_gs, dtype=np.complex128)
        x_sample = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        kernel_b = self.beta5_nth(x_sample, n=0)
        # Row kernel
        kernel_b_x = np.zeros(w, dtype=np.float32)
        kernel_b_x[:3] = kernel_b[2:]
        kernel_b_x[-2:] = kernel_b[:2]
        kernel_b_x = np.fft.fft(kernel_b_x)
        # across rows
        for i in range(h):
            plot_bcoef[i, :] = np.fft.ifft(np.fft.fft(plot_gs[i, :]) / kernel_b_x)
        # Column kernel
        kernel_b_y = np.zeros(h, dtype=np.float32)
        kernel_b_y[:3] = kernel_b[2:]
        kernel_b_y[-2:] = kernel_b[:2]
        kernel_b_y = np.fft.fft(kernel_b_y)
        # across columns
        for j in range(w):
            plot_bcoef[:, j] = np.fft.ifft(np.fft.fft(plot_bcoef[:, j]) / kernel_b_y)
        return plot_bcoef.real
    
    def _get_image_gradient(self):
        if not hasattr(self, "QK"):
            self._get_QK_QKdx_QKdxx()
        # 加载参考图像及其 B 样条系数
        refImg, ref_bcoef = self._get_refImg()
        roi_list = self._get_roiRegion()
        
        H, W = refImg.shape
        self.fx = np.zeros_like(refImg, dtype=np.float32)
        self.fy = np.zeros_like(refImg, dtype=np.float32)
        
        if self.config.subset_half_size >= 3:
            border = self.config.subset_half_size
        else:
            border = 3
        offset = 2
        roi_pixels = []
        for roi in roi_list:
            ys, xs = np.where(roi)
            roi_pixels.extend(zip(ys, xs))

        for y, x in tqdm(roi_pixels, desc="Computing image gradients", total=len(roi_pixels)):
            x_vec = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)  # (6,)
            w_vec = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
            
            top = y + border - offset
            left = x + border - offset
            bottom = top + 6
            right = left + 6
            block = ref_bcoef[top:bottom, left:right]
            
            self.fx[y, x] = x_vec @ (self.QK @ (block @ (self.QK.T @ w_vec)))
            self.fy[y, x] = w_vec @ (self.QK @ (block @ (self.QK.T @ x_vec)))
                
        if BufferManager.fx is None:
            BufferManager.fx = self.fx
        if BufferManager.fy is None:
            BufferManager.fy = self.fy
                
        return self.fx, self.fy
        
    def _get_buffer_QK_B_QKT(self, plot_bcoef):
        # Torch 全局设定
        device = torch.device("cpu")  # 也可以 cuda 加速
        dtype = torch.float32
        # subset parameters
        subset_r = self.config.subset_half_size if self.config.subset_half_size >= 3 else 3
        # ref_bcoef pad to avoid border checks for 6x6 block
        border = subset_r
        offset = 2
        # QK、梯度、参考图全部转 torch
        if not hasattr(self, "QK"):
            self._get_QK_QKdx_QKdxx()
        QK = _to_torch(self.QK, device, dtype)

        # reference and bcoef
        ref_bcoef = _to_torch(plot_bcoef, device, dtype)

        # ROI list and padded ROI's
        ROI_list = self._get_roiRegion()
        roi = ROI_list[0]          # shape: (H, W)
        H, W = roi.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        ys = ys.reshape(-1)
        xs = xs.reshape(-1)
        num_pts = ys.numel()

        # -------------------------
        # Pre-allocate buffers ONCE
        # -------------------------
        QK_B_QKT_6 = torch.zeros((num_pts, 6, 6), device=device, dtype=dtype)
        tmp6_buffer = torch.empty((6, 6), device=device, dtype=dtype)

        # Loop over points (only this single loop remains)
        for p_idx in tqdm(range(num_pts), desc="Computing QK B QK^T"):
            yc = ys[p_idx].item()
            xc = xs[p_idx].item()
            # ---- compute QK * block * QK^T and store (6x6 small) ----
            top = yc + border - offset
            left = xc + border - offset
            block = ref_bcoef[top:top + 6, left:left + 6]  # view or small array
            # small tmp created here (6x6) - unavoidable
            torch.matmul(QK, block, out=tmp6_buffer)
            torch.matmul(tmp6_buffer, QK.T, out=QK_B_QKT_6[p_idx])
            # if (p_idx + 1) % 1000 == 0:
            #     print(f"No.{p_idx + 1} has completed")
        # convert to numpy
        QK_B_QKT_6_np = _to_numpy(QK_B_QKT_6)
        QK_B_QKT_hash_map = {
            (int(ys[i]), int(xs[i])): QK_B_QKT_6_np[i]
            for i in range(num_pts)
        }
        
        # return buffers
        return QK_B_QKT_hash_map
    
def collate_fn(batch):
    return batch  

def _to_torch(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def _to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        device = getattr(x, "device", None)
        if device == "cuda":
            x_cpu = x.detach().to("cpu")
            np_x = x_cpu.numpy()
        else:
            np_x = x.numpy()
        if dtype is not None:
            return np_x.astype(dtype)
        return np_x

if __name__ == "__main__":
    from DIC_load_config import load_dic_config
    from scipy.io import savemat
    import time

    cfg = load_dic_config("./RD-DIC/config.json")
    imgGenDataset = Img_Dataset(cfg)
    
    imgGenDataset._get_QK_QKdx_QKdxx()
    print("_get_QK_QKdx_QKdxx over")
    
    start_time = time.time()
    imgGenDataset._get_image_gradient()
    total_time = time.time()-start_time
    print("_get_image_gradient over")
    print(f"cost {total_time}s")
    
    start_time = time.time()
    refImg, ref_bcoef = imgGenDataset._get_refImg()
    total_time = time.time()-start_time
    print("_get_refImg over")
    print(f"cost {total_time}s")
    
    imgGenDataset._get_roiRegion()
    print("_get_roiRegion over")
    
    

    
    
    
    