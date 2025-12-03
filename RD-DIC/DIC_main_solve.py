import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from collections import deque
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import cv2
from tqdm import tqdm
from scipy.io import savemat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from DIC_read_image import BufferManager, Img_Dataset, collate_fn
from DIC_icgn_newton import iterativesearch
from DIC_load_config import load_dic_config
from DIC_cal_seed import seed_math

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 
seed_everything(42)

class Subset_DIC_solver:
    def __init__(self, config):
        self.subset_r = config.subset_half_size
        self.step = config.step
        self.shape_order = config.shape_order
        self.max_iterations = config.max_iterations
        self.cutoff_diffnorm = config.cutoff_diffnorm
        self.lambda_reg = config.lambda_reg
        self.parallel_flag = config.parallel_flag
        self.smooth_flage = config.smooth_flage
        self.smooth_sigma = config.smooth_sigma
        self.strain_calculate_flage = config.strain_calculate_flage
        self.strain_window_half_size = config.strain_window_half_size
        self.show_plot = config.show_plot
        self.output_dir = config.output_dir
        
    def bfs_region_grow(self, seed_result):
        H, W = BufferManager.refImg.shape
        ROI_list = BufferManager.mask
        
        seed_valid_result = []
        seed_valid_pos = []
        for (cx, cy, flag, defvector, corrcoef) in seed_result:
            if flag and corrcoef < 0.1:
                seed_valid_result.append((cx, cy, flag, defvector, corrcoef))
                seed_valid_pos.append((cx, cy))
            else:
                continue
            
        threaddiagram = -1 * np.ones((H, W), dtype=int)
        queues = [deque() for _ in seed_valid_pos]
        
        for idx, (x, y) in enumerate(seed_valid_pos):
            threaddiagram[y, x] = idx
            queues[idx].append((y, x))
        
        # 多源扩散
        while any(queues):
            for i in range(len(seed_valid_pos)):
                if not queues[i]:
                    continue

                y, x = queues[i].popleft()

                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        for mask in ROI_list:
                            if mask[ny, nx]:
                                break
                        if mask[ny, nx] and threaddiagram[ny, nx] == -1:
                            threaddiagram[ny, nx] = i
                            queues[i].append((ny, nx))
        return threaddiagram, seed_valid_result
    
    def visualize_seed_BFS(self, idx, seed_result, threaddiagram):
        plt.figure(figsize=(6,6))
        plt.imshow(threaddiagram, cmap='tab10')
        plt.colorbar(label="Region ID")
        xs = [s[0] for s in seed_result]
        ys = [s[1] for s in seed_result]
        plt.scatter(xs, ys, c='red', s=80, label="Seeds")
        plt.legend()
        plt.title("BFS Region Growing from Seeds")
        plt.gca().invert_yaxis()
        
        # === 保存图像 ===
        if self.show_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, f"seed_bfs_region_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.close()
        
def main(config_path):
    cfg = load_dic_config(config_path)
    DIC_Solver = Subset_DIC_solver(cfg)
    imgGenDataset = Img_Dataset(cfg)
    imgGenDataset._get_QK_QKdx_QKdxx()
    imgGenDataset._get_roiRegion()
    
    imgGenDataset._get_refImg()
    imgGenDataset._get_image_gradient()
    
    seed_solver = seed_math(ROI_LIST=BufferManager.mask, config=cfg)
    img_loader = torch.utils.data.DataLoader(
        imgGenDataset, batch_size=1, 
        shuffle=False, collate_fn=collate_fn)
    
    for idx, DimageL in enumerate(img_loader):
        seed_result = seed_solver.solve_all_seed_points()
        threaddiagram, seed_valid_result = DIC_Solver.bfs_region_grow(seed_result)
        DIC_Solver.visualize_seed_BFS(idx, seed_valid_result, threaddiagram)
    
    
    
if __name__ == "__main__":
    config_path = "./RD-DIC/config.json"
    main(config_path)
        

        
        

