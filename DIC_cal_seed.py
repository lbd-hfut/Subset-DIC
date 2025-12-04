from PIL import Image
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List
import cv2
from tqdm import tqdm
from threading import RLock
from concurrent.futures import ThreadPoolExecutor, as_completed
from DIC_read_image import BufferManager
from DIC_icgn_newton import iterativesearch

class seed_math:
    def __init__(self, ROI_LIST:List[np.ndarray], config):
        
        self.config = config
        self.ROI_LIST = ROI_LIST
        self.seed_points_list = self.sample_kmeans()
        
        
    def sample_kmeans(self):
        if self.config.parallel_flag:
            n_points = self.config.max_workers
        else:
            n_points = 1
            
        seed_points_list = []
        for mask in self.ROI_LIST:
            ys, xs = np.nonzero(mask)
            pts = np.column_stack([xs, ys])          # (N,2) 所有 ROI 前景像素坐标

            if len(pts) < n_points:
                raise ValueError("ROI 中像素数量不足")

            # --- 特殊情况：n=1 ---
            if n_points == 1:
                idx = np.random.randint(0, len(pts))
                x, y = pts[idx]
                return [(int(x), int(y))]

            # --- 正常情况：K-means 聚类 ---
            kmeans = KMeans(n_clusters=n_points, n_init='auto').fit(pts)
            centers = np.rint(kmeans.cluster_centers_).astype(int)

            # --- 处理每个中心点 ---
            H, W = mask.shape

            for x, y in centers:
                # 中心点合法（落在 ROI 内）
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    seed_points_list.append((int(x), int(y)))
                    continue

                # 中心点无效 → 随机从 ROI 内重新采样
                idx = np.random.randint(0, len(pts))
                xr, yr = pts[idx]
                seed_points_list.append((int(xr), int(yr)))
        seed_points_list = np.array(seed_points_list, dtype=np.float32)
        # seed_points_list.append(result)
        return seed_points_list
    
    # ⭐⭐ 多线程求解所有种子点 ⭐⭐
    def solve_all_seed_points(self):
        subset_r = self.config.subset_half_size
        search_radius = self.config.search_radius
        max_iter = self.config.max_iterations
        cutoff_diffnorm = self.config.cutoff_diffnorm
        lambda_reg = self.config.lambda_reg
        x_offsets = np.arange(-subset_r, subset_r + 1, dtype=np.int32)
        y_offsets = np.arange(-subset_r, subset_r + 1, dtype=np.int32)
        xv, yv = np.meshgrid(x_offsets, y_offsets)  # shape (S,S)
        X_flat = xv.reshape(-1)   # (subset_area,)
        Y_flat = yv.reshape(-1)
        # 多线程求解全部 seed 点（使用 self.seed_points_list）
        def worker(seed_xy):
            cx, cy = int(seed_xy[0]), int(seed_xy[1])
            flag, defvector, corrcoef = cal_seed_point(
                cy=cy, cx=cx,
                X_flat=X_flat,
                Y_flat=Y_flat,
                subset_r=subset_r,
                search_radius=search_radius,
                max_iter=max_iter,
                cutoff_diffnorm=cutoff_diffnorm,
                lambda_reg=lambda_reg
            )
            return (cx, cy, flag, defvector, corrcoef)
        
        seed_points = self.seed_points_list
        n_points = len(seed_points)
        results = []
        if self.config.parallel_flag:
            max_workers = self.config.max_workers
        else:
            max_workers = 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(worker, seed_xy): seed_xy
                for seed_xy in seed_points
            }
            with tqdm(total=n_points, desc="Solving seed points", unit="pt") as pbar:
                for future in as_completed(future_to_seed):
                    results.append(future.result())
                    pbar.update(1)
        return results

    
def cal_seed_point(
    cy: int, cx: int, 
    X_flat: np.ndarray,
    Y_flat: np.ndarray,
    subset_r: int = 15, 
    search_radius: int = 10,
    max_iter: int = 20,
    cutoff_diffnorm: float = 1e-3,
    lambda_reg: float = 1e-3
):
    mask = None
    for roi_idx, roi in enumerate(BufferManager.mask):
        if roi[cy, cx]:
            mask = roi
            mask_idx = roi_idx
            break
    if mask is None:
        raise ValueError(f"种子点 (y:{cy}, x:{cx}) 不在ROI内")
    mask_pad = BufferManager.mask_pad[mask_idx]
    
    u0, v0 = coarse_search_int(cy, cx, mask, subset_r, search_radius)
    defvector_init = np.zeros(12)
    defvector_init[0], defvector_init[1] = u0, v0
    
    py = cy + subset_r
    px = cx + subset_r
    y0, y1 = py - subset_r, py + subset_r + 1
    x0, x1 = px - subset_r, px + subset_r + 1
    mask_seg = mask_pad[y0:y1, x0:x1].reshape(-1)
    valid_idx = np.nonzero(mask_seg)[0]
    dx, dy = X_flat[valid_idx], Y_flat[valid_idx]
    
    flag, defvector, corrcoef = iterativesearch(
        defvector_init=defvector_init, 
        xc=cx, yc=cy, dx=dx, dy=dy,
        max_iter=max_iter,
        cutoff_diffnorm=cutoff_diffnorm,
        lambda_reg=lambda_reg
        )
    return flag, defvector, corrcoef
    
def coarse_search_int(cy, cx, mask, subset_r, search_radius):
    H, W = BufferManager.defImg.shape
    y0 = cy - subset_r; y1 = cy + subset_r + 1
    x0 = cx - subset_r; x1 = cx + subset_r + 1   
    # clip to image
    y0c = max(0, y0); y1c = min(H, y1)
    x0c = max(0, x0); x1c = min(W, x1)
    h = y1c - y0c; w = x1c - x0c
    ref_patch = BufferManager.refImg[y0c:y1c, x0c:x1c].astype(np.float32)
    mask_patch = mask[y0c:y1c, x0c:x1c].astype(np.float32)
    
    y_min = int(max(0, y0c - search_radius))
    y_max = int(min(H, y1c + search_radius + 1))
    x_min = int(max(0, x0c - search_radius))
    x_max = int(min(W, x1c + search_radius + 1))
    
    search_def = BufferManager.defImg[y_min:y_max, x_min:x_max].astype(np.float32)
    res = cv2.matchTemplate(
        search_def, ref_patch,
        cv2.TM_CCOEFF_NORMED,
        mask=mask_patch
    )
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    best_y = y_min + max_loc[1]
    best_x = x_min + max_loc[0]
    
    dy_int = best_y - y0c
    dx_int = best_x - x0c
    return dy_int, dx_int
    
    
if __name__ == "__main__":
    from DIC_load_config import load_dic_config
    from DIC_read_image import Img_Dataset, collate_fn
    from scipy.io import savemat
    import time

    cfg = load_dic_config("./RD-DIC/config.json")
    imgGenDataset = Img_Dataset(cfg)
    
    imgGenDataset._get_QK_QKdx_QKdxx()
    imgGenDataset._get_roiRegion()
    # print("_get_QK_QKdx_QKdxx over")
    
    start_time = time.time()
    refImg, ref_bcoef = imgGenDataset._get_refImg()
    total_time = time.time()-start_time
    # print("_get_refImg over")
    print(f"cost {total_time}s")
    print()
    
    start_time = time.time()
    imgGenDataset._get_image_gradient()
    total_time = time.time()-start_time
    # print("_get_image_gradient over")
    print(f"cost {total_time}s")
    print()
    
    seed_solver = seed_math(ROI_LIST=BufferManager.mask, config=cfg)
    
    img_loader = torch.utils.data.DataLoader(
        imgGenDataset, batch_size=1, 
        shuffle=False, collate_fn=collate_fn)
    
    for idx, DimageL in enumerate(img_loader):
        start_time = time.time()
        results = seed_solver.solve_all_seed_points()
        total_time = time.time()-start_time
        print(f"cost {total_time*1000}ms")
        print()
        # 2️⃣ 打印 defvector 的前两个元素
        print("\n=== defvector 前两个值 ===")
        for (cx, cy, flag, defvector, corrcoef) in results:
            print(f"({cx},{cy}) flag={flag}: {defvector[:2]}, Ncc[{corrcoef}]")
        
    

