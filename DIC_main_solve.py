import torch
import numpy as np
import os
import random
from scipy.ndimage import zoom
from collections import deque
from tqdm import tqdm
from threading import RLock, Event
from concurrent.futures import ThreadPoolExecutor, as_completed

from DIC_read_image import BufferManager, Img_Dataset, collate_fn
from DIC_icgn_newton import iterativesearch, SUCCESS, FAILED
from DIC_load_config import load_dic_config
from DIC_cal_seed import seed_math, cal_seed_point
from DIC_result_plot import visualize_seed_BFS, visualize_imshow, visualize_contourf
from DIC_threaddiagram import bfs_region_grow
from DIC_post_processing import DIC_smooth_Displacement, DIC_Strain_from_Displacement
from DIC_save_results import DIC_save_mat

class Subset_DIC_Buffer:
    # DIC params
    subset_r = None
    search_radius = None
    step = 1
    max_iter = None
    cutoff_diffnorm = None
    lambda_reg = None
    parallel_flag = None
    max_workers = None
    # need init
    X_flat = None
    Y_flat = None
    plot_calcpoints = None
    plot_validpoints = None
    plot_u = None
    plot_v = None
    plot_ex = None
    plot_ey = None
    plot_rxy = None
    plot_corrcoef = None
    threaddiagram = None
    seeds_info = None
    w_origin = None
    h_origin = None
    w_resize = None
    h_resize = None
    # write data lock
    data_lock = RLock()

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
        Subset_DIC_Buffer.subset_r = config.subset_half_size
        Subset_DIC_Buffer.search_radius = config.search_radius
        Subset_DIC_Buffer.shape_order = config.shape_order
        Subset_DIC_Buffer.max_iter = config.max_iterations
        Subset_DIC_Buffer.cutoff_diffnorm = config.cutoff_diffnorm
        Subset_DIC_Buffer.lambda_reg = config.lambda_reg
        Subset_DIC_Buffer.parallel_flag = config.parallel_flag
        Subset_DIC_Buffer.max_workers = config.max_workers
        
        self.smooth_flage = config.smooth_flage
        self.smooth_sigma = config.smooth_sigma
        self.strain_calculate_flage = config.strain_calculate_flage
        self.strain_window_half_size = config.strain_window_half_size
        self.show_plot = config.show_plot
        self.output_dir = config.output_dir
        
        self.stop_event = Event()    # 新增全局停止事件
        
    def _init_DIC_BUFFER(self, seed_result, threaddiagram):
        Subset_DIC_Buffer.plot_calcpoints  = np.zeros_like(BufferManager.refImg, dtype=bool)
        Subset_DIC_Buffer.plot_validpoints = np.zeros_like(BufferManager.refImg, dtype=bool)
        Subset_DIC_Buffer.plot_u = np.zeros_like(BufferManager.refImg, dtype=np.float32)
        Subset_DIC_Buffer.plot_v = np.zeros_like(BufferManager.refImg, dtype=np.float32)
        Subset_DIC_Buffer.plot_corrcoef = np.ones_like(BufferManager.refImg, dtype=np.float32)
        Subset_DIC_Buffer.threaddiagram = threaddiagram
        seed_valid_info = []
        num_thread = 0
        for (cx, cy, flag, defvector, corrcoef) in seed_result:
            for roi_idx, mask in enumerate(BufferManager.mask):
                num_region = roi_idx
            seed_valid_info.append((cx, cy, defvector, corrcoef, num_region, num_thread))
            num_thread = num_thread + 1
        Subset_DIC_Buffer.seeds_info = seed_valid_info
        x_offsets = np.arange(-Subset_DIC_Buffer.subset_r, Subset_DIC_Buffer.subset_r + 1, dtype=np.int32)
        y_offsets = np.arange(-Subset_DIC_Buffer.subset_r, Subset_DIC_Buffer.subset_r + 1, dtype=np.int32)
        xv, yv = np.meshgrid(x_offsets, y_offsets)  # shape (S,S)
        Subset_DIC_Buffer.X_flat = xv.reshape(-1)   # (subset_area,)
        Subset_DIC_Buffer.Y_flat = yv.reshape(-1)
        
    def solve(self, idx):
        self.stop_event.clear()  # 每次开始前清空停止标志
        queues = [deque() for _ in Subset_DIC_Buffer.seeds_info]
        n_points = 0
        for mask in BufferManager.mask:
            n_points = n_points + np.sum(mask.reshape(-1))
            del mask
        pbar_lock = RLock()
        global_pbar = tqdm(total=n_points, desc=f"Solving No.{idx+1:03d} defImg", unit="pt")
        # 多线程求解seedBFS扩散区域
        def worker(queue, seed_info):
            analysis_queue(
                queue=queue,
                seed_info=seed_info,
                pbar=global_pbar,
                pbar_lock=pbar_lock,
                stop_event=self.stop_event
            )
            
        for i in range(len(queues)):
            worker(queues[i], Subset_DIC_Buffer.seeds_info[i])
        global_pbar.close()


def analysis_queue(queue, seed_info, pbar, pbar_lock, stop_event):
    cx, cy, defvector, corrcoef, num_region, num_thread = seed_info
    queue.append((cx, cy, defvector, corrcoef, num_region, num_thread))
    Subset_DIC_Buffer.plot_validpoints[cy,cx] = True
    Subset_DIC_Buffer.plot_calcpoints[cy, cx] = True
    while queue:
        if stop_event.is_set():   # ★ 线程立即退出
            return
        seed_info = queue.popleft()
        cx, cy, defvector, corrcoef, num_region, num_thread = seed_info
        # 写入结果图
        with Subset_DIC_Buffer.data_lock:
            Subset_DIC_Buffer.plot_u[cy,cx] = defvector[0]
            Subset_DIC_Buffer.plot_v[cy,cx] = defvector[1]
            Subset_DIC_Buffer.plot_corrcoef[cy,cx] = corrcoef
        # 四邻域
        step = Subset_DIC_Buffer.step
        neighs = [
                    (cx, cy - step),
                    (cx + step, cy),
                    (cx, cy + step),
                    (cx - step, cy)
                ]
        for x, y in neighs:
            dx, dy = x - cx, y - cy
            u = (defvector[0] +
                defvector[2] * dx +
                defvector[4] * dy +
                0.5 * defvector[6] * dx * dx +
                defvector[8] * dx * dy +
                0.5 * defvector[10] * dy * dy)
            defvector[0] = u
            v = (defvector[1] +
                defvector[3] * dx +
                defvector[5] * dy +
                0.5 * defvector[7] * dx * dx +
                defvector[9] * dx * dy +
                0.5 * defvector[11] * dy * dy)
            defvector[1] = v
            analyzepoint(
                queue=queue, x=x, y=y, 
                defvector_init = defvector,
                num_region = num_region,
                num_thread = num_thread, 
                pbar = pbar, 
                pbar_lock = pbar_lock,
                stop_event = stop_event
                )
        if queue:
            continue
        else:
            with Subset_DIC_Buffer.data_lock:
                # 查找该线程未计算点
                ys_u, xs_u = np.where(
                    (Subset_DIC_Buffer.threaddiagram == num_thread) & \
                        (~Subset_DIC_Buffer.plot_calcpoints) & \
                            BufferManager.mask[num_region]
                    )
            Num_left_points =len(ys_u)
            if Num_left_points== 0:
                continue
            else:
                random_idx = np.random.randint(0, Num_left_points)
                seed_y = ys_u[random_idx]
                seed_x = xs_u[random_idx]
                outstate, defvector, corrcoef = cal_seed_point(
                    cy=seed_y, cx=seed_x, 
                    X_flat=Subset_DIC_Buffer.X_flat, 
                    Y_flat=Subset_DIC_Buffer.Y_flat, 
                    subset_r=Subset_DIC_Buffer.subset_r,
                    search_radius=Subset_DIC_Buffer.search_radius, 
                    max_iter=Subset_DIC_Buffer.max_iter,
                    cutoff_diffnorm=Subset_DIC_Buffer.cutoff_diffnorm, 
                    lambda_reg=Subset_DIC_Buffer.lambda_reg
                )
                if (outstate == SUCCESS and corrcoef < 1.0):
                    Subset_DIC_Buffer.plot_validpoints[y,x] = True
                print(f" 新种子点 ({x},{y}) flag={outstate}: {defvector[:2]}, Ncc[{corrcoef}]")
                paramvector = (seed_x, seed_y, defvector, corrcoef, num_region, num_thread)
                queue.append(paramvector)
                Subset_DIC_Buffer.plot_calcpoints[y,x] = True
                if pbar is not None and pbar_lock is not None:
                    with pbar_lock:
                        pbar.update(1)

def analyzepoint(queue, x, y, defvector_init, num_region, num_thread, pbar, pbar_lock, stop_event):
    if stop_event.is_set():   # 立即退出该点计算
        return
    cutoff_corrcoef = 2.0
    cutoff_disp = 1
    H, W = BufferManager.mask[num_region].shape
    if x < 0 or x >= W or y < 0 or y >= H:
        return
    if not BufferManager.mask[num_region][y, x]:
        return
    if Subset_DIC_Buffer.plot_calcpoints[y,x]:
        return
    if Subset_DIC_Buffer.threaddiagram[y, x] != num_thread:
        return
    # ---------------- 执行 calcpoint ----------------
    outstate, defvector, corrcoef = cal_point(
        cy=y, cx=x, 
        defvector_init=defvector_init,
        num_region=num_region
        )
    # ---------------- 加入队列 ----------------
    if (outstate == SUCCESS and
        corrcoef < cutoff_corrcoef and
        abs(defvector[0] - defvector_init[0]) < cutoff_disp and
        abs(defvector[1] - defvector_init[1]) < cutoff_disp):
        paramvector = (x, y, defvector, corrcoef, num_region, num_thread)
        queue.append(paramvector)
        Subset_DIC_Buffer.plot_validpoints[y,x] = True
    else:
        # 失败了再从新整像素搜索和亚像素匹配执行，re_cal_failed_points
        outstate, defvector, corrcoef = re_cal_failed_points(cy=y, cx=x, num_region=num_region)
        print(f" 重新匹配 ({x},{y}) flag={outstate}: {defvector[:2]}, Ncc[{corrcoef}]")
        if (outstate == SUCCESS and
            corrcoef < cutoff_corrcoef):
            paramvector = (x, y, defvector, corrcoef, num_region, num_thread)
            queue.append(paramvector)
            Subset_DIC_Buffer.plot_validpoints[y,x] = True
        else:
            with Subset_DIC_Buffer.data_lock:
                Subset_DIC_Buffer.plot_u[y,x] = defvector[0]
                Subset_DIC_Buffer.plot_v[y,x] = defvector[1]
                Subset_DIC_Buffer.plot_corrcoef[y,x] = corrcoef
    # ---------------- 标记已计算 ----------------
    Subset_DIC_Buffer.plot_calcpoints[y,x] = True
    # ---------------- 线程更新进度 ----------------
    if pbar is not None and pbar_lock is not None:
        with pbar_lock:
            pbar.update(1)

def cal_point(
    cy: int, cx: int, 
    defvector_init: np.ndarray,
    num_region: int
):
    mask_pad = BufferManager.mask_pad[num_region]
    py = cy + Subset_DIC_Buffer.subset_r
    px = cx + Subset_DIC_Buffer.subset_r
    y0, y1 = py - Subset_DIC_Buffer.subset_r, py + Subset_DIC_Buffer.subset_r + 1
    x0, x1 = px - Subset_DIC_Buffer.subset_r, px + Subset_DIC_Buffer.subset_r + 1
    mask_seg = mask_pad[y0:y1, x0:x1].reshape(-1)
    valid_idx = np.nonzero(mask_seg)[0]
    dx, dy = Subset_DIC_Buffer.X_flat[valid_idx], Subset_DIC_Buffer.Y_flat[valid_idx]
    
    flag, defvector, corrcoef = iterativesearch(
        defvector_init=defvector_init, 
        xc=cx, yc=cy, dx=dx, dy=dy,
        max_iter=Subset_DIC_Buffer.max_iter,
        cutoff_diffnorm=Subset_DIC_Buffer.cutoff_diffnorm,
        lambda_reg=Subset_DIC_Buffer.lambda_reg
        )
    return flag, defvector, corrcoef

def re_cal_failed_points(
    cy: int, cx: int, 
    num_region: int
):
    flag, defvector, corrcoef = cal_seed_point(
        cy=cy, cx=cx, 
        X_flat=Subset_DIC_Buffer.X_flat, 
        Y_flat=Subset_DIC_Buffer.Y_flat, 
        subset_r=Subset_DIC_Buffer.subset_r,
        search_radius=Subset_DIC_Buffer.search_radius, 
        max_iter=Subset_DIC_Buffer.max_iter,
        cutoff_diffnorm=Subset_DIC_Buffer.cutoff_diffnorm, 
        lambda_reg=Subset_DIC_Buffer.lambda_reg
    )
    return flag, defvector, corrcoef


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
        threaddiagram, seed_valid_result = bfs_region_grow(seed_result)
        # solve subset dic
        DIC_Solver._init_DIC_BUFFER(seed_valid_result, threaddiagram)
        DIC_Solver.solve(idx)
        if (BufferManager.w_origin != BufferManager.w_resize) or \
            (BufferManager.h_origin != BufferManager.h_resize):
                yscale = BufferManager.h_origin / BufferManager.h_resize
                xcalse = BufferManager.w_origin / BufferManager.w_resize
                Subset_DIC_Buffer.plot_u = Subset_DIC_Buffer.plot_u * xcalse
                Subset_DIC_Buffer.plot_v = Subset_DIC_Buffer.plot_v * yscale
                DIC_STEP = [xcalse, yscale]
        else:
            DIC_STEP = [1, 1]
        Subset_DIC_Buffer.w_origin = BufferManager.w_origin
        Subset_DIC_Buffer.h_origin = BufferManager.h_origin
        Subset_DIC_Buffer.w_resize = BufferManager.w_resize
        Subset_DIC_Buffer.h_resize = BufferManager.h_resize
        # post processing
        if DIC_Solver.smooth_flage:
            Subset_DIC_Buffer.plot_u, Subset_DIC_Buffer.plot_v = \
                DIC_smooth_Displacement(
                    Subset_DIC_Buffer.plot_u, Subset_DIC_Buffer.plot_v,
                    Subset_DIC_Buffer.plot_calcpoints, DIC_Solver.smooth_sigma)
        if DIC_Solver.strain_calculate_flage:
            Subset_DIC_Buffer.plot_ex, Subset_DIC_Buffer.plot_ey, Subset_DIC_Buffer.plot_rxy = \
                DIC_Strain_from_Displacement(
                    Subset_DIC_Buffer.plot_u, Subset_DIC_Buffer.plot_v,
                    Subset_DIC_Buffer.plot_calcpoints, DIC_STEP,
                    DIC_Solver.strain_window_half_size
                )
        if (BufferManager.w_origin != BufferManager.w_resize) or \
            (BufferManager.h_origin != BufferManager.h_resize):
            # 几何插值
            zoom_factor_x = BufferManager.w_origin / BufferManager.w_resize
            zoom_factor_y = BufferManager.h_origin / BufferManager.h_resize

            Subset_DIC_Buffer.plot_u = zoom(Subset_DIC_Buffer.plot_u,
                (zoom_factor_y, zoom_factor_x),order=3)
            Subset_DIC_Buffer.plot_v = zoom(Subset_DIC_Buffer.plot_v,
                (zoom_factor_y, zoom_factor_x),order=3)
            Subset_DIC_Buffer.plot_ex = zoom(Subset_DIC_Buffer.plot_ex,
                (zoom_factor_y, zoom_factor_x),order=3)
            Subset_DIC_Buffer.plot_ey = zoom(Subset_DIC_Buffer.plot_ey,
                (zoom_factor_y, zoom_factor_x),order=3)
            Subset_DIC_Buffer.plot_rxy = zoom(Subset_DIC_Buffer.plot_rxy,
                (zoom_factor_y, zoom_factor_x),order=3)
            Subset_DIC_Buffer.plot_validpoints = zoom(Subset_DIC_Buffer.plot_validpoints,
                (zoom_factor_y, zoom_factor_x),order=0)
            Subset_DIC_Buffer.plot_calcpoints = zoom(Subset_DIC_Buffer.plot_calcpoints,
                (zoom_factor_y, zoom_factor_x),order=0)
            
        # visualize
        visualize_seed_BFS(idx, seed_valid_result, threaddiagram, DIC_Solver.output_dir)
        visualize_imshow(idx, Subset_DIC_Buffer, DIC_Solver.output_dir)
        visualize_contourf(idx, Subset_DIC_Buffer, DIC_Solver.output_dir)
        # save
        DIC_save_mat(idx, Subset_DIC_Buffer, DIC_Solver.output_dir)
        
if __name__ == "__main__":
    config_path = "./config.json"
    main(config_path)
        

        
        

