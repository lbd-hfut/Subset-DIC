import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_seed_BFS(idx, seed_result, threaddiagram, output_dir):
    plt.figure(figsize=(6,6))
    plt.imshow(threaddiagram, cmap='tab20')
    plt.colorbar(label="Region ID")
    xs = [s[0] for s in seed_result]
    ys = [s[1] for s in seed_result]
    plt.scatter(xs, ys, c='red', s=80, label="Seeds")
    plt.legend()
    plt.title("BFS Region Growing from Seeds")
    # === 保存图像 ===
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"seed_bfs_region_{idx+1:03d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_imshow(idx, Subset_DIC_Buffer, output_dir):
    """
    Visualize Subset_DIC_Buffer results and save images.
    First row: u, v, valid points
    Second row: ex, ey, ry (create 0-matrix if None)
    Colorbar for u/v based on valid points, ex/ey/ry fixed to [-1, 1] if zeros
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"result_imshow_{idx+1:03d}.png")

    # 获取数据
    u = Subset_DIC_Buffer.plot_u
    v = Subset_DIC_Buffer.plot_v
    ex = Subset_DIC_Buffer.plot_ex
    ey = Subset_DIC_Buffer.plot_ey
    rxy = Subset_DIC_Buffer.plot_rxy
    valid = Subset_DIC_Buffer.plot_calcpoints

    # 检查 ex/ey/rxy 是否为 None，如果是则创建 0 矩阵
    shape = u.shape if u is not None else (1,1)
    ex = np.zeros(shape) if ex is None else ex
    ey = np.zeros(shape) if ey is None else ey
    rxy = np.zeros(shape) if rxy is None else rxy

    # 准备绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行
    for ax, data, title in zip(axes[0], [u, v, valid], ["u", "v", "valid points"]):
        if title != "valid points":
            data_to_show = np.where(valid, data, np.nan)
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
            im = ax.imshow(data_to_show, cmap='jet', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(data, cmap='gray')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    # 第二行
    for ax, data, title in zip(axes[1], [ex, ey, rxy], ["ex", "ey", "rxy"]):
        data_to_show = np.where(valid, data, np.nan)
        # 如果全为零，则固定 colorbar 为 [-1, 1]
        if np.allclose(data, 0):
            vmin, vmax = -1, 1
        else:
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
        im = ax.imshow(data_to_show, cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_contourf(idx, Subset_DIC_Buffer, output_dir):
    """
    Visualize Subset_DIC_Buffer results with contourf and save images.
    First row: u, v, valid points
    Second row: ex, ey, rxy (create 0-matrix if None)
    Colorbar for u/v based on valid points, ex/ey/rxy fixed to [-1, 1] if zeros
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"result_contourf_{idx+1}.png")

    # 获取数据
    u = Subset_DIC_Buffer.plot_u
    v = Subset_DIC_Buffer.plot_v
    ex = Subset_DIC_Buffer.plot_ex
    ey = Subset_DIC_Buffer.plot_ey
    rxy = Subset_DIC_Buffer.plot_rxy
    valid = Subset_DIC_Buffer.plot_calcpoints

    # 检查 ex/ey/rxy 是否为 None，如果是则创建 0 矩阵
    shape = u.shape if u is not None else (1,1)
    ex = np.zeros(shape) if ex is None else ex
    ey = np.zeros(shape) if ey is None else ey
    rxy = np.zeros(shape) if rxy is None else rxy

    # 准备绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行
    for ax, data, title in zip(axes[0], [u, v, valid], ["u", "v", "valid points"]):
        if title != "valid points":
            data_to_show = np.where(valid, data, np.nan)
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
            im = ax.contourf(data_to_show, levels=100, cmap='jet', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.invert_yaxis()
        else:
            im = ax.imshow(data, cmap='gray')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    # 第二行
    for ax, data, title in zip(axes[1], [ex, ey, rxy], ["ex", "ey", "rxy"]):
        data_to_show = np.where(valid, data, np.nan)
        if np.allclose(data, 0):
            vmin, vmax = -1, 1
        else:
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
        im = ax.contourf(data_to_show, levels=100, cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()