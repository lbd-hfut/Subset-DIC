import os
from scipy.io import savemat

def DIC_save_mat(idx, Subset_DIC_Buffer, output_dir):
    """
    Save selected attributes of Subset_DIC_Buffer into a MATLAB .mat file.
    The saved struct allows MATLAB to access fields using dot notation.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Subset_DIC_{idx+1:03d}.mat")

    # 构建要保存的字典，对应 MATLAB struct
    dic_struct = {
        'subset_r': Subset_DIC_Buffer.subset_r,
        'step': Subset_DIC_Buffer.step,
        'plot_calcpoints': Subset_DIC_Buffer.plot_calcpoints,
        'plot_validpoints': Subset_DIC_Buffer.plot_validpoints,
        'plot_u': Subset_DIC_Buffer.plot_u,
        'plot_v': Subset_DIC_Buffer.plot_v,
        'plot_ex': Subset_DIC_Buffer.plot_ex,
        'plot_ey': Subset_DIC_Buffer.plot_ey,
        'plot_rxy': Subset_DIC_Buffer.plot_rxy,
        'plot_corrcoef': Subset_DIC_Buffer.plot_corrcoef,
        'seeds_info': Subset_DIC_Buffer.seeds_info
    }

    # 使用 savemat 保存，结构体形式
    savemat(save_path, {'DIC_result': dic_struct})
    print(f"Saved MATLAB .mat file: {save_path}")
