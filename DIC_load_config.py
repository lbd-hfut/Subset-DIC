import json
import os
from types import SimpleNamespace

def load_dic_config(json_path):
    """
    读取 DIC 配置文件（JSON），返回 SimpleNamespace 类型 cfg 对象：
        cfg.subset_half_size
        cfg.step
        cfg.shape_order
        ...
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"配置文件不存在：{json_path}")

    # === 读取 JSON 文件 ===
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # === 默认参数 ===
    default_cfg = {
        "input_dir": "./case/case1/",
        "output_dir": "./case/case1/result/",
        "subset_half_size": 15,
        "step": 25,
        "shape_order": 2,
        "search_radius": 10,
        "max_iterations": 50,
        "cutoff_diffnorm": 1e-6,
        "lambda_reg": 1e-3,
        "parallel": False,
        "max_workers": 8,
        "smooth_flage": False,
        "smooth_sigma": 1.0,
        "strain_calculate_flage": True,
        "strain_window_half_size": 15,
        "show_plot": True
    }

    # === 补齐缺失的默认值 ===
    for key, val in default_cfg.items():
        if key not in cfg:
            cfg[key] = val

    # === 强制类型转换，保证安全 ===
    int_keys = [
        "subset_half_size", "step", "shape_order", "search_radius",
        "max_iterations", "max_workers", "strain_window_half_size"
    ]
    float_keys = ["smooth_sigma", "lambda_reg", "cutoff_diffnorm"]
    bool_keys = ["parallel", "smooth_flage", "strain_calculate_flage", "show_plot"]

    for k in int_keys:
        cfg[k] = int(cfg[k])
    for k in float_keys:
        cfg[k] = float(cfg[k])
    for k in bool_keys:
        cfg[k] = bool(cfg[k])

    # === 确保路径最后有 "/" ===
    for key in ["input_dir", "output_dir"]:
        if not cfg[key].endswith("/"):
            cfg[key] += "/"

    # === 转换为 SimpleNamespace，支持 cfg.xxx 访问 ===
    return SimpleNamespace(**cfg)


if __name__ == "__main__":
    cfg = load_dic_config("./RD-DIC/config.json")

    for key, value in vars(cfg).items():
        print(f"{key}: {value}")
    print("=============================")
