# Subset-DIC (Digital Image Correlation)

This repository provides an implementation of a **subset-based two-dimensional Digital Image Correlation (2D-DIC)** algorithm.  
The framework supports **second-order shape functions**, **IC-GN (Inverse Compositional Gauss–Newton) optimization**, optional **parallel computation**, displacement smoothing, and strain calculation.  
It is designed for accurate displacement and strain measurement under complex deformation conditions.

After configuring `config.json`, the entire DIC analysis can be executed by running a single script.

---

## Project Structure

```text
Subset-DIC/
│
├── DIC_main_solve.py        # Main entry point
├── DIC_load_config.py      # Configuration loader
├── DIC_read_image.py       # Image reading and preprocessing
├── DIC_cal_seed.py         # Seed point / initial guess calculation
├── DIC_icgn_newton.py      # IC-GN solver
├── DIC_post_processing.py  # Post-processing (displacement & strain)
├── DIC_result_plot.py      # Visualization of results
├── DIC_save_results.py     # Saving numerical results
├── DIC_threaddiagram.py    # Parallel / threading control
│
├── config.json             # Configuration file
└── case/
    └── star/
        ├── ref.png         # Reference image
        ├── def.png         # Deformed image
        └── Subset/         # Output directory
```

## Configuration (`config.json`)

All parameters are specified in `config.json`.  
An example configuration is shown below:

```json
{
    "input_dir": "./case/star/",
    "output_dir": "./case/star/Subset/",
    "subset_half_size": 10,
    "step": 1,
    "shape_order": 2,
    "search_radius": 20,
    "max_iterations": 50,
    "cutoff_diffnorm": 1e-5,
    "lambda_reg": 1e-3,
    "parallel_flag": false,
    "max_workers": 1,
    "smooth_flage": true,
    "smooth_sigma": 2.0,
    "strain_calculate_flage": true,
    "strain_window_half_size": 51,
    "show_plot": true
}
```

### Parameter Description

| Parameter                  | Description |
|---------------------------|-------------|
| `input_dir`               | Directory containing reference and deformed images |
| `output_dir`              | Directory for saving results |
| `subset_half_size`        | Half size of subset (subset size = 2h + 1) |
| `step`                    | Subset step size (in pixels) |
| `shape_order`             | Shape function order (1: first-order, 2: second-order) |
| `search_radius`           | Search radius for correlation matching |
| `max_iterations`          | Maximum IC-GN iterations |
| `cutoff_diffnorm`         | Convergence threshold |
| `lambda_reg`              | Regularization parameter |
| `parallel_flag`           | Enable parallel computation |
| `max_workers`             | Number of worker threads |
| `smooth_flage`            | Enable displacement smoothing |
| `smooth_sigma`            | Gaussian smoothing parameter |
| `strain_calculate_flage`  | Enable strain calculation |
| `strain_window_half_size` | Half window size for strain calculation |
| `show_plot`               | Display result plots |


## How to Run

1. Prepare the reference and deformed images and place them in `input_dir`.
2. Modify `config.json` according to your experimental setup.
3. Run the main solver:

```bash
python DIC_main_solve.py
```

The program will automatically perform:

- Subset generation
- IC-GN iterative displacement solving
- Optional displacement smoothing
- Optional strain field calculation
- Result visualization and saving

## Output

The results saved in `output_dir` typically include:

- Displacement fields (`u`, `v`)
- Strain fields (`ε_xx`, `ε_yy`, `ε_xy`)
- Visualization images
- Numerical result files (e.g., `.npy`, `.mat`)

## Features

- Subset-based DIC framework
- IC-GN optimization
- Support for second-order shape functions
- Regularization for robust convergence
- Optional parallel computation
- Displacement smoothing and strain post-processing
- Modular and extensible code structure

---

## Notes

- Subset size and step size significantly affect accuracy and stability.
- Second-order shape functions are recommended for non-uniform deformation.
- Parallel execution may increase memory usage.

---

## License

This project is intended for research and educational use.
