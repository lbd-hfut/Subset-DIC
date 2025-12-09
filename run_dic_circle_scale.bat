@echo off
setlocal enabledelayedexpansion

:: 1) Activate conda
call conda activate dic

:: 2) Enter project directory
cd /d C:\01project\Subset-DIC

:: 3) Data root path
set DATA_ROOT=C:\01project\DIC_boundary_test_data-main\case\circle\scale

echo -----------------------------------------------
echo   Start scanning path: %DATA_ROOT%
echo -----------------------------------------------
echo.

for /d %%D in ("%DATA_ROOT%\*") do (
    echo Found subfolder: %%D

    :: Windows paths
    set INPUT_DIR=%%D\
    set OUTPUT_DIR=%%D\Subset\

    :: Only convert to UNIX when writing into JSON
    set INPUT_DIR_UNIX=!INPUT_DIR:\=/!
    set OUTPUT_DIR_UNIX=!OUTPUT_DIR:\=/!

    echo Using paths for JSON:
    echo     !INPUT_DIR_UNIX!
    echo     !OUTPUT_DIR_UNIX!
    echo.

    :: Create output dir (Windows path!)
    if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"

    :: Write JSON (use CALL to ensure expansion)
    > config.json echo {
    call echo     "input_dir": "!INPUT_DIR_UNIX!",>> config.json
    call echo     "output_dir": "!OUTPUT_DIR_UNIX!",>> config.json
    >> config.json echo     "subset_half_size": 10,
    >> config.json echo     "step": 1,
    >> config.json echo     "shape_order": 2,
    >> config.json echo     "search_radius": 20,
    >> config.json echo     "max_iterations": 25,
    >> config.json echo     "cutoff_diffnorm": 1e-3,
    >> config.json echo     "lambda_reg": 1e-3,
    >> config.json echo     "parallel_flag": false,
    >> config.json echo     "max_workers": 1,
    >> config.json echo     "smooth_flage": true,
    >> config.json echo     "smooth_sigma": 1.0,
    >> config.json echo     "strain_calculate_flage": true,
    >> config.json echo     "strain_window_half_size": 31,
    >> config.json echo     "show_plot": true
    >> config.json echo }

    echo config.json updated
    echo.

    :: Run Python solve
    python DIC_main_solve.py

    echo Completed folder: %%D
    echo -----------------------------------------------
    echo.
)

echo All done!
pause
