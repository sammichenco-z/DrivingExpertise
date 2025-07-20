#!/bin/bash

# Define command-line options
show_help() {
    echo "Usage: $0 [-t|--type res|full] [-h|--help]"
    echo "  -t, --type    Specify model type (res: residual model, full: full model)"
    echo "  -h, --help    Show this help message"
    exit 1
}

# Default options
model_type="full"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--type)
            model_type="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown parameter: $1" >&2
            show_help
            ;;
    esac
done

# Validate model type
if [[ "$model_type" != "res" && "$model_type" != "full" ]]; then
    echo "Error: Model type must be 'res' or 'full'" >&2
    show_help
fi

# Determine Python script based on selected model type
if [[ "$model_type" == "res" ]]; then
    python_script="fit_res_model.py"
else
    python_script="fit_full_model.py"
fi

# Define data parameters
channels=('AF3' 'AF4' 'CP1' 'CP2' 'CPZ' 'F3' 'F4' 'F5' 'F6' 'F7' 'F8' \
          'FC1' 'FC2' 'FP1' 'FP2' 'FPZ' 'FT7' 'FT8' 'O1' 'O2' 'OZ' \
          'PO3' 'PO4' 'PO5' 'PO6' 'PO7' 'PO8' 'POZ' 'T7' 'T8')

centers=('P1' 'N1' 'P150' 'N170' 'P200' 'P250' 'N300' 'P3a' 'P3b')

models=('full' 'occlusion_reduced' 'hazard_occlusion_reduced')

# Execute nested loops
for model in "${models[@]}"; do
    for channel in "${channels[@]}"; do
        for center in "${centers[@]}"; do
            python "$python_script" --type "$model" --channel "$channel" --center "$center"
        done
    done
done