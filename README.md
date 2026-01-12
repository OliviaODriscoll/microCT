# MicroCT Segmentation with nnUNet

This repository contains scripts for running inference on microCT data using trained nnUNet models.

## Overview

This repository is set up for **inference only** - it does not include training functionality. It provides tools to:
1. Prepare microCT data for inference (convert Analyze format to NIfTI)
2. Run inference using pre-trained nnUNet models with **automatic ensembling of all folds**
3. Copy model weights to a specified directory
4. Generate segmentation predictions

## Key Features

- **Automatic Ensembling**: By default, the inference script automatically detects and ensembles all available folds (typically 5 folds from cross-validation) for better predictions
- **Model Weight Copying**: Option to copy model weights to a local directory before inference
- **Flexible Fold Selection**: Use all folds, specific folds, or a single fold
- **Easy Data Preparation**: Converts Analyze format to nnUNet-compatible NIfTI format

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd microCT
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

4. **Set environment variables** (see [Installation Guide](INSTALL.md) for details):
   ```bash
   export nnUNet_results="/path/to/your/nnUNet_results"
   ```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Prerequisites

- **Python 3.10 or higher**
- **CUDA-capable GPU** (recommended, but CPU inference is supported)
- **Trained nnUNet model weights** (see [Where Are the Weights Stored?](#where-are-the-weights-stored))
- **Your microCT data** in one of these formats:
  - Analyze format (`.img` and `.hdr` files)
  - NIfTI format (`.nii` or `.nii.gz` files)
   - nnunetv2

3. **Trained model weights**: You need access to trained nnUNet model weights

## Model Weights

**⚠️ Important**: Model weights are **NOT included** in this repository. You need to obtain them separately.

See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for detailed instructions on:
- How to obtain model weights
- How to set them up
- How to verify they're working
- How to share weights with others

### Quick Setup

1. **Get model weights** from the training location or shared storage
2. **Copy to your system**:
   ```bash
   mkdir -p ~/nnUNet_results/Dataset001_MicroCT
   cp -r /path/to/weights/nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0 \
         ~/nnUNet_results/Dataset001_MicroCT/
   ```
3. **Set environment variable**:
   ```bash
   export nnUNet_results="$HOME/nnUNet_results"
   ```
4. **Verify setup**:
   ```bash
   python3 verify_weights.py
   ```

## Where Are the Weights Stored?

The trained model weights are stored in the `nnUNet_results` directory. The location depends on your environment setup:

### Default Location Structure

```
$nnUNet_results/
└── Dataset001_MicroCT/  (or Dataset001_microCT)
    └── nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0/
        ├── fold_0/
        │   └── checkpoint_final.pth    # Final model checkpoint for fold 0
        ├── fold_1/
        │   └── checkpoint_final.pth    # Final model checkpoint for fold 1
        ├── fold_2/
        │   └── checkpoint_final.pth    # Final model checkpoint for fold 2
        ├── fold_3/
        │   └── checkpoint_final.pth    # Final model checkpoint for fold 3
        ├── fold_4/
        │   └── checkpoint_final.pth    # Final model checkpoint for fold 4
        ├── plans.json                   # Training plan
        ├── dataset.json                 # Dataset configuration
        └── debug.json                   # Debug information
```

**Note**: The script automatically detects all available folds and ensembles them for better predictions. You typically have 5 folds (0-4) from 5-fold cross-validation.

### Finding Your Weights

1. **Check environment variable**:
   ```bash
   echo $nnUNet_results
   ```

2. **Search for model directory**:
   ```bash
   find $nnUNet_results -name "*MicroCT*" -type d
   find $nnUNet_results -name "*checkpoint_final.pth" -type f
   ```


### Model Directory Naming

The model directory follows this pattern:
```
nnUNetTrainer__nnUNetPlans__{configuration}__fold_{fold_number}
```

Examples:
- `nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0`
- `nnUNetTrainer__nnUNetPlans__2d__fold_1`
- `nnUNetTrainer__nnUNetPlans__3d_cascade_fullres__fold_2`

## Quick Start

### Step 1: Prepare Data for Inference

Convert Analyze format data to NIfTI format:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Prepare data
python3 prepare_data_for_inference.py \
    --input_dir /path/to/your/raw_data \
    --output_dir /path/to/inference_data
```

**Options**:
- `--input_dir`: Path to raw data directory
- `--output_dir`: Path to save inference-ready NIfTI files
- `--case_ids`: Specific cases to process (e.g., `--case_ids 5011 5023`)
- `--enable_intensity_clipping`: Apply intensity clipping (default: True)
- `--lower_percentile`: Lower percentile for clipping (default: 1.0)
- `--upper_percentile`: Upper percentile for clipping (default: 99.0)

### Step 2: Run Inference

Run inference using trained model weights. **By default, the script will automatically ensemble all available folds for better predictions.**

```bash
python3 run_inference.py \
    --input_dir /DATA/summer_students/process_OO/microCT/inference_data \
    --output_dir /DATA/summer_students/process_OO/microCT/predictions \
    --model_dir /path/to/trained/model
```

**Or let the script find the model automatically and ensemble all folds**:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Set environment variable (if not already set)
export nnUNet_results="/path/to/nnUNet_results"

# Run inference with automatic ensembling
python3 run_inference.py \
    --input_dir /path/to/inference_data \
    --output_dir /path/to/predictions \
    --results_dir $nnUNet_results \
    --dataset_id 1 \
    --configuration 3d_fullres
```

**To use a specific fold instead of ensembling**:

```bash
python3 run_inference.py \
    --input_dir /DATA/summer_students/process_OO/microCT/inference_data \
    --output_dir /DATA/summer_students/process_OO/microCT/predictions \
    --results_dir $nnUNet_results \
    --fold 0
```

**To ensemble specific folds**:

```bash
python3 run_inference.py \
    --input_dir /DATA/summer_students/process_OO/microCT/inference_data \
    --output_dir /DATA/summer_students/process_OO/microCT/predictions \
    --results_dir $nnUNet_results \
    --folds 0 1 2 3 4
```

**To copy model weights to a directory before inference**:

```bash
python3 run_inference.py \
    --input_dir /DATA/summer_students/process_OO/microCT/inference_data \
    --output_dir /DATA/summer_students/process_OO/microCT/predictions \
    --results_dir $nnUNet_results \
    --copy_weights /path/to/copied/weights
```

**Options**:
- `--input_dir`: Directory with prepared NIfTI images (required)
- `--output_dir`: Directory to save predictions (required)
- `--model_dir`: Direct path to model directory (optional)
- `--results_dir`: Path to nnUNet_results directory (if model_dir not specified)
- `--dataset_id`: Dataset ID (default: 1)
- `--configuration`: Model configuration: `2d`, `3d_fullres`, `3d_lowres`, `3d_cascade_fullres` (default: `3d_fullres`)
- `--fold`: Specific fold number to use (default: None = ensemble all available folds)
- `--folds`: List of specific folds to ensemble (e.g., `--folds 0 1 2 3 4`). Overrides `--fold`.
- `--copy_weights`: Copy model weights to specified directory before inference
- `--device`: Device to use: `cuda` or `cpu` (default: `cuda`)
- `--no_gaussian`: Disable Gaussian weighting
- `--no_mirroring`: Disable test-time augmentation

## Complete Workflow Example

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Set environment variable
export nnUNet_results="/path/to/your/nnUNet_results"

# 3. Prepare data for inference
python3 prepare_data_for_inference.py \
    --input_dir /path/to/your/raw_data \
    --output_dir /path/to/inference_data \
    --case_ids 5011 5023 4327 4331

# 4. Run inference with automatic ensembling (uses all available folds)
python3 run_inference.py \
    --input_dir /path/to/inference_data \
    --output_dir /path/to/predictions \
    --results_dir $nnUNet_results \
    --dataset_id 1 \
    --configuration 3d_fullres

# 5. (Optional) Copy model weights to a local directory
python3 run_inference.py \
    --input_dir /path/to/inference_data \
    --output_dir /path/to/predictions \
    --results_dir $nnUNet_results \
    --copy_weights /path/to/copied_weights

# 6. Check predictions
ls -lh /path/to/predictions/
```

**Note**: The script automatically detects and ensembles all available folds (typically 0-4 for 5-fold cross-validation) unless you specify `--fold` or `--folds`. Ensembling multiple folds generally produces better predictions than using a single fold.

## Alternative: Using nnUNet Command Line

You can also use nnUNet's command-line interface directly:

```bash
# Prepare data first (using our script)
python3 prepare_data_for_inference.py \
    --input_dir /DATA/summer_students/process_OO/microCT/raw_data \
    --output_dir /DATA/summer_students/process_OO/microCT/inference_data

# Run inference with nnUNet CLI
nnUNetv2_predict \
    -i /DATA/summer_students/process_OO/microCT/inference_data \
    -o /DATA/summer_students/process_OO/microCT/predictions \
    -d 1 \
    -c 3d_fullres \
    -f 0
```

## Data Structure

The scripts expect data in the following structure:

```
raw_data/
├── 04 - 5011/
│   ├── 5011.img
│   └── 5011.hdr
├── 07 - 5023/
│   ├── 5023.img
│   └── 5023.hdr
├── 10 - 4327/
│   ├── 4327.img
│   └── 4327.hdr
└── 11 - 4331/
    ├── 4331.res2x.img
    └── 4331.res2x.hdr
```

## Verifying Your Setup

Before running inference, verify that everything is set up correctly:

```bash
# Verify model weights
python3 verify_weights.py

# Check that scripts work
python3 prepare_data_for_inference.py --help
python3 run_inference.py --help
```

## Troubleshooting

### Issue: "Model directory not found" or "Model weights not found"

**Solution**: 
1. Check that `nnUNet_results` environment variable is set:
   ```bash
   echo $nnUNet_results
   ```

2. Manually specify model directory:
   ```bash
   python3 run_inference.py \
       --input_dir ... \
       --output_dir ... \
       --model_dir /full/path/to/model/directory
   ```

3. Verify model exists:
   ```bash
   ls -la /path/to/model/directory/fold_0/checkpoint_final.pth
   ```

### Issue: "CUDA out of memory"

**Solutions**:
1. Use CPU instead:
   ```bash
   --device cpu
   ```

2. Use a smaller configuration (if available):
   ```bash
   --configuration 2d
   ```

3. Process fewer cases at a time

### Issue: "Input file not found"

**Solution**: Make sure you've run the data preparation script first:
```bash
python3 prepare_data_for_inference.py --input_dir ... --output_dir ...
```

### Issue: "Checkpoint not found"

**Solution**: The checkpoint might be in a different location. Check:
- `model_dir/fold_0/checkpoint_final.pth`
- `model_dir/checkpoint_final.pth`

## Output Format

Predictions are saved as NIfTI files (`.nii.gz`) with the same naming convention as input files:
- Input: `5011_0000.nii.gz` → Output: `5011.nii.gz`

The segmentation labels follow:
- `0` = background
- `1` = tissue/foreground

## Additional Resources

- [nnUNet Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [nnUNet Inference Guide](documentation/dataset_format_inference.md)
- [nnUNet Usage Instructions](documentation/how_to_use_nnunet.md)

## Support

For issues with:
- **Data preparation**: Check `prepare_data_for_inference.py`
- **Inference**: Check `run_inference.py`
- **Model weights**: Verify `nnUNet_results` environment variable and model directory structure
