# Model Weights Guide

This guide explains how to obtain, set up, and use the trained nnUNet model weights for microCT inference.

## Important Note

**Model weights are NOT included in this repository** because they are too large for Git. You need to obtain them separately.

## Where to Get Model Weights

Model weights can be obtained from:

1. **From the original training location**: If you have access to the training system, copy the weights from:
   ```
   $nnUNet_results/Dataset001_MicroCT/nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0/
   ```

2. **From a shared location**: If weights are stored on a shared drive or server, copy them to your local machine.

3. **From a colleague**: Ask the person who trained the models to share the weights directory.

4. **Download from cloud storage**: If weights are stored in cloud storage (Google Drive, Dropbox, etc.), download them.

## Required Directory Structure

The model weights must be organized in the following structure:

```
$nnUNet_results/
└── Dataset001_MicroCT/  (or Dataset001_microCT)
    └── nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0/
        ├── fold_0/
        │   └── checkpoint_final.pth    # ~500MB - 2GB per fold
        ├── fold_1/
        │   └── checkpoint_final.pth
        ├── fold_2/
        │   └── checkpoint_final.pth
        ├── fold_3/
        │   └── checkpoint_final.pth
        ├── fold_4/
        │   └── checkpoint_final.pth
        ├── plans.json                   # Required
        ├── dataset.json                 # Required
        └── debug.json                   # Optional
```

## Setting Up Model Weights

### Step 1: Choose a Location

Create a directory to store model weights. This can be anywhere on your system:

```bash
# Example: Create a directory in your home folder
mkdir -p ~/nnUNet_results/Dataset001_MicroCT
```

### Step 2: Copy Model Weights

Copy the entire model directory to your chosen location:

```bash
# Example: Copy from source location
cp -r /path/to/source/nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0 \
      ~/nnUNet_results/Dataset001_MicroCT/
```

Or use the provided script (see below).

### Step 3: Set Environment Variable

Tell nnUNet where to find the weights:

```bash
export nnUNet_results="$HOME/nnUNet_results"
```

Or add to `~/.bashrc` for permanent setup:

```bash
echo 'export nnUNet_results="$HOME/nnUNet_results"' >> ~/.bashrc
source ~/.bashrc
```

## Verifying Model Weights

Use the verification script to check that your weights are set up correctly:

```bash
python3 verify_weights.py --results_dir $nnUNet_results
```

Or let it auto-detect:

```bash
python3 verify_weights.py
```

## Using Model Weights

Once weights are set up, you can use them in two ways:

### Option 1: Automatic Detection (Recommended)

The inference script will automatically find weights if `nnUNet_results` is set:

```bash
export nnUNet_results="$HOME/nnUNet_results"
python3 run_inference.py \
    --input_dir /path/to/data \
    --output_dir /path/to/predictions \
    --results_dir $nnUNet_results
```

### Option 2: Direct Path

Specify the model directory directly:

```bash
python3 run_inference.py \
    --input_dir /path/to/data \
    --output_dir /path/to/predictions \
    --model_dir ~/nnUNet_results/Dataset001_MicroCT/nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0
```

## Copying Weights to Local Directory

You can copy weights to a local directory before inference:

```bash
python3 run_inference.py \
    --input_dir /path/to/data \
    --output_dir /path/to/predictions \
    --results_dir $nnUNet_results \
    --copy_weights ./local_weights
```

This is useful if:
- You want to work offline
- You want to keep a local copy
- You're sharing weights with others

## Sharing Model Weights

If you need to share model weights with others:

### Option 1: Compress and Share

```bash
# Create a compressed archive
cd $nnUNet_results
tar -czf microct_weights.tar.gz Dataset001_MicroCT/

# Or use zip
zip -r microct_weights.zip Dataset001_MicroCT/
```

### Option 2: Use nnUNet Export

nnUNet provides a built-in export function:

```bash
nnUNetv2_export_model_to_zip \
    -m $nnUNet_results/Dataset001_MicroCT/nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0 \
    -o microct_weights.zip
```

### Option 3: Cloud Storage

Upload to:
- Google Drive
- Dropbox
- AWS S3
- Institutional file sharing system

## Storage Requirements

Model weights are large:
- **Per fold**: ~500MB - 2GB
- **5 folds**: ~2.5GB - 10GB total
- **With all files**: ~3GB - 12GB

Make sure you have enough disk space!

## Troubleshooting

### "Model directory not found"

**Solution**: 
1. Check that `nnUNet_results` is set: `echo $nnUNet_results`
2. Verify the directory exists: `ls -la $nnUNet_results/Dataset001_MicroCT/`
3. Check the directory structure matches the expected format

### "Checkpoint not found"

**Solution**:
1. Verify checkpoints exist: `ls -la $nnUNet_results/Dataset001_MicroCT/*/fold_*/checkpoint_final.pth`
2. Check that you have at least one fold with a checkpoint
3. Make sure the checkpoint file is not corrupted

### "Insufficient disk space"

**Solution**:
1. Check available space: `df -h`
2. Clean up unnecessary files
3. Consider using only specific folds instead of all 5

### "Permission denied"

**Solution**:
1. Check file permissions: `ls -la $nnUNet_results/`
2. Fix permissions if needed: `chmod -R 755 $nnUNet_results/`

## Best Practices

1. **Keep a backup**: Model weights take time to train. Keep backups!
2. **Use symbolic links**: If weights are on a network drive, use symlinks:
   ```bash
   ln -s /network/drive/weights ~/nnUNet_results/Dataset001_MicroCT
   ```
3. **Verify after copying**: Always verify weights after copying to ensure integrity
4. **Document location**: Keep track of where your weights are stored
