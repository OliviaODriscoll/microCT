# Installation Guide

This guide will help you set up the microCT inference pipeline on your computer.

## Prerequisites

- **Python 3.10 or higher** (check with `python3 --version`)
- **Git** (to clone the repository)
- **CUDA-capable GPU** (recommended, but CPU inference is also supported)
- **At least 8GB RAM** (16GB+ recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd microCT
```

### 2. Run the Setup Script

The easiest way to set up the environment is using the provided setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install nnUNet and all dependencies
- Set up the environment

### 3. Manual Installation (Alternative)

If you prefer to install manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install nnUNet
pip install -e .

# Install additional requirements
pip install -r requirements.txt
```

### 4. Verify Installation

Check that nnUNet is installed correctly:

```bash
source venv/bin/activate  # Activate virtual environment if not already active
nnUNetv2_predict --help
```

You should see the nnUNet help message.

## Setting Up Environment Variables

nnUNet requires environment variables to know where to find model weights and data.

### Option 1: Temporary (Current Session Only)

```bash
export nnUNet_results="/path/to/your/nnUNet_results"
```

### Option 2: Permanent (Recommended)

Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
echo 'export nnUNet_results="/path/to/your/nnUNet_results"' >> ~/.bashrc
source ~/.bashrc
```

### Option 3: Create a Setup Script

Create a file `setup_env.sh`:

```bash
#!/bin/bash
export nnUNet_results="/path/to/your/nnUNet_results"
```

Then source it before running inference:

```bash
source setup_env.sh
```

## Getting Model Weights

**⚠️ Important**: Model weights are **NOT included** in this repository. You need to obtain them separately.

See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for complete instructions.

### Quick Summary

Model weights should be organized as:

```
$nnUNet_results/
└── Dataset001_MicroCT/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres__fold_0/
        ├── fold_0/
        │   └── checkpoint_final.pth
        ├── fold_1/
        │   └── checkpoint_final.pth
        └── ...
```

**Note**: You need to obtain the trained model weights separately. They are not included in this repository.

After setting up weights, verify them:
```bash
python3 verify_weights.py
```

## Testing the Installation

Once installed, you can test with a simple command:

```bash
# Activate virtual environment
source venv/bin/activate

# Check that scripts are accessible
python3 prepare_data_for_inference.py --help
python3 run_inference.py --help
```

## Troubleshooting

### Issue: "Command not found: nnUNetv2_predict"

**Solution**: Make sure you've activated the virtual environment and installed nnUNet:
```bash
source venv/bin/activate
pip install -e .
```

### Issue: "CUDA not available"

**Solution**: 
- Check that PyTorch can see your GPU: `python3 -c "import torch; print(torch.cuda.is_available())"`
- If False, you may need to install CUDA-enabled PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- You can still run inference on CPU by using `--device cpu`

### Issue: "Module not found"

**Solution**: Install missing dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Permission denied" when running setup.sh

**Solution**: Make the script executable:
```bash
chmod +x setup.sh
```

## Next Steps

After installation, see the main [README.md](README.md) for usage instructions.
