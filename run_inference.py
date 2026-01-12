#!/usr/bin/env python3
"""
Run inference on microCT data using trained nnUNet models.
This script loads a trained model and runs inference on prepared data.
Supports ensembling multiple folds for better predictions.
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import subdirs, join, isfile


def find_model_weights(results_dir: str, dataset_id: int = 1, 
                      configuration: str = "3d_fullres", fold: Optional[int] = None):
    """
    Find the trained model weights directory and available folds.
    
    Args:
        results_dir: Path to nnUNet_results directory
        dataset_id: Dataset ID (default: 1 for Dataset001_MicroCT)
        configuration: Model configuration (default: 3d_fullres)
        fold: Specific fold number (None to find all folds)
    
    Returns:
        Tuple of (model directory path, list of available fold numbers)
    """
    results_path = Path(results_dir)
    
    # Try different dataset name formats
    dataset_names = [
        f"Dataset{dataset_id:03d}_MicroCT",
        f"Dataset{dataset_id:03d}_microCT"
    ]
    
    # First try the new structure (folds as subdirectories)
    for dataset_name in dataset_names:
        model_dir = results_path / dataset_name / f"nnUNetTrainer__nnUNetPlans__{configuration}"
        if model_dir.exists():
            # Find all available folds
            fold_folders = subdirs(str(model_dir), prefix='fold_', join=False)
            fold_folders = [f for f in fold_folders if f != 'fold_all']
            available_folds = []
            
            for fold_folder in fold_folders:
                checkpoint_file = model_dir / fold_folder / "checkpoint_final.pth"
                if checkpoint_file.exists():
                    fold_num = int(fold_folder.split('_')[-1])
                    available_folds.append(fold_num)
            
            if available_folds:
                # If specific fold requested, check if it exists
                if fold is not None:
                    if fold in available_folds:
                        return str(model_dir), [fold]
                    else:
                        print(f"Warning: Fold {fold} not found. Available folds: {available_folds}")
                        return str(model_dir), available_folds
                else:
                    # Return all available folds
                    return str(model_dir), sorted(available_folds)
    
    # Fallback to old structure (fold in directory name)
    for dataset_name in dataset_names:
        model_dir = results_path / dataset_name / f"nnUNetTrainer__nnUNetPlans__{configuration}__fold_0"
        if model_dir.exists():
            # Find all available folds
            fold_folders = subdirs(str(model_dir), prefix='fold_', join=False)
            fold_folders = [f for f in fold_folders if f != 'fold_all']
            available_folds = []
            
            for fold_folder in fold_folders:
                checkpoint_file = model_dir / fold_folder / "checkpoint_final.pth"
                if checkpoint_file.exists():
                    fold_num = int(fold_folder.split('_')[-1])
                    available_folds.append(fold_num)
            
            if available_folds:
                # If specific fold requested, check if it exists
                if fold is not None:
                    if fold in available_folds:
                        return str(model_dir), [fold]
                    else:
                        print(f"Warning: Fold {fold} not found. Available folds: {available_folds}")
                        return str(model_dir), available_folds
                else:
                    # Return all available folds
                    return str(model_dir), sorted(available_folds)
    
    return None, []


def copy_model_weights(model_dir: str, output_dir: str, folds: Optional[List[int]] = None):
    """
    Copy model weights to a specified directory.
    
    Args:
        model_dir: Source model directory
        output_dir: Destination directory for copied weights
        folds: List of fold numbers to copy (None = copy all available folds)
    """
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying model weights from: {model_dir}")
    print(f"Destination: {output_dir}")
    
    # Find available folds
    fold_folders = subdirs(str(model_path), prefix='fold_', join=False)
    fold_folders = [f for f in fold_folders if f != 'fold_all']
    
    if folds is None:
        # Copy all available folds
        folds_to_copy = []
        for fold_folder in fold_folders:
            checkpoint_file = model_path / fold_folder / "checkpoint_final.pth"
            if checkpoint_file.exists():
                fold_num = int(fold_folder.split('_')[-1])
                folds_to_copy.append(fold_num)
    else:
        folds_to_copy = folds
    
    # Copy dataset.json and plans.json
    for json_file in ['dataset.json', 'plans.json']:
        src_file = model_path / json_file
        if src_file.exists():
            dst_file = output_path / json_file
            shutil.copy2(src_file, dst_file)
            print(f"  Copied: {json_file}")
    
    # Copy each fold
    for fold_num in sorted(folds_to_copy):
        fold_folder = f"fold_{fold_num}"
        src_fold_dir = model_path / fold_folder
        dst_fold_dir = output_path / fold_folder
        
        if src_fold_dir.exists():
            dst_fold_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy checkpoint file
            checkpoint_file = src_fold_dir / "checkpoint_final.pth"
            if checkpoint_file.exists():
                shutil.copy2(checkpoint_file, dst_fold_dir / "checkpoint_final.pth")
                print(f"  Copied: {fold_folder}/checkpoint_final.pth")
            
            # Copy any other files in the fold directory
            for file in src_fold_dir.iterdir():
                if file.is_file() and file.name != "checkpoint_final.pth":
                    shutil.copy2(file, dst_fold_dir / file.name)
                    print(f"  Copied: {fold_folder}/{file.name}")
    
    print(f"\nModel weights copied successfully to: {output_dir}")
    print(f"Copied {len(folds_to_copy)} fold(s): {sorted(folds_to_copy)}")


def run_inference(input_dir: str,
                 output_dir: str,
                 model_dir: str,
                 folds: Optional[List[int]] = None,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 device: str = "cuda"):
    """
    Run inference on prepared data. Automatically ensembles all available folds if folds=None.
    
    Args:
        input_dir: Directory with input images (NIfTI format)
        output_dir: Directory to save predictions
        model_dir: Path to trained model directory
        folds: List of fold numbers to use (None = use all available folds for ensembling)
        use_gaussian: Use Gaussian weighting for predictions (default: True)
        use_mirroring: Use test-time augmentation with mirroring (default: True)
        device: Device to use ('cuda' or 'cpu')
    """
    print("="*60)
    print("MicroCT Inference")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Auto-detect folds if not specified
    if folds is None:
        print("Auto-detecting available folds for ensembling...")
        fold_folders = subdirs(model_dir, prefix='fold_', join=False)
        fold_folders = [f for f in fold_folders if f != 'fold_all']
        folds = []
        for fold_folder in fold_folders:
            checkpoint_file = Path(model_dir) / fold_folder / "checkpoint_final.pth"
            if checkpoint_file.exists():
                fold_num = int(fold_folder.split('_')[-1])
                folds.append(fold_num)
        folds = sorted(folds)
        print(f"Found {len(folds)} fold(s): {folds}")
    
    if not folds:
        raise FileNotFoundError(f"No valid folds found in {model_dir}")
    
    if len(folds) > 1:
        print(f"Ensembling {len(folds)} folds: {folds}")
    else:
        print(f"Using fold: {folds[0]}")
    
    print(f"Device: {device}")
    print("="*60)
    
    # Initialize predictor
    print("\nInitializing nnUNet predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    
    # Initialize from trained model (will ensemble if multiple folds provided)
    print(f"Loading model from: {model_dir}")
    if len(folds) == 1:
        use_folds = (folds[0],)
    else:
        use_folds = tuple(folds)
    
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=use_folds,
        checkpoint_name='checkpoint_final.pth'
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"\nRunning inference on images in: {input_dir}")
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_dir,
        output_folder_or_list_of_truncated_output_files=str(output_path),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    
    print("\n" + "="*60)
    print("Inference completed successfully!")
    if len(folds) > 1:
        print(f"Ensembled predictions from {len(folds)} folds saved to: {output_dir}")
    else:
        print(f"Predictions saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run inference on microCT data with automatic ensembling')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with input images (NIfTI format, prepared by prepare_data_for_inference.py)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save predictions')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Path to trained model directory. If not specified, will search in nnUNet_results.')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Path to nnUNet_results directory (if model_dir not specified)')
    parser.add_argument('--dataset_id', type=int, default=1,
                       help='Dataset ID (default: 1)')
    parser.add_argument('--configuration', type=str, default='3d_fullres',
                       choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'],
                       help='Model configuration (default: 3d_fullres)')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold number to use (default: None = ensemble all available folds)')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                       help='List of specific folds to ensemble (e.g., --folds 0 1 2 3 4). Overrides --fold.')
    parser.add_argument('--copy_weights', type=str, default=None,
                       help='Copy model weights to specified directory before inference')
    parser.add_argument('--no_gaussian', action='store_true',
                       help='Disable Gaussian weighting')
    parser.add_argument('--no_mirroring', action='store_true',
                       help='Disable test-time augmentation with mirroring')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Find model directory if not specified
    if args.model_dir is None:
        if args.results_dir is None:
            # Try to get from environment variable
            results_dir = os.environ.get('nnUNet_results')
            if results_dir is None:
                raise ValueError(
                    "Either --model_dir or --results_dir must be specified, "
                    "or nnUNet_results environment variable must be set."
                )
        else:
            results_dir = args.results_dir
        
        print(f"Searching for model in: {results_dir}")
        model_dir, available_folds = find_model_weights(
            results_dir=results_dir,
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            fold=args.fold
        )
        
        if model_dir is None:
            raise FileNotFoundError(
                f"Could not find trained model. "
                f"Expected location: {results_dir}/Dataset{args.dataset_id:03d}_*/"
                f"nnUNetTrainer__nnUNetPlans__{args.configuration}/\n"
                f"Make sure:\n"
                f"  1. nnUNet_results environment variable is set: export nnUNet_results=\"/path/to/nnUNet_results\"\n"
                f"  2. Model weights are installed from model_weights.zip\n"
                f"  3. Or specify --results_dir or --model_dir directly"
            )
        
        print(f"Found model directory: {model_dir}")
        print(f"Available folds: {available_folds}")
    else:
        model_dir = args.model_dir
        # Find available folds in the specified directory
        fold_folders = subdirs(model_dir, prefix='fold_', join=False)
        fold_folders = [f for f in fold_folders if f != 'fold_all']
        available_folds = []
        for fold_folder in fold_folders:
            checkpoint_file = Path(model_dir) / fold_folder / "checkpoint_final.pth"
            if checkpoint_file.exists():
                fold_num = int(fold_folder.split('_')[-1])
                available_folds.append(fold_num)
        available_folds = sorted(available_folds)
    
    # Determine which folds to use
    if args.folds is not None:
        folds_to_use = args.folds
    elif args.fold is not None:
        folds_to_use = [args.fold]
    else:
        # Use all available folds for ensembling
        folds_to_use = None
    
    # Copy weights if requested
    if args.copy_weights:
        copy_model_weights(model_dir, args.copy_weights, folds=folds_to_use)
    
    # Run inference
    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=model_dir,
        folds=folds_to_use,
        use_gaussian=not args.no_gaussian,
        use_mirroring=not args.no_mirroring,
        device=args.device
    )


if __name__ == "__main__":
    main()
