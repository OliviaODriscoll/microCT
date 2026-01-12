#!/usr/bin/env python3
"""
Prepare microCT data for inference.
Converts Analyze format (.img/.hdr) to NIfTI format and prepares data for nnUNet inference.
This script is for inference only - no training data preparation needed.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional


class DataPreparer:
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 enable_intensity_clipping: bool = True,
                 lower_percentile: float = 1.0,
                 upper_percentile: float = 99.0):
        """
        Initialize the data preparer.
        
        Args:
            input_dir: Path to raw data directory
            output_dir: Path to output directory for inference-ready data
            enable_intensity_clipping: Apply intensity clipping to remove streak artifacts
            lower_percentile: Lower percentile for clipping (default: 1.0)
            upper_percentile: Upper percentile for clipping (default: 99.0)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.enable_intensity_clipping = enable_intensity_clipping
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        # Case mapping (directory name to case ID)
        # Update this mapping to match your data structure
        self.case_mapping = {
            "04 - 5011": "5011",
            "07 - 5023": "5023", 
            "10 - 4327": "4327",
            "11 - 4331": "4331"
        }
        
        # File patterns for each case
        # Update these patterns to match your file naming convention
        self.file_patterns = {
            "5011": {"img": "5011.img", "nii": "5011.nii.gz"},
            "5023": {"img": "5023.img"},
            "4327": {"img": "4327.img"},
            "4331": {"img": "4331.res2x.img"}
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clip_image_intensity(self, image_data: np.ndarray) -> np.ndarray:
        """
        Clip image intensity to remove streak artifacts.
        
        Args:
            image_data: Input image data as numpy array
            
        Returns:
            Clipped image data
        """
        lower_bound = np.percentile(image_data, self.lower_percentile)
        upper_bound = np.percentile(image_data, self.upper_percentile)
        clipped_data = np.clip(image_data, lower_bound, upper_bound)
        
        print(f"  Intensity clipping: [{lower_bound:.2f}, {upper_bound:.2f}] "
              f"(percentiles: {self.lower_percentile}%, {self.upper_percentile}%)")
        
        return clipped_data
    
    def load_analyze_image(self, img_path: str) -> nib.Nifti1Image:
        """
        Load Analyze format image and convert to NIfTI.
        
        Args:
            img_path: Path to .img file
            
        Returns:
            NIfTI image object
        """
        try:
            analyze_img = nib.analyze.load(img_path)
            data = analyze_img.get_fdata()
            affine = analyze_img.affine
            
            # Remove singleton dimension if present
            if data.ndim == 4 and data.shape[3] == 1:
                data = data.squeeze(axis=3)
            
            nifti_img = nib.Nifti1Image(data, affine)
            return nifti_img
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise
    
    def process_case(self, case_id: str) -> bool:
        """
        Process a single case for inference.
        
        Args:
            case_id: Case identifier (e.g., "5011")
            
        Returns:
            Success status
        """
        try:
            # Find the source directory
            src_dir = None
            for dir_name, mapped_id in self.case_mapping.items():
                if mapped_id == case_id:
                    src_dir = self.input_dir / dir_name
                    break
            
            if src_dir is None or not src_dir.exists():
                print(f"Source directory not found for case {case_id}")
                return False
            
            patterns = self.file_patterns[case_id]
            
            # Load image
            if "nii" in patterns and (src_dir / patterns["nii"]).exists():
                # Use existing NIfTI file
                img_nifti = nib.load(src_dir / patterns["nii"])
                print(f"Loaded existing NIfTI for case {case_id}")
            else:
                # Convert from Analyze format
                img_path = src_dir / patterns["img"]
                
                if not img_path.exists():
                    print(f"Image file not found: {img_path}")
                    return False
                
                img_nifti = self.load_analyze_image(str(img_path))
                print(f"Converted Analyze to NIfTI for case {case_id}")
            
            # Apply intensity clipping
            img_data = img_nifti.get_fdata()
            if self.enable_intensity_clipping:
                print(f"Applying intensity clipping for case {case_id}...")
                clipped_data = self.clip_image_intensity(img_data)
            else:
                clipped_data = img_data
            
            # Create new NIfTI image with clipped data
            img_nifti_clipped = nib.Nifti1Image(clipped_data, img_nifti.affine)
            
            # Save image in nnUNet format (with _0000 suffix for channel)
            output_path = self.output_dir / f"{case_id}_0000.nii.gz"
            nib.save(img_nifti_clipped, str(output_path))
            print(f"Saved image: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            return False
    
    def process_all_cases(self, case_ids: Optional[list] = None):
        """
        Process all cases or specified cases.
        
        Args:
            case_ids: List of case IDs to process. If None, processes all available cases.
        """
        if case_ids is None:
            case_ids = list(self.case_mapping.values())
        
        print(f"Processing {len(case_ids)} cases for inference...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        success_count = 0
        for case_id in case_ids:
            print(f"\nProcessing case: {case_id}")
            if self.process_case(case_id):
                success_count += 1
            else:
                print(f"Failed to process case: {case_id}")
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Successfully processed: {success_count}/{len(case_ids)} cases")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Prepare microCT data for inference')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with raw data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for inference-ready data')
    parser.add_argument('--case_ids', nargs='+',
                       default=None,
                       help='Specific case IDs to process (e.g., 5011 5023). If not specified, processes all.')
    parser.add_argument('--enable_intensity_clipping', action='store_true', default=True,
                       help='Enable intensity clipping to remove streak artifacts')
    parser.add_argument('--no_intensity_clipping', action='store_true',
                       help='Disable intensity clipping')
    parser.add_argument('--lower_percentile', type=float, default=1.0,
                       help='Lower percentile for intensity clipping (default: 1.0)')
    parser.add_argument('--upper_percentile', type=float, default=99.0,
                       help='Upper percentile for intensity clipping (default: 99.0)')
    
    args = parser.parse_args()
    
    # Create preparer
    preparer = DataPreparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        enable_intensity_clipping=args.enable_intensity_clipping and not args.no_intensity_clipping,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile
    )
    
    # Process cases
    preparer.process_all_cases(case_ids=args.case_ids)
    
    print("\nNext steps:")
    print("1. Run inference using:")
    print("   python3 run_inference.py --input_dir", args.output_dir)
    print("   or")
    print("   nnUNetv2_predict -i", args.output_dir, "-o /path/to/predictions -d 1 -c 3d_fullres -f 0")


if __name__ == "__main__":
    main()
