#!/usr/bin/env python3
"""
Prepare microCT data for inference.
Loads Analyze format (.img + .hdr) and writes NIfTI for nnUNet inference.
This script is for inference only - no training data preparation needed.

Pass --image (path to the Analyze .img file) and --output_dir. Optional --core_mask applies a mask;
if omitted, no masking is performed.
"""

import argparse
import traceback
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


def case_id_from_image_path(image_path: Path) -> str:
    """Stem used for nnUNet output CASE_0000.nii.gz (from Analyze .img basename)."""
    name = image_path.name
    if name.lower().endswith(".img"):
        return name[: -len(".img")]
    return image_path.stem


class DataPreparer:
    def __init__(
        self,
        output_dir: str,
        enable_intensity_clipping: bool = True,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.enable_intensity_clipping = enable_intensity_clipping
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _squeeze_trailing_unit_axes(data: np.ndarray) -> np.ndarray:
        out = data
        while out.ndim > 1 and out.shape[-1] == 1:
            out = out[..., 0]
        return out

    def clip_image_intensity(self, image_data: np.ndarray) -> np.ndarray:
        lower_bound = np.percentile(image_data, self.lower_percentile)
        upper_bound = np.percentile(image_data, self.upper_percentile)
        clipped = np.clip(image_data, lower_bound, upper_bound)
        print(
            f"  Intensity clipping: [{lower_bound:.2f}, {upper_bound:.2f}] "
            f"(percentiles: {self.lower_percentile}%, {self.upper_percentile}%)"
        )
        return clipped

    def load_analyze_image(self, img_path: str) -> nib.Nifti1Image:
        try:
            analyze_img = nib.analyze.load(img_path)
            data = DataPreparer._squeeze_trailing_unit_axes(analyze_img.get_fdata())
            return nib.Nifti1Image(data, analyze_img.affine)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise

    def load_analyze_volume(self, image_path: Path) -> nib.Nifti1Image:
        """Load primary scan; must be an Analyze pair (.img with .hdr alongside)."""
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not image_path.name.lower().endswith(".img"):
            raise ValueError(f"Expected Analyze .img path, got: {image_path}")
        hdr = image_path.with_suffix(".hdr")
        if not hdr.is_file():
            print(f"Warning: No matching .hdr next to image: {hdr}")
        print(f"Loading Analyze volume: {image_path}")
        return self.load_analyze_image(str(image_path))

    @staticmethod
    def _find_paired_analyze_img(mask_hdr: Path) -> Optional[Path]:
        """
        Locate the .img next to an Analyze .hdr: exact .img names first, then any .img in
        the same folder whose stem matches the .hdr stem ignoring case (e.g. .mask. vs .Mask.).
        """
        parent = mask_hdr.parent
        stem = mask_hdr.stem
        candidates = [
            mask_hdr.with_suffix(".img"),
            Path(str(mask_hdr).replace(".hdr", ".img")),
        ]
        for c in candidates:
            if c.is_file():
                return c
        for p in parent.iterdir():
            if not p.is_file() or p.suffix.lower() != ".img":
                continue
            if p.stem.lower() == stem.lower():
                return p
        return None

    def _load_mask_nifti(self, mask_file: Path) -> Optional[nib.Nifti1Image]:
        if not mask_file.is_file():
            print(f"Warning: Mask file not found: {mask_file}")
            return None
        print(f"Applying mask: {mask_file}")
        if mask_file.suffix.lower() == ".hdr" or mask_file.name.lower().endswith(".mask.hdr"):
            mask_img_path = self._find_paired_analyze_img(mask_file)
            if mask_img_path is not None:
                return self.load_analyze_image(str(mask_img_path))
            tried = mask_file.with_suffix(".img")
            print(f"Warning: Mask .img file not found for {mask_file} (tried {tried} and same-stem .img in folder)")
            return None
        if mask_file.name.endswith(".nii.gz") or mask_file.suffix.lower() == ".nii":
            return nib.load(str(mask_file))
        print(f"Warning: Unknown mask format: {mask_file.name}")
        return None

    def process_image(
        self,
        image_path: str,
        core_mask: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> bool:
        """
        Load image from disk, optional core mask, clip, save nnUNet-style CASE_0000.nii.gz.

        Args:
            image_path: Path to Analyze .img (paired .hdr in the same directory)
            core_mask: Optional path to mask (.hdr/.img pair or NIfTI). Relative paths are
                resolved against the image file's directory.
            case_id: Output case name; default is derived from the image filename.
        """
        try:
            img_path = Path(image_path).expanduser()
            cid = case_id.strip() if case_id else case_id_from_image_path(img_path)

            img_nifti = self.load_analyze_volume(img_path)
            img_data = DataPreparer._squeeze_trailing_unit_axes(np.asarray(img_nifti.get_fdata()))

            if self.enable_intensity_clipping:
                print(f"Applying intensity clipping for case {cid}...")
                clipped_data = self.clip_image_intensity(img_data)
            else:
                clipped_data = img_data

            if core_mask:
                mask_file = Path(core_mask).expanduser()
                if not mask_file.is_absolute():
                    mask_file = (img_path.parent / mask_file).resolve()
                mask_nifti = self._load_mask_nifti(mask_file)
                if mask_nifti is not None:
                    mask_data = DataPreparer._squeeze_trailing_unit_axes(np.asarray(mask_nifti.get_fdata()))
                    if mask_data.shape == clipped_data.shape:
                        mask_binary = (mask_data > 0).astype(bool)
                        clipped_data = clipped_data.copy()
                        clipped_data[~mask_binary] = (
                            np.min(clipped_data[mask_binary]) if np.any(mask_binary) else 0
                        )
                        print(f"  Mask applied: {np.sum(mask_binary):,} voxels inside mask")
                    else:
                        print(
                            f"Warning: Mask shape {mask_data.shape} does not match "
                            f"image shape {clipped_data.shape}; skipping mask"
                        )

            out = nib.Nifti1Image(clipped_data, img_nifti.affine)
            output_path = self.output_dir / f"{cid}_0000.nii.gz"
            nib.save(out, str(output_path))
            print(f"Saved image: {output_path}")
            return True
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Prepare microCT data for inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input Analyze volume (.img; .hdr must sit beside it)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for nnUNet-style output (CASE_0000.nii.gz)",
    )
    parser.add_argument(
        "--core_mask",
        type=str,
        default=None,
        help="Optional core mask (.hdr with paired .img, or NIfTI). Relative paths are "
        "resolved next to --image.",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="Case id for output filename (default: derived from --image basename)",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="If set, print a sample nnUNetv2_predict command using this output folder.",
    )
    parser.add_argument(
        "--enable_intensity_clipping",
        action="store_true",
        default=True,
        help="Enable intensity clipping",
    )
    parser.add_argument(
        "--no_intensity_clipping",
        action="store_true",
        help="Disable intensity clipping",
    )
    parser.add_argument("--lower_percentile", type=float, default=1.0)
    parser.add_argument("--upper_percentile", type=float, default=99.0)

    args = parser.parse_args()

    preparer = DataPreparer(
        output_dir=args.output_dir,
        enable_intensity_clipping=args.enable_intensity_clipping and not args.no_intensity_clipping,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )

    ok = preparer.process_image(
        args.image,
        core_mask=args.core_mask,
        case_id=args.case_id,
    )

    if not ok:
        raise SystemExit(1)

    print("\nNext step — run inference (set nnUNet_results / nnUNet_raw env vars as needed):")
    out_in = str(Path(args.output_dir).resolve())
    if args.predictions_dir:
        pred = str(Path(args.predictions_dir).resolve())
        print(f"   nnUNetv2_predict -i {out_in} -o {pred} -d DATASET_ID -c 3d_fullres -f 0")
    else:
        print(f"   nnUNetv2_predict -i {out_in} -o PREDICTIONS_DIR -d DATASET_ID -c 3d_fullres -f 0")
    print("   Use your dataset id for -d, list folds after -f for ensembling, and a real predictions path for -o.")


if __name__ == "__main__":
    main()
