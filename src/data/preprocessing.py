import os
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional


def resample_to_mni(
    input_nifti: str,
    output_nifti: str,
    mni_template: str = "MNI152_T1_2mm",
    interpolation: str = "linear",
    work_dir: Optional[str] = None,
) -> str:
    """Resample a subject-space fMRI volume to MNI152 2mm space using ANTs.

    Assumes the subject's T1-to-MNI transform is available alongside the data.
    Falls back to FSL FLIRT if ANTs is unavailable.

    Args:
        input_nifti: path to native-space fMRI NIfTI.
        output_nifti: path for resampled NIfTI.
        mni_template: name/path of the MNI template.
        interpolation: 'linear' or 'nearest'.
        work_dir: scratch directory for intermediate files.

    Returns:
        Path to the resampled NIfTI.
    """
    input_path = Path(input_nifti)
    output_path = Path(output_nifti)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_dir = input_path.parent
    t1_to_mni_mat = subject_dir / "t1_to_mni.mat"
    t1_to_mni_warp = subject_dir / "t1_to_mni_warp.nii.gz"
    t1_to_mni_affine = subject_dir / "t1_to_mni_0GenericAffine.mat"
    fmri_to_t1_mat = subject_dir / "fmri_to_t1.mat"

    # Strategy 1: ANTs composite transform (T1 -> MNI warp + fMRI -> T1)
    if t1_to_mni_warp.exists() and t1_to_mni_affine.exists():
        _run_ants_registration(
            input_nifti, output_nifti, t1_to_mni_warp, t1_to_mni_affine, interpolation
        )
    # Strategy 2: FSL FLIRT with precomputed matrix
    elif t1_to_mni_mat.exists():
        _run_flirt_registration(
            input_nifti, output_nifti, t1_to_mni_mat, mni_template, interpolation
        )
    # Strategy 3: Direct FLIRT to MNI template
    else:
        _run_flirt_12dof(input_nifti, output_nifti, mni_template, interpolation)

    return str(output_path)


def _run_ants_registration(
    fixed, moving, warp, affine, interpolation="linear"
):
    """Apply ANTs composite transform."""
    interp_flag = "Linear" if interpolation == "linear" else "NearestNeighbor"
    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", fixed,
        "-r", fixed,
        "-t", str(warp),
        "-t", str(affine),
        "-o", moving,
        "-n", interp_flag,
    ]
    subprocess.run(cmd, check=True)


def _run_flirt_registration(
    input_nifti, output_nifti, mat_file, template, interpolation="linear"
):
    """Apply FSL FLIRT with precomputed affine."""
    interp_flag = "spline" if interpolation == "linear" else "nearestneighbour"
    cmd = [
        "flirt",
        "-in", input_nifti,
        "-ref", template,
        "-init", mat_file,
        "-out", output_nifti,
        "-applyxfm",
        "-interp", interp_flag,
    ]
    subprocess.run(cmd, check=True)


def _run_flirt_12dof(input_nifti, output_nifti, template, interpolation="linear"):
    """Run FSL FLIRT with 12 DOF (no prior alignment)."""
    interp_flag = "spline" if interpolation == "linear" else "nearestneighbour"
    cmd = [
        "flirt",
        "-in", input_nifti,
        "-ref", template,
        "-out", output_nifti,
        "-dof", "12",
        "-interp", interp_flag,
    ]
    subprocess.run(cmd, check=True)


def pad_or_crop_to_96(volume: np.ndarray) -> np.ndarray:
    """Pad or crop a 3D volume to 96x96x96 (SwiFT input size).

    Centers the brain in the target volume.
    """
    target_shape = (96, 96, 96)
    result = np.zeros(target_shape, dtype=volume.dtype)

    offsets = []
    for d in range(3):
        if volume.shape[d] > target_shape[d]:
            start = (volume.shape[d] - target_shape[d]) // 2
            offsets.append((start, start + target_shape[d]))
        else:
            start = (target_shape[d] - volume.shape[d]) // 2
            offsets.append(start)

    if volume.shape[0] > target_shape[0]:
        v = volume[offsets[0][0] : offsets[0][1], :, :]
    else:
        result[offsets[0] : offsets[0] + volume.shape[0], :, :] = volume
        v = result

    if v.shape[1] > target_shape[1]:
        v = v[:, offsets[1][0] : offsets[1][1], :]
    elif v.shape[1] < target_shape[1]:
        pad = np.zeros((v.shape[0], target_shape[1], v.shape[2]), dtype=v.dtype)
        pad[:, offsets[1] : offsets[1] + v.shape[1], :] = v
        v = pad

    if v.shape[2] > target_shape[2]:
        v = v[:, :, offsets[2][0] : offsets[2][1]]
    elif v.shape[2] < target_shape[2]:
        pad = np.zeros((v.shape[0], v.shape[1], target_shape[2]), dtype=v.dtype)
        pad[:, :, offsets[2] : offsets[2] + v.shape[2]] = v
        v = pad

    return v


def zscore_normalize(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Whole-brain z-score normalization (SwiFT default)."""
    if mask is not None:
        brain = volume[mask > 0]
        mean, std = brain.mean(), brain.std()
        volume = volume.copy()
        volume[mask > 0] = (volume[mask > 0] - mean) / (std + 1e-8)
    else:
        mean, std = volume.mean(), volume.std()
        volume = (volume - mean) / (std + 1e-8)
    return volume.astype(np.float32)
