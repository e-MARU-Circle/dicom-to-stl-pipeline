# Attribution / 帰属表示

This project builds on external work. Please credit the following when you use it.
本プロジェクトは外部成果物の上に成り立っています。利用時は以下を明記してください。

## Trained model: DentalSegmentator (CC-BY-4.0)

The recommended segmentation model is **DentalSegmentator**, distributed separately
under Creative Commons Attribution 4.0. It is **not** included in this repository;
download it yourself and keep the attribution below.

- Model: DentalSegmentator nnU-Net pretrained model for CBCT/CT segmentation
- Source (Zenodo): https://zenodo.org/records/10829675  (DOI: 10.5281/zenodo.10829675)
- License: CC-BY-4.0
- Citation (required):
  > Dot G, et al. *DentalSegmentator: robust open source deep learning-based CT and
  > CBCT image segmentation.* Journal of Dentistry (2024). doi:10.1016/j.jdent.2024.105130

## Framework: nnU-Net v2

  > Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a
  > self-configuring method for deep learning-based biomedical image segmentation.*
  > Nat Methods. 2021;18(2):203-211. doi:10.1038/s41592-020-01008-z

## DICOM conversion: dcm2niix

  > Li X, Morgan PS, Ashburner J, Smith J, Rorden C. *The first step for
  > neuroimaging data analysis: DICOM to NIfTI conversion.* J Neurosci Methods. 2016.
  > https://github.com/rordenlab/dcm2niix

## Mesh/IO libraries

VTK, SimpleITK, scikit-image, NumPy, pydicom — see each project's license.
