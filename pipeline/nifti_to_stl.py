"""NIfTI セグメンテーション → ラベル別 STL 変換。

DICOM_to_STL アプリ (v4.4) の nifti_to_stl.py を踏襲。
marching cubes でメッシュ抽出 → VTK で平滑化 → 法線を外向きに整えて
バイナリ STL を書き出す。個人情報は一切扱わない（ボクセルラベルのみ）。
"""

from __future__ import annotations

import os

import numpy as np
import SimpleITK as sitk
from skimage import measure
import vtk
from vtk.util import numpy_support


# Dataset111_453CT の 5 ラベル定義（学習時のラベルに対応）
LABEL_NAMES: dict[int, str] = {
    1: "Upper_Skull",       # 上顎・頭蓋
    2: "Mandible",          # 下顎骨
    3: "Upper_Teeth",       # 上顎歯列
    4: "Lower_Teeth",       # 下顎歯列
    5: "Mandibular_canal",  # 下顎管
}


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """三角メッシュの符号付き体積。負なら法線が反転している。"""
    tri = verts[faces]
    ref = tri.mean(axis=(0, 1), keepdims=True)
    tri -= ref
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def _orient_polydata_outward(poly: "vtk.vtkPolyData") -> "vtk.vtkPolyData":
    """法線の平均向きを重心から評価し、内向きが多数なら反転する。"""
    if poly is None or poly.GetNumberOfPoints() == 0:
        return poly

    point_data = poly.GetPointData()
    normal_array = point_data.GetNormals() if point_data is not None else None
    if normal_array is None:
        return poly

    normals_np = numpy_support.vtk_to_numpy(normal_array)
    if normals_np.size == 0:
        return poly

    points_np = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    if points_np.size == 0:
        return poly

    centroid = points_np.mean(axis=0)
    vectors = points_np - centroid
    dots = np.einsum("ij,ij->i", normals_np, vectors)

    if np.mean(dots) >= 0:
        return poly

    flipper = vtk.vtkReverseSense()
    flipper.SetInputData(poly)
    flipper.ReverseCellsOn()
    flipper.ReverseNormalsOn()
    flipper.Update()
    flipped = flipper.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(flipped)
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def nifti_to_stl(
    nifti_path: str,
    output_dir: str,
    label_values: list[int],
) -> list[str]:
    """NIfTI から指定ラベルの 3D メッシュを抽出し STL として保存する。

    Args:
        nifti_path: 入力 NIfTI（セグメンテーション）ファイルのパス。
        output_dir: STL 出力先ディレクトリ。
        label_values: 抽出対象のラベル値。

    Returns:
        書き出した STL ファイルパスのリスト。
    """
    print(f"Loading NIfTI: {os.path.basename(nifti_path)}")
    image = sitk.ReadImage(nifti_path)
    image_array = sitk.GetArrayFromImage(image)

    os.makedirs(output_dir, exist_ok=True)

    spacing_xyz = np.array(image.GetSpacing(), dtype=np.float64)
    origin_xyz = np.array(image.GetOrigin(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing_for_mc = spacing_xyz[::-1]  # marching_cubes は (z, y, x)

    written: list[str] = []

    for label_value in label_values:
        label_name = LABEL_NAMES.get(label_value, f"label_{label_value}")
        output_stl_file = os.path.join(output_dir, f"{label_name}.stl")

        mask = image_array == label_value
        if not np.any(mask):
            print(f"Warn: no voxels for label {label_value} ({label_name}), skip")
            continue

        print(f"Marching cubes for {label_name}...")
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(np.uint8), level=0.5, spacing=spacing_for_mc
        )
        if len(verts) == 0 or len(faces) == 0:
            print(f"Warn: empty mesh for {label_name}, skip")
            continue

        verts_xyz = verts[:, ::-1]  # (z,y,x) -> (x,y,z)
        verts_physical = np.ascontiguousarray(
            (direction @ verts_xyz.T).T + origin_xyz
        )

        if _signed_volume(verts_physical, faces) < 0:
            print(f"Orientation flipped for {label_name}; fixing winding")
            faces = faces[:, [0, 2, 1]]

        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(verts_physical, deep=True))

        polys = vtk.vtkCellArray()
        vtk_faces = np.hstack((np.full((faces.shape[0], 1), 3), faces)).ravel()
        polys.SetCells(
            faces.shape[0],
            numpy_support.numpy_to_vtk(
                vtk_faces, deep=True, array_type=vtk.VTK_ID_TYPE
            ),
        )

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(polys)

        print(f"Smoothing {label_name}...")
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(poly_data)
        smoother.SetNumberOfIterations(30)
        smoother.SetPassBand(0.01)
        smoother.SetFeatureEdgeSmoothing(False)
        smoother.SetBoundarySmoothing(True)
        smoother.SetNonManifoldSmoothing(True)
        smoother.Update()

        final_poly = smoother.GetOutput()
        if not final_poly or final_poly.GetNumberOfPoints() == 0:
            print(f"Warn: no smoothed data for {label_name}, skip write")
            continue

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(final_poly)
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.AutoOrientNormalsOn()
        normals.Update()

        oriented_poly = _orient_polydata_outward(normals.GetOutput())

        print(f"Writing STL: {os.path.basename(output_stl_file)}")
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(output_stl_file)
        writer.SetInputData(oriented_poly)
        set_binary = getattr(writer, "SetFileModeToBinary", None)
        if callable(set_binary):
            set_binary()
        else:
            writer.SetFileTypeToBinary()
        if writer.Write() != 1 or not os.path.exists(output_stl_file):
            raise RuntimeError(f"STL の書き込みに失敗しました: {output_stl_file}")
        written.append(output_stl_file)

    return written
