#!/usr/bin/env python3
"""DICOM 匿名化（de-identification）ステップ。

入力 DICOM フォルダを **コピー** し、コピー側から PHI（患者個人情報）タグを
除去／空白化する。元データは変更しない。形状再構成に必要な幾何情報
（ImagePositionPatient / ImageOrientationPatient / PixelSpacing / 画素）は保持する。

プライバシー設計:
  - 標準出力にタグの **値** を一切表示しない（タグ名と件数のみ）。
  - 呼び出し側エージェントは値を見ずに「何件処理したか」だけを確認できる。

参考: DICOM PS3.15 Annex E（Basic Application Level Confidentiality Profile）の
実務サブセット。完全準拠ではないため、院外公開・論文用途では追加確認を推奨。

使い方:
  python3 anonymize_dicom.py --in /path/RAW_DICOM --out /path/ANON_DICOM \
      [--pseudo-id CASE001]
終了コード: 0=成功 / 2=エラー
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Callable, Optional

Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(msg, flush=True)


# 完全に除去（削除）するタグ：直接識別子・連絡先・施設識別など
REMOVE_TAGS: list[tuple[int, int]] = [
    (0x0010, 0x1040),  # PatientAddress
    (0x0010, 0x2154),  # PatientTelephoneNumbers
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x2160),  # EthnicGroup
    (0x0010, 0x4000),  # PatientComments
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0008, 0x1050),  # PerformingPhysicianName
    (0x0008, 0x1070),  # OperatorsName
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1010),  # StationName
    (0x0010, 0x2180),  # Occupation
    (0x0010, 0x21B0),  # AdditionalPatientHistory
    (0x0032, 0x1032),  # RequestingPhysician
    (0x0038, 0x0300),  # CurrentPatientLocation
]

# 空文字に置換（ゼロ長化）するタグ：日付など、削除すると読込に支障が出る場合
BLANK_TAGS: list[tuple[int, int]] = [
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0032),  # PatientBirthTime
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1020),  # PatientSize
    (0x0010, 0x1030),  # PatientWeight
]


def anonymize_dataset(ds, pseudo_id: str) -> int:
    """1 データセットを匿名化。変更したタグ数を返す（値は返さない）。"""
    changed = 0
    for tag in REMOVE_TAGS:
        if tag in ds:
            del ds[tag]
            changed += 1
    for tag in BLANK_TAGS:
        if tag in ds:
            ds[tag].value = ""
            changed += 1
    # 患者名・患者 ID は擬似 ID に置換（完全削除すると不整合になりうるため）
    if (0x0010, 0x0010) in ds:
        ds[(0x0010, 0x0010)].value = pseudo_id
        changed += 1
    if (0x0010, 0x0020) in ds:
        ds[(0x0010, 0x0020)].value = pseudo_id
        changed += 1
    # 画像コメント等の自由記述も空白化
    for tag in [(0x0020, 0x4000), (0x0008, 0x103E)]:  # ImageComments, SeriesDescription
        if tag in ds and isinstance(ds[tag].value, str):
            ds[tag].value = ""
            changed += 1
    # PatientIdentityRemoved を明示
    try:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "PenClaw dicom-to-stl-pipeline anonymize_dicom"
    except Exception:  # noqa: BLE001
        pass
    return changed


def anonymize_dir(
    src: str, dst: str, pseudo_id: str, log: Log = _log
) -> tuple[int, int]:
    """src を dst にコピーしつつ匿名化。(処理ファイル数, 変更タグ総数) を返す。"""
    try:
        import pydicom  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"pydicom が必要です（pip install pydicom）。詳細: {exc}"
        )

    if not os.path.isdir(src):
        raise RuntimeError(f"入力フォルダが存在しません: {src}")
    if os.path.abspath(src) == os.path.abspath(dst):
        raise RuntimeError("入力と出力は別フォルダにしてください。")
    os.makedirs(dst, exist_ok=True)

    n_files = 0
    n_changed = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_root = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out_root, exist_ok=True)
        for fn in files:
            in_path = os.path.join(root, fn)
            out_path = os.path.join(out_root, fn)
            try:
                ds = pydicom.dcmread(in_path, force=True)
            except Exception:
                # DICOM でないファイルはそのままコピー（DICOMDIR 等を除き通常は無害）
                shutil.copy2(in_path, out_path)
                continue
            n_changed += anonymize_dataset(ds, pseudo_id)
            ds.save_as(out_path)
            n_files += 1

    if n_files == 0:
        raise RuntimeError("DICOM ファイルが 1 枚も見つかりませんでした。")
    log(f"匿名化完了: {n_files} 枚を処理、{n_changed} タグを除去/置換しました。")
    log("（タグ値はプライバシー保護のため非表示）")
    return n_files, n_changed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DICOM 匿名化（PHI 除去）")
    p.add_argument("--in", dest="src", required=True, help="生 DICOM フォルダ")
    p.add_argument("--out", dest="dst", required=True, help="匿名化出力フォルダ")
    p.add_argument(
        "--pseudo-id",
        default="ANON",
        help="患者名・ID に入れる擬似 ID（既定: ANON）",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        anonymize_dir(args.src, args.dst, args.pseudo_id)
    except Exception as exc:  # noqa: BLE001
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
