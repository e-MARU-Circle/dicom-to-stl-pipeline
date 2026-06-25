#!/usr/bin/env python3
"""DICOM → NIfTI → nnU-Netv2 → STL を 1 コマンドで実行するヘッドレス・パイプライン。

DICOM_to_STL アプリ (v4.4 / main_app.py) のロジックを GUI・PyInstaller 依存なしに
再構成したもの。学習モデル本体と venv は同梱しない（各自が用意する。再配布禁止）。

外部配布（実行環境がまちまち）を想定し、元アプリで苦労した推論パラメータを
**プリセット＋個別フラグ**で切替可能にしてある。背景は references/parameter_tuning.md。

設計上の原則（プライバシー）:
  - 本スクリプトは DICOM の画素・タグを **標準出力に一切表示しない**。
    進捗・ファイル名・件数のみをログする。呼び出し側エージェントは
    DICOM/NIfTI の中身を読まず、パスを渡して終了コードだけを見ればよい。
  - PHI 除去は anonymize_dicom.py（別ステップ）で行う。--require-anonymized 指定時、
    入力に PHI が残っていれば実行を拒否する（どのエージェントでも機械的に効く防壁）。

使い方の例:
  python3 run_pipeline.py --check --model-dir /path/to/nnUNet_results   # 環境プリフライト
  python3 run_pipeline.py --in ANON_DIR --out STL_DIR \
      --model-dir /path/to/nnUNet_results --device auto --accept-disclaimer

終了コード: 0=成功 / 1=免責未同意 or check失敗 / 2=エラー / 3=PHIガード違反 / 99=想定外
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Iterable, Optional

# 既定は公開モデル DentalSegmentator（Zenodo 10829675, CC-BY-4.0）。
# 解凍後のフォルダ名をそのまま使う（例: Dataset112_DentalSegmentator）。
# 自前モデルを使う場合は --dataset で上書き（例: Dataset111_453CT）。
DEFAULT_DATASET = "Dataset112_DentalSegmentator"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_FOLD = "0"
TARGET_LABELS = [1, 2, 3, 4, 5]

# 推論プリセット（元アプリの実測チューニング由来。詳細は references/parameter_tuning.md）
#   fast        : 速度優先。TTA 無効 + step_size 0.7（精度差は軽微、約 8〜10 倍速）
#   quality     : 精度優先。TTA 有効 + step_size 0.5（GPU 推奨。CPU では非常に遅い）
#   low-resource: 低メモリ/権限制約環境。逐次実行(npp=nps=0) + TTA 無効 + step_size 0.7
PRESETS = {
    "fast": dict(step_size=0.7, tta=False, sequential=False),
    "quality": dict(step_size=0.5, tta=True, sequential=False),
    "low-resource": dict(step_size=0.7, tta=False, sequential=True),
}

DISCLAIMER = (
    "本ソフトウェアは薬機法上の医療機器ではなく、研究用途に限定されます。"
    "診断・治療の根拠として使用しないでください。出力結果について一切の保証・"
    "責任を負いません。同意する場合は --accept-disclaimer を付けて実行してください。"
)

Log = Callable[[str], None]


class PipelineError(Exception):
    """パイプラインのいずれかの工程が失敗したときに送出。"""


class PHIGuardError(Exception):
    """匿名化されていない入力を検出したときに送出。"""


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 外部実行ファイルの解決
# --------------------------------------------------------------------------- #
def resolve_dcm2niix(explicit: Optional[str]) -> str:
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        raise PipelineError(f"指定された dcm2niix が実行できません: {explicit}")
    found = shutil.which("dcm2niix") or shutil.which("dcm2niix.exe")
    if found:
        return found
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("dcm2niix", "dcm2niix.exe"):
        cand = os.path.join(here, "bin", name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise PipelineError(
        "dcm2niix が見つかりません。PATH に追加するか --dcm2niix で指定してください。"
    )


# --------------------------------------------------------------------------- #
# デバイス検出
# --------------------------------------------------------------------------- #
def available_devices(log: Optional[Log] = None) -> list[str]:
    available = ["cpu"]
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            available.insert(0, "cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            idx = 1 if available and available[0] == "cuda" else 0
            available.insert(idx, "mps")
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"PyTorch 検出をスキップ（CPU 前提）: {exc}")
    return list(dict.fromkeys(available))


def detect_device(preferred: str, log: Log) -> str:
    available = available_devices(log)
    preferred = preferred.lower()
    if preferred == "auto":
        chosen = available[0]
        log(f"利用可能デバイス: {', '.join(available)} → '{chosen}' を使用")
        return chosen
    if preferred not in available:
        log(f"デバイス '{preferred}' は不可。利用可能: {', '.join(available)}。CPU へフォールバック")
        return "cpu"
    log(f"デバイス '{preferred}' を使用")
    return preferred


# --------------------------------------------------------------------------- #
# 推論フラグ構築（プリセット + 個別上書き）
# --------------------------------------------------------------------------- #
def build_predict_flags(
    *,
    preset: str,
    step_size: Optional[float],
    tta: Optional[bool],
    sequential: Optional[bool],
    npp: Optional[int],
    nps: Optional[int],
    disable_progress_bar: bool,
) -> list[str]:
    base = dict(PRESETS[preset])
    if step_size is not None:
        base["step_size"] = step_size
    if tta is not None:
        base["tta"] = tta
    if sequential is not None:
        base["sequential"] = sequential

    if not (0 < float(base["step_size"]) <= 1):
        raise PipelineError("step_size は 0 < 値 <= 1 で指定してください。")

    flags: list[str] = ["-step_size", str(base["step_size"])]
    if not base["tta"]:
        flags.append("--disable_tta")
    if base["sequential"]:
        flags += ["-npp", "0", "-nps", "0"]
    else:
        if npp is not None:
            flags += ["-npp", str(npp)]
        if nps is not None:
            flags += ["-nps", str(nps)]
    if disable_progress_bar:
        flags.append("--disable_progress_bar")
    return flags


# --------------------------------------------------------------------------- #
# PHI ガード（軽量チェック。値は表示しない）
# --------------------------------------------------------------------------- #
def assert_anonymized(dicom_dir: str, log: Log) -> None:
    """匿名化されていなければ PHIGuardError。タグ値は出力しない。

    判定:
      - anonymize_dicom.py が立てる PatientIdentityRemoved == "YES" があれば合格。
      - マーカーが無い場合は、生の識別子タグ（氏名/ID/生年月日/住所/電話）が
        非空なら未匿名化とみなして拒否する。
    """
    try:
        import pydicom  # type: ignore
    except Exception:
        log("注意: pydicom 未導入のため PHI ガードを省略します（匿名化済み前提で続行）。")
        return

    phi_tags = [
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x1040),  # PatientAddress
        (0x0010, 0x2154),  # PatientTelephoneNumbers
    ]
    checked = 0
    for root, _dirs, files in os.walk(dicom_dir):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            checked += 1

            if str(getattr(ds, "PatientIdentityRemoved", "")).strip().upper() == "YES":
                if checked >= 50:
                    log(f"PHI ガード: {checked} 枚を検査、匿名化マーカーを確認。")
                    return
                continue

            for tag in phi_tags:
                if tag in ds and str(ds[tag].value).strip():
                    raise PHIGuardError(
                        "入力が匿名化されていません。先に anonymize_dicom.py を実行してください。"
                        "（どのタグかはセキュリティのため非表示）"
                    )
            if checked >= 50:
                log(f"PHI ガード: {checked} 枚を検査、識別子タグなしを確認。")
                return
    log(f"PHI ガード: {checked} 枚を検査、未匿名化の識別子タグなしを確認。")


# --------------------------------------------------------------------------- #
# 各工程
# --------------------------------------------------------------------------- #
def run_command(cmd: Iterable[str], env: dict[str, str], log: Log) -> None:
    cmd = list(cmd)
    log(f"実行: {cmd[0]} ...（引数省略）")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        log(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise PipelineError(f"コマンドが異常終了しました（code={proc.returncode}）。")


def step_dicom_to_nifti(
    dcm2niix: str, dicom_dir: str, nifti_dir: str, log: Log
) -> str:
    log("--- ステップ1: DICOM → NIfTI ---")
    # -d 9: サブフォルダを 9 階層まで探索 / -i n: 派生画像も含める / -z y: gzip
    # （元アプリで「DICOM が見つからない」「複数シリーズ」問題に対応した実測値）
    cmd = [dcm2niix, "-o", nifti_dir, "-z", "y", "-d", "9", "-i", "n", dicom_dir]
    run_command(cmd, os.environ.copy(), log)

    produced = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
    if len(produced) != 1:
        raise PipelineError(
            f"NIfTI が 1 つではありません（{len(produced)} 個）。"
            "1 症例 = 1 シリーズになるよう DICOM フォルダを分けてください。"
        )
    case = os.path.join(nifti_dir, "case_0000.nii.gz")
    os.rename(os.path.join(nifti_dir, produced[0]), case)
    log("ステップ1 完了")
    return case


def step_segmentation(
    nifti_dir: str,
    seg_dir: str,
    *,
    dataset: str,
    configuration: str,
    fold: str,
    device: str,
    model_dir: str,
    predict_flags: list[str],
    threads: Optional[int],
    log: Log,
) -> None:
    log("--- ステップ2: nnU-Netv2 セグメンテーション ---")
    model_root = os.path.join(
        model_dir, dataset, f"nnUNetTrainer__nnUNetPlans__{configuration}"
    )
    if not os.path.isdir(model_root):
        raise PipelineError(
            f"学習モデルが見つかりません: {model_root}\n"
            f"各自で {dataset} の学習モデルを {model_dir} 配下に配置してください"
            f"（references/model_acquisition.md 参照）。"
        )

    env = os.environ.copy()
    env["nnUNet_results"] = model_dir
    env.setdefault("nnUNet_raw", os.path.join(tempfile.gettempdir(), "nnUNet_raw"))
    env.setdefault(
        "nnUNet_preprocessed",
        os.path.join(tempfile.gettempdir(), "nnUNet_preprocessed"),
    )
    os.makedirs(env["nnUNet_raw"], exist_ok=True)
    os.makedirs(env["nnUNet_preprocessed"], exist_ok=True)
    if threads and threads > 0:
        # CPU 推論のスレッド数を明示（環境差で既定が極端になるのを防ぐ）
        env["OMP_NUM_THREADS"] = str(threads)

    log(f"推論フラグ: {' '.join(predict_flags)}")
    common = [
        "-i", nifti_dir, "-o", seg_dir,
        "-d", dataset, "-c", configuration, "-f", fold,
        "-device", device, *predict_flags,
    ]
    exe = shutil.which("nnUNetv2_predict")
    if exe:
        run_command([exe, *common], env, log)
    else:
        log("nnUNetv2_predict が無いため python -m にフォールバックします。")
        run_command(
            [sys.executable, "-m", "nnunetv2.inference.predict", *common], env, log
        )
    log("ステップ2 完了")


def step_nifti_to_stl(seg_dir: str, out_dir: str, log: Log) -> list[str]:
    log("--- ステップ3: NIfTI → STL ---")
    from nifti_to_stl import nifti_to_stl  # 重い依存は実行時に読み込む

    seg_files = [f for f in os.listdir(seg_dir) if f.endswith(".nii.gz")]
    if not seg_files:
        raise PipelineError("セグメンテーション結果（.nii.gz）が見つかりません。")
    seg_path = os.path.join(seg_dir, sorted(seg_files)[0])
    written = nifti_to_stl(seg_path, out_dir, TARGET_LABELS)
    log(f"ステップ3 完了: {len(written)} 個の STL を出力")
    return written


# --------------------------------------------------------------------------- #
# 環境プリフライト（--check）: DICOM に触れず依存と準備状況だけを点検
# --------------------------------------------------------------------------- #
def run_check(model_dir: Optional[str], dataset: str, configuration: str, log: Log) -> int:
    ok = True
    log(f"Python: {sys.version.split()[0]}")

    try:
        import torch  # type: ignore

        log(f"torch: {torch.__version__} / デバイス: {', '.join(available_devices())}")
    except Exception as exc:  # noqa: BLE001
        log(f"[NG] torch 未導入: {exc}")
        ok = False

    try:
        import nnunetv2  # type: ignore  # noqa: F401

        log("nnunetv2: OK")
    except Exception as exc:  # noqa: BLE001
        log(f"[NG] nnunetv2 未導入: {exc}")
        ok = False

    for mod in ("SimpleITK", "skimage", "vtk", "numpy", "pydicom"):
        try:
            __import__(mod)
            log(f"{mod}: OK")
        except Exception as exc:  # noqa: BLE001
            critical = mod in ("SimpleITK", "skimage", "vtk", "numpy")
            log(f"[{'NG' if critical else '警告'}] {mod}: {exc}")
            ok = ok and not critical

    try:
        log(f"dcm2niix: {resolve_dcm2niix(None)}")
    except PipelineError as exc:
        log(f"[NG] {exc}")
        ok = False

    if model_dir:
        root = os.path.join(
            model_dir, dataset, f"nnUNetTrainer__nnUNetPlans__{configuration}"
        )
        ckpt = os.path.join(root, "fold_0", "checkpoint_final.pth")
        if os.path.isfile(ckpt):
            log(f"モデル: OK ({root})")
        else:
            log(f"[NG] モデル未配置: {root}（model_acquisition.md 参照）")
            ok = False
    else:
        log("モデル: --model-dir 未指定のため未確認")

    log("=== 準備完了 ===" if ok else "=== 不足あり（上記 [NG] を解消してください）===")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# オーケストレーション
# --------------------------------------------------------------------------- #
def run(
    dicom_dir: str,
    out_dir: str,
    *,
    model_dir: str,
    device: str,
    dataset: str,
    configuration: str,
    fold: str,
    dcm2niix: Optional[str],
    require_anonymized: bool,
    predict_flags: list[str],
    threads: Optional[int],
    log: Log = _log,
) -> list[str]:
    if not os.path.isdir(dicom_dir):
        raise PipelineError(f"入力フォルダが存在しません: {dicom_dir}")
    os.makedirs(out_dir, exist_ok=True)

    if require_anonymized:
        assert_anonymized(dicom_dir, log)

    dcm2niix_path = resolve_dcm2niix(dcm2niix)
    resolved_device = detect_device(device, log)

    case_name = os.path.basename(os.path.normpath(dicom_dir))
    case_out = os.path.join(out_dir, f"{case_name}_stl")
    os.makedirs(case_out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        nifti_dir = os.path.join(tmp, "nifti")
        seg_dir = os.path.join(tmp, "seg")
        os.makedirs(nifti_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        step_dicom_to_nifti(dcm2niix_path, dicom_dir, nifti_dir, log)
        step_segmentation(
            nifti_dir,
            seg_dir,
            dataset=dataset,
            configuration=configuration,
            fold=fold,
            device=resolved_device,
            model_dir=model_dir,
            predict_flags=predict_flags,
            threads=threads,
            log=log,
        )
        written = step_nifti_to_stl(seg_dir, case_out, log)

    log("=== 完了: 出力 STL ===")
    for path in written:
        log(f"  {os.path.basename(path)}")
    return written


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DICOM → NIfTI → nnU-Netv2 → STL ヘッドレス・パイプライン"
    )
    p.add_argument("--in", dest="dicom_dir", help="入力 DICOM フォルダ（匿名化済み推奨）")
    p.add_argument("--out", dest="out_dir", help="STL 出力フォルダ")
    p.add_argument(
        "--model-dir",
        default=os.environ.get("NNUNET_RESULTS_DIR", "./nnUNet_results"),
        help="学習モデルを置いた nnUNet_results ディレクトリ（各自で配置）",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    p.add_argument("--fold", default=DEFAULT_FOLD)
    p.add_argument("--dcm2niix", default=None, help="dcm2niix 実行ファイルの明示パス")

    # 推論パラメータ（プリセット + 個別上書き）。詳細は references/parameter_tuning.md
    p.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="fast",
        help="推論プリセット: fast(既定/高速) / quality(高精度・GPU推奨) / low-resource(低メモリ)",
    )
    p.add_argument("--step-size", type=float, default=None, help="スライディング窓の重なり(0<x<=1)。小さいほど高精度・低速")
    tta = p.add_mutually_exclusive_group()
    tta.add_argument("--tta", dest="tta", action="store_true", default=None, help="TTA(8方向ミラー)を有効化（高精度・約8倍遅い）")
    tta.add_argument("--no-tta", dest="tta", action="store_false", help="TTA を無効化（高速）")
    p.add_argument("--sequential", dest="sequential", action="store_true", default=None, help="逐次実行(npp=nps=0)。低メモリ/権限制約環境向け")
    p.add_argument("--npp", type=int, default=None, help="前処理ワーカ数")
    p.add_argument("--nps", type=int, default=None, help="セグ出力ワーカ数")
    p.add_argument("--threads", type=int, default=None, help="CPU推論スレッド数(OMP_NUM_THREADS)")
    p.add_argument("--disable-progress-bar", action="store_true", help="進捗バーを無効化")

    p.add_argument("--require-anonymized", action="store_true", help="PHI タグが残る入力を実行前に拒否（院外共有時に推奨）")
    p.add_argument("--accept-disclaimer", action="store_true", help="研究用途・免責事項に同意した場合に指定")
    p.add_argument("--check", action="store_true", help="DICOM に触れず実行環境の準備状況だけを点検して終了")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.check:
        return run_check(args.model_dir, args.dataset, args.configuration, _log)

    if not args.dicom_dir or not args.out_dir:
        print("--in と --out は必須です（環境点検だけなら --check）。", file=sys.stderr)
        return 1
    if not args.accept_disclaimer:
        print(DISCLAIMER, file=sys.stderr)
        return 1

    try:
        predict_flags = build_predict_flags(
            preset=args.preset,
            step_size=args.step_size,
            tta=args.tta,
            sequential=args.sequential,
            npp=args.npp,
            nps=args.nps,
            disable_progress_bar=args.disable_progress_bar,
        )
        run(
            args.dicom_dir,
            args.out_dir,
            model_dir=args.model_dir,
            device=args.device,
            dataset=args.dataset,
            configuration=args.configuration,
            fold=args.fold,
            dcm2niix=args.dcm2niix,
            require_anonymized=args.require_anonymized,
            predict_flags=predict_flags,
            threads=args.threads,
        )
    except PHIGuardError as exc:
        print(f"PHI ガード違反: {exc}", file=sys.stderr)
        return 3
    except PipelineError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"想定外のエラー: {exc}", file=sys.stderr)
        return 99
    return 0


if __name__ == "__main__":
    sys.exit(main())
