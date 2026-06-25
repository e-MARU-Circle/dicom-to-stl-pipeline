# dicom-to-stl-pipeline

歯科CT(DICOM/CBCT)を AI でセグメンテーションし、**5つの解剖パーツ（上顎・下顎骨・上顎歯・下顎歯・下顎管）を STL** に書き出す、ローカル完結のヘッドレス・パイプラインです。

> **English summary**: A local, headless pipeline that turns dental CT/CBCT (DICOM) into STL meshes
> of 5 anatomical structures (upper skull, mandible, upper teeth, lower teeth, mandibular canal)
> via nnU-Net v2 segmentation. **Model weights and runtime are not bundled** — you provide them
> yourself. DICOM is de-identified first, and the tooling never prints PHI. **Research use only.**

```
DICOM ──► dcm2niix ──► NIfTI ──► nnU-Netv2 (AI) ──► 5ラベル ──► marching cubes + 平滑化 ──► STL
```

---

## ⚠️ はじめに（重要）

- **研究用途限定。** 本ソフトは薬機法上の医療機器ではなく、診断・治療の根拠に使うものではありません。
- **学習モデル（重み）は同梱しません。** 推奨モデル DentalSegmentator（CC-BY-4.0）を各自でダウンロードしてください（[ATTRIBUTION.md](ATTRIBUTION.md)）。
- **患者個人情報(PHI)は扱いません。** 付属の匿名化ツールで PHI を除去してから処理します。スクリプトは DICOM のタグ値を画面に出しません。

---

## できること

CBCT/CT の DICOM フォルダを渡すと、3Dプリントや設計に使える STL を5つ出力します。

| ラベル | ファイル名 |
|---|---|
| 上顎・頭蓋 | `Upper_Skull.stl` |
| 下顎骨 | `Mandible.stl` |
| 上顎歯列 | `Upper_Teeth.stl` |
| 下顎歯列 | `Lower_Teeth.stl` |
| 下顎管 | `Mandibular_canal.stl` |

## 必要なもの

1. **Python 3.10〜3.13** が動く PC（Mac / Windows / Linux）
2. **学習モデル**：DentalSegmentator を [Zenodo](https://zenodo.org/records/10829675) からダウンロード（[docs](references/model_acquisition.md)）
3. **dcm2niix**（DICOM→NIfTI 変換ツール）
4. Python ライブラリ（PyTorch / nnU-Netv2 / SimpleITK / scikit-image / VTK / pydicom）

セットアップ手順は環境別に用意しています → [references/environment_setup.md](references/environment_setup.md)
（macOS の MPS/CPU、Windows の CUDA、汎用CPU）。

## 入手

```bash
git clone https://github.com/e-MARU-Circle/dicom-to-stl-pipeline.git
cd dicom-to-stl-pipeline
```

## クイックスタート

```bash
# 0) 環境が整っているか点検（DICOM には触れません）
python3 pipeline/run_pipeline.py --check --model-dir ./nnUNet_results

# 1) 匿名化（PHI 除去・元データは変更しない）
python3 pipeline/anonymize_dicom.py --in RAW_CASE --out ANON_CASE --pseudo-id CASE001

# 2) 変換（DICOM → STL）
python3 pipeline/run_pipeline.py \
    --in  ANON_CASE \
    --out STL_OUT \
    --model-dir ./nnUNet_results \
    --device auto \
    --require-anonymized \
    --accept-disclaimer
```

出力 → `STL_OUT/CASE001_stl/` に 5 つの STL。

## 速度と精度（プリセット）

環境に合わせて選べます。詳しい背景は [references/parameter_tuning.md](references/parameter_tuning.md)。

| プリセット | 用途 |
|---|---|
| `--preset fast`（既定） | まず動かす。TTA 無効＋step 0.7 で高速 |
| `--preset quality` | GPU(CUDA) で精度重視。TTA 有効＋step 0.5 |
| `--preset low-resource` | 低メモリ／処理が固まる環境。逐次実行 |

## プライバシー設計（二段）

1. **匿名化**：`anonymize_dicom.py` が PHI（氏名・ID・生年月日・住所・電話など）を除去した別フォルダを作る。元データは不変。
2. **分離実行**：処理はローカルのサブプロセスで完結し、DICOM/STL の中身を画面に出さない。`--require-anonymized` を付けると、匿名化されていない入力を機械的に拒否する。

外部にデータを送信しません。すべてお手元の PC 内で処理されます。

## ライセンス / クレジット

- 本リポジトリの**コードは MIT License**（[LICENSE](LICENSE)）。
- **学習モデル DentalSegmentator は CC-BY-4.0**（同梱せず・各自取得・出典表示が必要）。
- 出典・引用は [ATTRIBUTION.md](ATTRIBUTION.md) を参照（Dot G et al. 2024 / nnU-Net / dcm2niix）。

## 免責

本ソフトウェアの使用または使用不能によって生じたいかなる損害についても、作者は一切の責任を負いません。
医療機器ではなく、研究用途に限定されます。患者データの取り扱いは各自の責任で、関連法規・院内規定に従ってください。

---

<sub>Claude（Cowork）のスキルとしても利用できます（`SKILL.md` 同梱）。</sub>
