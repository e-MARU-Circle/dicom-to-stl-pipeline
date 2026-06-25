# pipeline/ — DICOM → STL 変換の実体

このフォルダのスクリプトがパイプライン本体です。**学習モデルと venv は同梱しません**
（各自で用意。`references/` の手順を参照）。

## 構成

| ファイル | 役割 |
|---|---|
| `run_pipeline.py` | DICOM → NIfTI(dcm2niix) → nnU-Netv2 推論 → STL を 1 コマンド実行 |
| `anonymize_dicom.py` | DICOM から PHI を除去（**二段プライバシーの 1 段目**） |
| `nifti_to_stl.py` | セグメンテーション NIfTI から 5 ラベル別 STL を生成（marching cubes + 平滑化） |
| `requirements.txt` | 後段・匿名化の共通依存（torch/nnunetv2 は別途） |

## 標準フロー（匿名化 → 変換）

```bash
# 1) 匿名化（PHI 除去・元データは不変、別フォルダにコピー）
python3 anonymize_dicom.py --in /data/RAW_CASE --out /data/ANON_CASE --pseudo-id CASE001

# 2) 変換（匿名化済みのみ許可）
python3 run_pipeline.py \
    --in  /data/ANON_CASE \
    --out /data/STL_OUT \
    --model-dir /path/to/nnUNet_results \
    --device auto \
    --require-anonymized \
    --accept-disclaimer
```

出力: `/data/STL_OUT/CASE001_stl/` 配下に
`Upper_Skull.stl` `Mandible.stl` `Upper_Teeth.stl` `Lower_Teeth.stl` `Mandibular_canal.stl`。

## ラベル定義（DentalSegmentator 互換 / 5 クラス）

| ラベル | 名称 |
|---|---|
| 1 | Upper_Skull（上顎・頭蓋） |
| 2 | Mandible（下顎骨） |
| 3 | Upper_Teeth（上顎歯列） |
| 4 | Lower_Teeth（下顎歯列） |
| 5 | Mandibular_canal（下顎管） |

## 終了コード（run_pipeline.py）

`0`=成功 / `1`=免責未同意 / `2`=パイプラインエラー / `3`=PHI ガード違反 / `99`=想定外

## プライバシー上の約束

- どちらのスクリプトも **DICOM のタグ値・画素を標準出力に出さない**（ファイル名・件数・進捗のみ）。
- そのため、呼び出し側エージェントは中身を読まずに終了コードと出力ファイル名だけで完了確認できる。
- 後段の STL はボクセルラベル由来の形状のみで、患者氏名等は含まれない。
