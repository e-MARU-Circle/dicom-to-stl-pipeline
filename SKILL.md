---
name: dicom-to-stl-pipeline
description: "歯科CT(DICOM)を AI セグメンテーションで 5 ラベル(上顎/下顎骨/上顎歯/下顎歯/下顎管)に分割し STL を生成するヘッドレス・パイプライン。DICOM→NIfTI(dcm2niix)→nnU-Netv2 推論→marching cubes+平滑化→STL を 1 コマンドで実行。学習モデルと実行環境は各自で用意（同梱・再配布なし）。エージェントは DICOM の中身・患者個人情報(PHI)に触れず、匿名化→パス渡しのみで実行する二段プライバシー設計。「DICOM」「CBCT」「CT」「STL変換」「DICOMをSTL」「セグメンテーション」「歯のセグメンテーション」「下顎管」「nnUNet」「nnU-Net」「歯科CT」「3D再構成」「匿名化」「de-identify」「DICOM匿名化」「PHI除去」「run_pipeline」「DICOM_to_STL」「DentalSegmentator」と言われたら発動。実行・チューニングの主担当はコード(penclaw-ml)。"
---

# dicom-to-stl-pipeline — 歯科CT(DICOM) → STL 変換

歯科CT(DICOM)から AI で骨・歯・下顎管を分割し、3D プリント／設計に使える STL を生成する
ヘッドレス・パイプライン。DICOM_to_STL アプリ (v4.4) のロジックを GUI 非依存で再構成し、
配布できる形にしたもの。3D実行の主担当は**コード（penclaw-ml）**。生成 STL は
**blender-dental** の中空オープン模型パイプラインへそのまま流せる。

## このスキルが「やらない」こと（重要）

- **学習モデル（重み .pth）を同梱しない・再配布しない。** 各自が用意する → `references/model_acquisition.md`。
- **実行環境(venv)を同梱しない。** 各自が一度だけ構築する → `references/environment_setup.md`。
- **エージェントは患者個人情報(PHI)に触れない。** DICOM のタグ値・画素を読み込まず、
  匿名化済みフォルダのパスを渡して終了コードと出力ファイル名だけを確認する。

## 二段プライバシー設計（必ず守る）

DICOM はヘッダに患者氏名・ID・生年月日などの PHI を含む。エージェントが PHI に触れないよう、
2 段で分離する。

1. **匿名化（1 段目）**: `pipeline/anonymize_dicom.py` を**サブプロセスとして**実行し、
   PHI を除去した別フォルダを作る。元データは変更しない。スクリプトはタグ**値**を出力しない。
2. **オーケストレーション分離（2 段目）**: エージェントは DICOM/NIfTI/STL の中身を
   `Read` で開かない。`run_pipeline.py` にフォルダ**パス**を渡し、標準出力（進捗・件数・
   ファイル名のみ）と終了コードだけを扱う。

### エージェントの禁止事項

- DICOM ファイルを `Read` で開く／タグを `print` する／画素を読む — **禁止**。
- NIfTI(.nii.gz)・STL の中身をコンテキストに展開する — 不要（パスで十分）。
- 患者データを外部（Notion/Chatwork/Gmail/クラウド）へ送る — **禁止**。
- 生 DICOM を匿名化せずに `run_pipeline.py` へ渡す — 院外用途では `--require-anonymized` で機械的に拒否。

## 配布・実行環境について（他者の環境で動かす前提）

このスキルは **PenClaw 固有のエージェントや MCP に依存しない**。ローカルでシェルを実行でき、
フォルダパスを渡せる環境ならどのエージェント／手動でも動く。`run_pipeline.py` の機械的な防壁
（`--require-anonymized` と「PHI を標準出力に出さない」設計）が、エージェントの挙動に関わらず効く。
PenClaw 内ではコード（penclaw-ml）が主担当だが、**外部環境ではセットアップ担当の任意のエージェント／利用者**が
references のプロンプトを実行すればよい。

## 起動時チェック

0. **まず環境点検**: `python3 pipeline/run_pipeline.py --check --model-dir <DIR>` を実行。
   DICOM に触れず、依存（torch/nnunetv2/dcm2niix 等）とモデル配置の準備状況だけを報告する。
1. 環境が未構築なら **`references/environment_setup.md`** の該当プロンプトを案内（Mac MPS/CPU / Win CUDA / 汎用CPU）。
2. 学習モデル未配置なら **`references/model_acquisition.md`** を案内。`--model-dir` のパスを確認。
3. 入力が生 DICOM か匿名化済みかを利用者に確認。院外共有・論文用途なら匿名化必須。
4. 環境に応じて推論プリセットを選ぶ（**`references/parameter_tuning.md`**）。迷ったら既定 `fast`。

## 標準ワークフロー

```bash
# 1) 匿名化（PHI 除去・別フォルダにコピー、元データ不変）
python3 pipeline/anonymize_dicom.py \
    --in  /data/RAW_CASE --out /data/ANON_CASE --pseudo-id CASE001

# 2) 変換（DICOM→NIfTI→nnU-Net→STL）
python3 pipeline/run_pipeline.py \
    --in  /data/ANON_CASE \
    --out /data/STL_OUT \
    --model-dir /path/to/nnUNet_results \
    --device auto \
    --require-anonymized \
    --accept-disclaimer
```

出力: `/data/STL_OUT/CASE001_stl/` に
`Upper_Skull.stl` `Mandible.stl` `Upper_Teeth.stl` `Lower_Teeth.stl` `Mandibular_canal.stl`。

### よく使うオプション（run_pipeline.py）

| オプション | 意味 |
|---|---|
| `--check` | DICOM に触れず環境の準備状況だけを点検して終了（プリフライト） |
| `--preset fast/quality/low-resource` | 推論プリセット。既定 fast。詳細・個別フラグは `references/parameter_tuning.md` |
| `--device auto/cpu/cuda/mps` | 推論デバイス。auto は cuda>mps>cpu の順に自動選択 |
| `--model-dir DIR` | 学習モデルの nnUNet_results。`NNUNET_RESULTS_DIR` でも可 |
| `--dataset / --configuration / --fold` | 既定 `Dataset112_DentalSegmentator / 3d_fullres / 0`（自前は `--dataset Dataset111_453CT`） |
| `--dcm2niix PATH` | dcm2niix を明示指定（PATH に無い場合） |
| `--require-anonymized` | PHI タグが残る入力を実行前に拒否（院外用途で推奨） |
| `--accept-disclaimer` | 研究用途・免責に同意（未指定だと実行しない） |

推論チューニング（TTA・step_size・逐次実行・スレッド）で迷ったら **`references/parameter_tuning.md`**。
元アプリで苦労した実測知見をプリセットに集約してある。

終了コード: `0`成功 / `1`免責未同意 / `2`エラー / `3`PHIガード違反 / `99`想定外。

## ラベル定義（DentalSegmentator 互換 / 5 クラス）

1=Upper_Skull（上顎/頭蓋）, 2=Mandible（下顎骨）, 3=Upper_Teeth（上顎歯）,
4=Lower_Teeth（下顎歯）, 5=Mandibular_canal（下顎管）。

推奨の公開モデルは **DentalSegmentator**（Zenodo 10829675, CC-BY-4.0）。取得・配置・出典表示は
`references/model_acquisition.md`。自前の `Dataset111_453CT` も同構造で `--dataset` 上書きで利用可。

## 構成

```
dicom-to-stl-pipeline/
├── SKILL.md
├── pipeline/
│   ├── run_pipeline.py      # DICOM→NIfTI→nnU-Net→STL 本体（CLI）
│   ├── anonymize_dicom.py   # PHI 除去（二段プライバシー 1 段目）
│   ├── nifti_to_stl.py      # NIfTI→5ラベルSTL（marching cubes+平滑化）
│   ├── requirements.txt     # 後段・匿名化の共通依存（torch/nnunetv2 は別途）
│   └── README.md
└── references/
    ├── environment_setup.md # 環境構築プロンプト（Mac MPS/CPU・Win CUDA・汎用CPU）
    ├── model_acquisition.md # 学習モデル取得・配置プロンプト（DentalSegmentator/Zenodo・再配布禁止）
    └── parameter_tuning.md  # 推論/メッシュのパラメータ調整知見＋プリセット早見
```

## 注意・免責

研究用途限定。薬機法上の医療機器ではなく、診断・治療の根拠に使わない。
出力結果について保証・責任を負わない。患者データの取り扱いは PenClaw のハードルール
（患者氏名・ID をリポジトリに置かない）に従う。

## 関連

- 下流: **blender-dental**（STL → 中空オープン模型 3D プリント）。
- 研究: PointNet2 STL セグメンテーション研究（コード＋ケン）。CBCT 由来 STL の入力源。
