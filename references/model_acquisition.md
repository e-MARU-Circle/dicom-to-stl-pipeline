# 学習モデル取得・配置プロンプト（各自で実行）

> **重要: 学習モデル（重み）は本スキルに同梱せず、再配布もしません。**
> 各自が一次配布元から直接ダウンロードして、自分のマシンに配置します。
> 以下はそのまま担当エージェント（コード＝penclaw-ml / ハブ＝penclaw-hub）に渡せる指示プロンプトです。

---

## 推奨モデル: DentalSegmentator（公開・CC-BY-4.0）

本パイプラインのラベル定義（1上顎/2下顎骨/3上顎歯/4下顎歯/5下顎管）は、公開モデル
**DentalSegmentator** と一致します。重みはこちらから取得します。

- ダウンロード元（一次配布）: https://zenodo.org/records/10829675
- ファイル: `Dataset112_DentalSegmentator_v100.zip`（約 229.7 MB / md5 `b71cd5230168d28a4f71b078265b76be`）
- ネットワーク: nnU-Net v2.2 / 3d_fullres / 5 クラス（CBCT・CT 対応）
- ライセンス: **Creative Commons Attribution 4.0（CC-BY-4.0）**
- 出典・引用（必須）:
  > Dot G, et al. *DentalSegmentator: robust open source deep learning-based CT and CBCT image
  > segmentation.* Journal of Dentistry (2024). doi:10.1016/j.jdent.2024.105130
  > Isensee F, et al. *nnU-Net.* Nat Methods 2021;18(2):203-211. doi:10.1038/s41592-020-01008-z

> 注意: CC-BY-4.0 は再配布を許容しますが、**PenClaw の方針として本スキルには同梱・再配布しません。**
> 各自がリンクから直接ダウンロードし、表示義務（上記引用）を守ってください。

---

## 必要なフォルダ構造（nnU-Netv2 results 形式）

`run_pipeline.py --model-dir <DIR>` が指す `<DIR>` の下が、nnU-Netv2 の results 構造になっている必要があります。

```
<model-dir>/                              ← --model-dir で指定（nnUNet_results 相当）
└── Dataset112_DentalSegmentator/        ← --dataset（既定。解凍後の実フォルダ名で確認）
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/   ← --configuration（既定 3d_fullres）
        ├── dataset.json
        ├── plans.json
        └── fold_0/                       ← --fold（既定 0）
            └── checkpoint_final.pth
```

---

## ▼ 取得・配置プロンプト（DentalSegmentator）

```
あなたはこのスキルのモデル配置担当です（PenClawではコード=penclaw-ml。無い環境では任意のエージェント/利用者）。dicom-to-stl-pipeline 用の学習済みモデル
DentalSegmentator を配置してください。再配布・外部アップロードは禁止。手順:

1. 一次配布元から zip を取得（ブラウザ可。CLI 例）:
     curl -L -o Dataset112_DentalSegmentator_v100.zip \
       "https://zenodo.org/records/10829675/files/Dataset112_DentalSegmentator_v100.zip?download=1"
2. md5 を照合（任意・推奨）: 期待値 b71cd5230168d28a4f71b078265b76be
3. 解凍し、出てきた DatasetXXX_ フォルダ名を確認（例: Dataset112_DentalSegmentator）:
     unzip -q Dataset112_DentalSegmentator_v100.zip -d ./nnUNet_results
     ls ./nnUNet_results        # ← 実フォルダ名を確認
4. 構造確認（ファイルの存在のみ。中身は開かない）:
     ls ./nnUNet_results/<DatasetXXX_...>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
5. パイプラインに渡す:
     run_pipeline.py の --model-dir に ./nnUNet_results を指定。
     --dataset に手順3で確認した実フォルダ名を渡す（既定と違う場合）。
     環境変数 NNUNET_RESULTS_DIR=./nnUNet_results でも可（--model-dir 省略）。

注意:
- checkpoint_final.pth を Git にコミットしない／クラウド共有しない。
- 配置の成否（OK/見つからない）のみ報告し、ファイル内容は表示しない。
- CC-BY-4.0 の出典表示（Dot G et al. 2024 / nnU-Net）を成果物・院内資料に明記する。
```

実行例:
```bash
python3 pipeline/run_pipeline.py \
    --in ANON_CASE --out STL_OUT \
    --model-dir ./nnUNet_results \
    --dataset Dataset112_DentalSegmentator \
    --device auto --require-anonymized --accept-disclaimer
```

---

## 自前モデル（Dataset111_453CT 等）を使う場合

先生の自前学習モデルを使う場合は、同じ results 構造に置き、`--dataset Dataset111_453CT` で上書き。
新規学習は nnU-Netv2 標準フロー（コード＝penclaw-ml 担当）:

```
nnUNetv2_plan_and_preprocess -d <ID> --verify_dataset_integrity
nnUNetv2_train <ID> 3d_fullres 0
```

**学習に使う患者データの取り扱いは PenClaw のハードルール（患者氏名・ID をリポジトリに置かない）に従う。**

---

## .gitignore 推奨

モデルや患者データを誤ってコミットしないため、利用側リポジトリに以下を追加:

```
nnUNet_results/
*.pth
*.zip
*_DICOM/
*_ANON/
*.nii.gz
*.stl
```
