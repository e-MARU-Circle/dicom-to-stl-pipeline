# 環境構築プロンプト（各自で実行）

> このスキルは **環境（venv）を同梱しません**。各自のマシンで一度だけ構築します。
> 以下はそのまま**セットアップ担当エージェント（または利用者本人）**に渡せる指示プロンプトです。
> PenClaw 環境ではコード（penclaw-ml）／ハブ（penclaw-hub）が担当。**外部環境ではそのエージェントは
> 不要**で、シェルを実行できる任意のエージェント／手動でそのまま使えます。
> Python 3.10〜3.13 を想定。構築後に `python3 pipeline/run_pipeline.py --check --model-dir <DIR>` で点検。

---

## 共通の前提

- Python と pip が使えること（`python3 --version`）。
- `pipeline/requirements.txt` に後段・匿名化の共通依存（SimpleITK / scikit-image / vtk / numpy / pydicom）。
- **torch と nnunetv2 はプラットフォーム依存**のため下記で個別に入れます。
- DICOM→NIfTI 変換に **dcm2niix**（pip 不可・別途）。

---

## ▼ macOS（Apple Silicon: MPS / CPU）— 担当エージェントへのプロンプト

```
あなたはこのスキルのセットアップ担当です（PenClawではコード=penclaw-ml。無い環境では任意のエージェント/利用者）。macOS (Apple Silicon) に dicom-to-stl-pipeline の
実行環境を構築してください。手順:

1. 作業ディレクトリで venv を作成し有効化:
     python3 -m venv venv && source venv/bin/activate
2. pip を更新:
     pip install -U pip
3. 共通依存をインストール:
     pip install -r pipeline/requirements.txt
4. PyTorch（MPS/CPU 対応の標準ホイール。CUDA は不要）:
     pip install torch torchvision
5. nnU-Netv2:
     pip install nnunetv2
6. dcm2niix（Homebrew）:
     brew install dcm2niix
7. 動作確認:
     python3 -c "import torch; print('mps', torch.backends.mps.is_available())"
     dcm2niix --version
     python3 pipeline/run_pipeline.py -h

注意:
- 値や患者情報は出力しないこと。コマンドの成否（バージョン文字列・終了コード）のみ報告。
- MPS で nnU-Net がエラーになる場合は --device cpu で実行する旨を伝える。
```

推論時間の目安: CPU で数十分、MPS で短縮（モデル・症例による）。

---

## ▼ Windows（CUDA / NVIDIA GPU）— 担当エージェントへのプロンプト

```
あなたはこのスキルのセットアップ担当です（PenClawではコード=penclaw-ml。無い環境では任意のエージェント/利用者）。Windows + NVIDIA GPU に dicom-to-stl-pipeline の
CUDA 実行環境を構築してください。前提: NVIDIA ドライバ導入済み。手順:

1. venv を作成し有効化（PowerShell）:
     python -m venv venv ; .\venv\Scripts\Activate.ps1
2. pip 更新:
     pip install -U pip
3. 共通依存:
     pip install -r pipeline\requirements.txt
4. PyTorch（CUDA 12.1 ホイールの例。環境に合う cuXXX を選ぶ）:
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
5. nnU-Netv2:
     pip install nnunetv2
6. dcm2niix:
     https://github.com/rordenlab/dcm2niix/releases から dcm2niix.exe を取得し
     PATH の通った場所に置く（または run_pipeline.py の --dcm2niix で明示）。
7. 動作確認:
     python -c "import torch; print('cuda', torch.cuda.is_available())"
     dcm2niix --version
     python pipeline\run_pipeline.py -h

注意:
- torch.cuda.is_available() が False の場合はドライバ/CUDA ホイールの不一致。
  cuXXX のバージョンを見直す。
- 患者情報は出力しないこと。成否のみ報告。
```

推論時間の目安: CUDA GPU で 5〜10 分程度（GPU・症例による）。

---

## ▼ 汎用 CPU のみ（GPU 非依存・最小）

上記 macOS 手順の 4 を `pip install torch torchvision`（CPU ホイール）に置き換えるだけ。
`--device cpu` で実行。どのマシンでも動くが推論は遅い。

---

## トラブルシュート早見

| 症状 | 対処 |
|---|---|
| `dcm2niix が見つかりません` | PATH を確認 or `--dcm2niix /full/path/dcm2niix` |
| `学習モデルが見つかりません` | `references/model_acquisition.md` でモデル配置を確認 |
| MPS でクラッシュ | `--device cpu` |
| `torch.cuda.is_available()=False` | CUDA ホイール（cuXXX）とドライバの整合を確認 |
| NIfTI が 1 つでないエラー | DICOM フォルダに複数シリーズ混在。1 症例 1 シリーズに分ける |
