# パラメータ調整ガイド（元アプリの実測知見）

> 元の DICOM_to_STL アプリ開発で**最も苦労したのがパラメータ調整**だった。
> その結論を「既定値」と「切替フラグ」に落とし込んである。再調整で同じ轍を踏まないための記録。
> 他者の環境（GPU 有無・メモリ・OS）で動かす前提なので、迷ったら下表の通り。

## まず迷ったら（プリセット）

| 環境 | コマンド | 中身 |
|---|---|---|
| 通常（CPU/MPS、まず動かす） | `--preset fast`（既定） | TTA 無効 + step_size 0.7 |
| GPU(CUDA) で精度重視 | `--preset quality` | TTA 有効 + step_size 0.5 |
| 低メモリ／multiprocessing 制約 | `--preset low-resource` | 逐次(npp=nps=0) + TTA 無効 + step_size 0.7 |

個別フラグ（`--step-size` `--tta/--no-tta` `--sequential` `--npp` `--nps` `--threads`）はプリセットを上書きする。

---

## 推論パラメータ（ここが本番の苦労どころ）

### TTA（Test-Time Augmentation / 8方向ミラー予測）
- **効果と代償**: 精度はわずかに上がるが **計算量が約 8 倍**。CPU だと致命的に遅い。
- **結論**: 既定は **無効（`--disable_tta` 相当）**。GPU で精度を詰めたいときだけ `--tta`。
- 元アプリでは「60 分→数分」短縮の主因がこれ。

### step_size（スライディングウィンドウの重なり）
- 小さいほど重なりが増え高精度・低速。nnU-Net 既定 0.5。
- **結論**: 既定 **0.7**（重なりを減らし約 30% 高速化、精度差は実用上軽微）。精度を詰めるなら `--step-size 0.5`。

### 逐次実行（npp / nps = 0）
- nnU-Net は前処理・出力をマルチプロセスで回すが、**権限制約・低メモリ・PyInstaller 等で
  `multiprocessing.Manager` が落ちる**ことがある（元アプリの最大のハマりどころ）。
- **対処**: `--sequential`（= `-npp 0 -nps 0`）で 1 プロセス実行に倒す。遅いが確実に動く。
- 「アプリがステップ2で固まる/無限ループ」系は、ほぼこれで回避できた。

### CPU スレッド（--threads）
- CPU 推論時、既定スレッド数が環境で極端に振れる。`--threads <コア数>` で固定すると安定。

### デバイス（--device）
- `auto` は cuda > mps > cpu の順で自動選択。
- **MPS 注意**: Apple GPU で nnU-Net が落ちる例がある。失敗時は `--device cpu`。
- `perform_everything_on_device=True` は CUDA 専用。本パイプラインは堅牢側（False 相当・標準 CLI）に倒している。

---

## DICOM→NIfTI（dcm2niix）

- `-d 9`: サブフォルダ探索を 9 階層へ（既定 5 では深い構造で「DICOM が見つからない」エラー）。
- `-i n`: 派生画像も取り込む（除外で取りこぼす症例があった）。
- `-z y`: gzip 圧縮 NIfTI。
- **1 症例 = 1 シリーズ**。複数シリーズが混ざると NIfTI が複数生成されてエラーになる。フォルダを分ける。

---

## NIfTI→STL（メッシュ品質。ここも長く苦労した）

これらは `nifti_to_stl.py` に**確定値として実装済み**（再調整不要）。背景のみ記す。

### 平滑化アルゴリズム
- `trimesh` の laplacian / taubin / humphrey を渡り歩き、パラメータ（iterations・lamb・nu・alpha・beta）で
  形状が崩れる迷走があった。
- **結論**: **VTK `vtkWindowedSincPolyDataFilter`**（`NumberOfIterations=30`, `PassBand=0.01`,
  境界平滑化 ON、非多様体平滑化 ON）に収束。形状を保ったまま滑らかになる。

### メッシュの裏表（法線）
- STL の面が裏返る問題。`vtkPolyDataNormals`（`ConsistencyOn` / `SplittingOff` / 自動向き付け）に加え、
  **符号付き体積が負なら三角形の巻き順を反転**してから平滑化することで安定化。

### スケール／位置
- marching cubes に **spacing を (z, y, x) 順**で渡す（SimpleITK は (x, y, z)）。これを忘れると拡大縮小する。
- 頂点に DICOM の `direction` 行列と `origin` を適用し、**元の物理空間**に一致させる。ラベル間の位置関係も保たれる。

---

## トラブル別・効くフラグ早見

| 症状 | まず試す |
|---|---|
| ステップ2で固まる/落ちる | `--preset low-resource`（または `--sequential`） |
| とにかく遅い（CPU） | `--preset fast` ＋ `--threads <コア数>` |
| MPS でクラッシュ | `--device cpu` |
| 精度を上げたい（GPU） | `--preset quality`（または `--tta --step-size 0.5`） |
| DICOM が見つからない | フォルダ階層/シリーズを確認（`-d 9 -i n` は適用済み） |
| 形状が荒い/裏返る | 既に確定処理済み。モデル・入力解像度を疑う |
