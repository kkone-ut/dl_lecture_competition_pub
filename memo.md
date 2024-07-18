# フォルダ構造

- configs
    - config.yaml:
    - config_*.yaml: main.pyで指定する

- data
    - train: 65728
        - train_X.pt: [65728, 271, 281] (データ数, MEGチャネル数, シーケンス長)
        - train_subject_idxs.pt: [65728] それぞれのデータがどの被験者に対応しているか (0~3の値, 4*16432=65728)
        - train_y.pt: [65728] それぞれのデータがどの画像を見た時のものか (0~1853の値, 1854*32=65728)
        - train_image_path.txt: trainに使われる画像のパス
    - val: 16432
    - test: 16432

- dlbasics: 仮想環境
    - 起動コマンド: source dlbasics/bin/activate

- images: 各クラス12枚程度の画像 (1854クラス*12=22448)
    - {class}/{class}_{idx}{アルファベット}.jpg

- notebooks
    - data_check.ipynb: データセットの確認

- outputs: 実行結果
    - {YYYY-MM-DD}/{HH-MM-SS: UTC+0基準(9時間前になっていることに注意する)}
        - wandb
        - model_best.pt: ベスト重み
        - model_last.pt: 最終重み
        - submission.npy: テスト入力に対する予測

- src
    - datasets.py
        - データセットの定義。基本的には変更しない
    - models.py
        - モデルの定義。変更する
    - utils.py
        - ランダムシードの設定

- eval.py
    - 実行コマンド: python eval.py model_path={評価したい重みのパス}.pt

- main.py: 訓練
    - 実行コマンド: python main.py use_wandb=True
    - 実行の流れ
        1. config の読み込み
        2. データセット
        3. モデル
        4. Optimizer
        5. Training
        6. ベストモデルでの評価
        7. 予測結果の保存


# メモ
- ベースラインのtest accuracy=1.637%
- 評価はtop-10 accuracy (モデルの予測確率トップ10に正解クラスが含まれているかどうか) で行う
  - chance level = 10 / 1,854 ≒ 0.54%
