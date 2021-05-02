# conv_high_quality_movies
# Overview
対象の動画を高画質に変換する。

# ディレクトリ
- input
    - movies
        - 高画質化したい動画を置く
    - images
        - 連番画像を置く

# how to use

```
# 動画のfpsを確認
source mo2im.sh input/sample.wmv

# 動画を連番画像に変換
source mo2im.sh input/sample.wmv 30

# settings.pyを書き換える

# pix2pix
python3 train.py

# 学習済モデルを使用して高画質に変換
python3 inference.py

# 連番画像を動画化
source im2mo.sh sample 30
```

# メモ
学習には適当な動画を縮小→拡大した画像を使用  
メモリ8GBだとバッチサイズを大きく出来ない…  
手元で試すもそこまで高画質にはならなかった…


