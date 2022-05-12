# SNU_torch
Spiking Neural Unit --torch ver

# 必要ライブラリ
使用するライブラリは'conda_emv.yml'に記載してあります。そのファイルから次のコマンドで環境を作ってください。
```bash
conda create -n 環境名 -f conda_env.yml
```


# 使い方
### クローン
保存したいディレクトリまで移動して、コマンドプロンプトで次のようにしてクローン。
```bash
git clone https://github.com/oshima-yoppi/SNU-torch.git
```
### データセット
[blender](https://github.com/oshima-yoppi/Blender)で物体が回転する動画を作成。そこで作った動画を[v2e](https://github.com/oshima-yoppi/v2e)でイベントカメラバージョンに変換させる。

### 学習
'train.py'を実行することで、イベントカメラデータを読み込んで、学習を行う。


# 注意
特になし


# Author


* oshima-yoppi
* kimura-lab
