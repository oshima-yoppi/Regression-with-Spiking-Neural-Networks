# SNU_torch
Spiking Neural Unit --torch ver

# 必要ライブラリ
使用するライブラリは`conda_emv.yml`に記載してあります。そのファイルから次のコマンドで環境を作ってください。
```bash
conda create -n 環境名 -f conda_env.yml
```


# 使い方
## クローン
保存したいディレクトリまで移動して、コマンドプロンプトで次のようにしてクローン。
```bash
git clone https://github.com/oshima-yoppi/SNU-torch.git
```
## データセット
[Blender](https://github.com/oshima-yoppi/Blender)で物体が回転する動画を作成。そこで作った動画を[v2e](https://github.com/oshima-yoppi/v2e)でイベントカメラバージョンに変換させる。

### blenderの使い方
[Blender](https://github.com/oshima-yoppi/Blender)のサイトに書いてあります。

### v2eの使い方
[v2e](https://github.com/oshima-yoppi/v2e)の使い方を軽く説明します。  
v2eはかなり多くのargumentがありますが、[v2e](https://github.com/oshima-yoppi/v2e)の`README.md`に詳しく説明されてあります。そこの英語を読んでがんばってください＾＾。  
`cmd.py`を実行することによって、学習用の入力データと正解データを作成して保存します。  
入力データはblenderで作成した動画をイベントカメラバージョンに変換させたときのデータになります。(t, p, x, y)のデータ型で保存される。（時間、極性、ｘ座標、ｙ座標）  
正解データはblenderで作成した動画のファイル名から読み込んでいます。
入力データと正解データはtorch.tensor型でoutputフォルダ内に`.h5`ファイルに保存されます。
## 学習
`train.py`を実行することで、イベントカメラデータを読み込んで、学習を行う。


# 注意
特になし


# Author


* oshima-yoppi
* kimura-lab
