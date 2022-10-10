# Regression with Spiking Neural Network
SNNを用いて回帰を行うネットワーク。Spiking Neural Unitをpytorchベースで書いています。  
イベントカメラから出力されるパルスを直接SNNで処理し、回帰タスクを実現。

# 必要ライブラリ
使用するライブラリは`conda_emv.yml`に記載してあります。そのファイルから次のコマンドで環境を作ってください。
```bash
conda create -n 環境名 -f conda_env.yml
```


# 使い方
## クローン
保存したいディレクトリまで移動して、コマンドプロンプトで次のようにしてクローン。
```bash
git clone https://github.com/oshima-yoppi/Regression-with-Spiking-Neural-Networks.git
```
## データセット
[Blender](https://github.com/oshima-yoppi/Blender)で物体が回転する動画を作成。そこで作った動画を[v2e](https://github.com/oshima-yoppi/v2e)でイベントカメラバージョンに変換させることによってデータセットを作ります。

### blenderの使い方
[Blender](https://github.com/oshima-yoppi/Blender)のサイトに書いてあります。

### v2eの使い方
[v2e](https://github.com/oshima-yoppi/v2e)の使い方を軽く説明します。  
v2eはかなり多くのargumentがありますが、[v2e](https://github.com/oshima-yoppi/v2e)の`README.md`に詳しく説明されてあります。そこの英語を読んでがんばってください＾＾  
`cmd.py`を実行することによって、学習用の入力データと正解データを作成して保存します。  
入力データはblenderで作成した動画をイベントカメラバージョンに変換させたときのデータになります。(t, p, x, y)のデータ型で保存される。（時間、極性、ｘ座標、ｙ座標）  
正解データはblenderで作成した動画のファイル名から読み込んでいます。
入力データと正解データはtorch.tensor型でoutputフォルダ内に`.h5`ファイルに保存されます。  
v2eにより次のように動画をイベントデータに変換できます。  
![11_-82 15004684584068_0_0_](https://user-images.githubusercontent.com/82073759/171681741-ba964a32-5cfd-43fe-bb4c-d69e6c965247.gif)
![dvs-video (1)](https://user-images.githubusercontent.com/82073759/171682207-dc1ed076-8993-4d68-8a97-b6469c0b08bd.gif)  
![RGB](https://github.com/oshima-yoppi/SNU_torch/blob/bf0eeb282c6cc7c25fda309d9154aa09b9ee3c4b/youtube/douga_.gif)


# ネットワーク構造  
- `model/network.py`にネットワーク構造が記述されています。  
- `model/snu_layer.py`にSNUの構造が記述されています。  
- `model/step_func.py`に誤差逆伝搬の代理勾配の構造が定義されています。
# 学習
`train3verctor.py`を実行することで、イベントカメラデータを読み込んで、学習を行う。

# 学習結果のデータ分析 
`/analysis.py`の"save_loss"という関数を実行することでpandasを用いてcsvに保存する。  
そこで生成されたcsvを`/analysis/analysis.py`によってグラフ化する。かなり便利。  
`/analysis_read.py`とかはすごく使いづらいから使わないほうがいいと思う。


# 注意
特になし


