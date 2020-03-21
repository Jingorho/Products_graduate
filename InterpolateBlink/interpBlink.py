###############################
# ライブラリの読み込みとJupyter Notebookの設定
###############################
import pandas as pd # csvファイルとかを扱うためのライブラリ
import numpy as np # 数値計算を扱うためのライブラリ
import matplotlib.pyplot as plt # グラフ表示するためのライブラリ
import os # ファイルとディレクトリ処理のためのライブラリ
import glob # ファイル検索のライブラリ
import sys # コマンドライン引数のためのライブラリ
# %matplotlib inline
# Jupyter Notebook上にグラフを表示させるコマンド



###############################
# メイン関数 JupyterNotebookを実行するとこの関数が実行される
###############################
def main():

    # 平滑化処理をするときの移動平均の幅.
    # ここの値を変えると平均をとる幅が変わる
    windowWidth = 100

    # [A] (デバッグ用) 特定のcsvファイル一つだけを実行する場合はこちら
#     SpecificCsvFile = "視線データ/1211_1_1/bd/bd5.csv"
#     MakeInterpolatedCsv(SpecificCsvFile, windowWidth)

    # [B] このスクリプトが置かれた場所からファイルを探索して信号処理を行う
    ProcessDir(windowWidth)


###############################
# 補間と平滑化のメイン処理
###############################
def interpCsv(fi, _windowWidth):

    print("\n- - - Start interpCsv() - - -")

    ###############################
    # データ読み込みと前処理
    ###############################
    colnames = ["left", "right", "leftValid", "rightValid"] # 列名を作成
    df = pd.read_csv(fi, header = None, names = colnames) # 作成した列名でcsv読み込み
    # header=Noneにしてるのに1行目に名前が入っちゃうとき用
    if type(df.iloc[0,0]) is str:
        df = df.drop([0]) # "Pupil duration right"とかの行を消す


    print(df.head()) # 最初の5行表示してデータの様子をチラ見
    print("NAの行数:\n" + str(len(df) - df.count())) # データの様子をチラ見

    # もともとNaNだろうが値あろうが、validationに基づいて無効データをNaNで上書き
    # (値があってもvalidationが4の場合もあるかも?のため)
    df.left[df.leftValid == 4] = pd.np.nan
    df.right[df.rightValid == 4] = pd.np.nan

#     # 数値データでないノイズとかがまざってるときあるのでNaNで置き換え
#     # 左
#     left_isStr = [isinstance(df.left[i], str) for i in range(0, len(df.left), 1)]
#     df.left = np.where(left_isStr, np.nan, df.left)
#     print(str(sum(left_isStr)) + " string values are converted to NaN")
#     # 右
#     right_isStr = [isinstance(df.right[i], str) for i in range(0, len(df.right), 1)]
#     df.right = np.where(right_isStr, np.nan, df.right)
#     print(str(sum(right_isStr)) + " string values are converted to NaN")
#     print(df.head())


    ###############################
    # 線形補間処理
    ###############################
    # 左
    y_left = df.left.interpolate() # 線形補間したyの値
#     plt.plot(x, y_left) # 線形補間 後のグラフ
#     plt.plot(x, df.left) # 線形補間 前のグラフ
#     plt.show() # グラフ表示
    # 右
    y_right = df.right.interpolate() # 線形補間したyの値
#     plt.plot(x, y_right) # 線形補間 後のグラフ
#     plt.plot(x, df.right) # 線形補間 前のグラフ
#     plt.show() # グラフ表示

    # (デバッグ用) 欠損値あった部分だけ拡大してグラフ表示
    # NaNId = np.where(df["leftValid"] == 4)[0] # タプルの0要素目(0列目)
    # NaNRanges = pd.Series(range(NaNId[0] - 100, NaNId[0] + 100, 1)) # NaN周辺
    # y_NaNRanges = pd.Series(df.left.interpolate().iloc[NaNRanges])
    # plt.plot(NaNRanges, y_NaNRanges)
    # plt.show()


    ###############################
    # 平滑化処理 - とりあえず移動平均
    ###############################
    # 左
    y_left_rol = y_left.rolling(window = _windowWidth, min_periods=1).mean() # 移動平均計算. 簡単すぎて笑うわ
    # 右
    y_right_rol = y_right.rolling(window = _windowWidth, min_periods=1).mean() # 移動平均計算. 簡単すぎて笑うわ


    # 移動平均で窓関数をあててNaNになった最初の方の行や、
    # 元からのNaNをデータ列の平均値で埋める
    print(str(y_left_rol.isnull().sum()) + " NaN values <- mean " + str(np.nanmean(y_left_rol)))
    print(str(y_right_rol.isnull().sum()) + " NaN values <- mean " + str(np.nanmean(y_right_rol)))
    y_left_rol = y_left_rol.fillna(np.nanmean(y_left_rol))
    y_right_rol = y_right_rol.fillna(np.nanmean(y_right_rol))


    ###############################
    # グラフ表示
    ###############################
    plt.figure(figsize=(15, 5)) # グラフの画像のサイズと解像度

    x = pd.Series(range(0, len(df.left), 1)) # グラフ表示用のx軸の値
    lw = 0.4 # グラフの線の太さ


    plt.plot(x, y_left, linewidth = lw, color = '#6ec16b', label='After Interpolate')   # 線形補間 後のグラフ
    plt.plot(x, df.left, linewidth = lw, color = '#ff9933', label='Before Interpolate') # 線形補間 前のグラフ
    plt.plot(x, y_left_rol, linewidth = lw, color = '#69a3d2', label='After Moving Average') # 移動平均後のグラフ
    plt.legend() # 凡例を表示する
    plt.title('') # タイトルここに入力
    plt.xlabel('Time Series index') # x軸のラベル
    plt.ylabel('Pupil size') # y軸のラベル

    # plt.show() # グラフ表示
    # 画像保存用のフォルダがなければ作る
    if not os.path.isdir('data/img'):
        os.mkdir('data/img') # ディレクトリ作る
    pngName = fi.strip(".csv")
    pngName = pngName.strip("data/")
    pngName = pngName.replace('/', '_')
    print("  Saved " + pngName + ".png")
    plt.savefig("data/img/" + pngName + '.png')
    plt.clf() # グラフクリア

#     plt.plot(x, y_right) # 線形補間 後のグラフ
#     plt.plot(x, df.right) # 線形補間 前のグラフ
#     plt.plot(x, y_right_rol) # 移動平均後のグラフ
#     plt.show() # グラフ表示


    ###############################
    # 処理後の出力用データフレーム作成
    ###############################
    outputDataFrame = pd.concat([y_left_rol, y_right_rol], axis = 1)
    print("\nOutput : \n")
    print(outputDataFrame.head())

    print("- - - End interpCsv() - - -\n")


    return outputDataFrame




###############################
# ディレクトリ直下のcsvファイルに対して線形補間処理を提示する関数
###############################
def ProcessDir(_windowWidth):
    # 取得したディレクトリの中に存在してるcsvファイルの数分だけパスを取得
    dataRootPath = "data/"
    fileList = glob.glob(dataRootPath + '[0-9].csv') # 数値部分は正規表現
    # 取得したディレクトリ内に"1.csv"とかいう名前のファイルが一つでもあれば
    if(fileList is not None):
        # "1.csv", "2.csv", ...とファイルの数分だけループを回す
        for file in fileList:
            # 線形補間の処理へ
            MakeInterpolatedCsv(file, _windowWidth)



###############################
# 指定したファイルで線形補間と平滑化処理
###############################
def MakeInterpolatedCsv(_file, _windowWidth):

    print("------------------------------------------")
    print("\n\n> Reading " + _file + " ...")

    # 線形補間の処理
    outputDf = interpCsv(_file, _windowWidth)

    dataRootPath = "data/"
    fileId = _file.strip(dataRootPath) # 番号("1.csv"のうち"1")だけ取り出し
    fileId = fileId.strip(".csv") # 番号("1.csv"のうち"1")だけ取り出し
    
    # "processed"というディレクトリがなければ作るし、あれば何もしない
    if not os.path.isdir(dataRootPath + 'processed'):
        os.mkdir(dataRootPath + 'processed') # ディレクトリ作る
        print(" >> Made : " + dataRootPath + 'processed')
    else:
        print(" >> Already Exist : " + dataRootPath + "processed")

    # 出力用のパス作成
    outputCsvPath = dataRootPath + 'processed/' + fileId + '_pd.csv'
    # csv出力
    outputDf.to_csv(outputCsvPath, index = None)

    print("\n> Writing " + outputCsvPath + " ... \n\n")




###############################
# main関数を実行する処理
###############################
if __name__ == '__main__':
  main()


