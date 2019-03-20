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
    windowWidth = 50

    # [A] (デバッグ用) 特定のcsvファイル一つだけを実行する場合はこちら
#     SpecificCsvFile = "視線データ/1211_1_1/bd/bd5.csv"
#     MakeInterpolatedCsv(SpecificCsvFile, windowWidth)

    # [B] このスクリプトが置かれた場所からファイルを探索して信号処理を行う
    LoopProcessDir(windowWidth)


###############################
# 補間と平滑化のメイン処理
###############################
def interpCsv(fi, _windowWidth):

    print("\n  - - - Start interpCsv() - - -")

    ###############################
    # データ読み込みと前処理
    ###############################
    colnames = ["left", "right", "leftValid", "rightValid"] # 列名を作成
    df = pd.read_csv(fi, header = None, names = colnames) # 作成した列名でcsv読み込み
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
    x = pd.Series(range(0, len(df.left), 1)) # グラフ表示用のx軸の値

    plt.plot(x, y_left, linewidth = 1.0, label='After Interpolate')   # 線形補間 後のグラフ
    plt.plot(x, df.left, linewidth = 1.0, label='Before Interpolate') # 線形補間 前のグラフ
    plt.plot(x, y_left_rol, linewidth = 1.0, label='After Moving Average') # 移動平均後のグラフ
    plt.legend() # 凡例を表示する
    plt.title('') # タイトルここに入力
    plt.xlabel('Time Series index') # x軸のラベル
    plt.ylabel('Pupil size') # y軸のラベル

    # plt.show() # グラフ表示
    # 画像保存用のフォルダがなければ作る
    if not os.path.isdir('視線データ/img'):
        os.mkdir('視線データ/img') # ディレクトリ作る
    pngName = fi.strip(".csv")
    pngName = pngName.strip("視線データ/")
    pngName = pngName.replace('/', '_')
    print("  Saved " + pngName + ".png")
    plt.savefig("視線データ/img/" + pngName + '.png')
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

    print("  - - - End interpCsv() - - -\n")


    return outputDataFrame






###############################
# ディレクトリで処理を回す関数
###############################
def LoopProcessDir(_windowWidth):
    dataRootPath = "視線データ/"
    dirList = os.listdir(dataRootPath) # dataRootPath内に存在するディレクトリ一覧取得

    edMatTypes = ["bd", "nw", "an"]

    arg = sys.argv
    if len(arg) == 2:
        if arg[1] == "an":
            edMatTypes = ["an"]
            print("###########################################")
            print("#    EXECUTE ONLY an FILES !!!!!!!!!!!    #")
            print("###########################################")
    else:
        print("EXECUTE bd, nw, an FILES")

    print( "Length of Directries : " + str(len(dirList)) )

    # 取得したディレクトリ一覧の数分だけループを回す
    for fileNum in range(1, len(dirList), 1):
        # "bd"か"nw"かでループを回す
        for edMatType in edMatTypes:
            # 取得したディレクトリの中に存在してるcsvファイルの数分だけパスを取得
            path = str(dataRootPath + dirList[fileNum]) + "/" + edMatType + "/"

            if edMatType == "an":
            	fileList = glob.glob(path + edMatType + '.csv')
            else:
            	fileList = glob.glob(path + edMatType + '[0-9].csv') # 数値部分は正規表現

            # 取得したディレクトリ内に"bd1.csv"とかいう名前のファイルが一つでもあれば
            if(fileList is not None):
                # "bd1.csv", "bd2.csv", ...とファイルの数分だけループを回す
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

    # 無理やり特急工事...
    filePath = ""
    fileId = "0"
    fileEdMatType = ""

    if _file.strip(".csv")[-2:] == "an":
        filePath = _file.strip("an.csv") # パスのうち"an/an.csv"を除いたパス
    else:
        fileId = _file.strip(".csv")[-1] # 番号("bd1.csv"のうち"1")だけ取り出し
        fileEdMatType = _file.strip(fileId + ".csv")[-2:] # "bd1.csv"のうち"bd"
        filePath = _file.strip(fileEdMatType + fileId + ".csv") # パスのうち"bd1.csv"を除いたパス

    # "補間済"というディレクトリがなければ作るし、あれば何もしない
    if not os.path.isdir(filePath + '補間済'):
        os.mkdir(filePath + '補間済') # ディレクトリ作る
        print(" >> Made : " + filePath + '補間済')
    else:
        print(" >> Already Exist : " + filePath + "補間済")

    # 出力用のパス作成
    if _file.strip(".csv")[-2:] == "an":
    	outputCsvPath = filePath + '補間済/an_pd.csv'
    else:
    	outputCsvPath = filePath + '補間済/' + fileEdMatType + fileId + '_pd.csv'
    # csv出力
    outputDf.to_csv(outputCsvPath, index = None)

    print("> Writing " + outputCsvPath + " ... \n\n")




###############################
# main関数を実行する処理
###############################
if __name__ == '__main__':
  main()


