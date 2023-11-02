import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
from datetime import datetime


def im_surface(df):  # 波动率曲面作图
    df = df.set_index("date")
    df = df.stack().reset_index()
    df.columns = ["date", "colors", "price"]
    fig = px.line(df, x="date", y="price", color="colors", line_shape="spline")
    plot(fig)



def getBackData(baseData, fromYear, endYear, etf50CloseData):
    backData = pd.DataFrame()
    for year in range(fromYear, endYear + 1):
        currentYearData = baseData.loc[baseData.name.str.contains(str(year) + "年")]
        if currentYearData.empty:
                continue
        for month in range(1, 4):
            currentMonthData = currentYearData.loc[currentYearData.name.str.contains(str(month) + "月")]
            if currentMonthData.empty:
                continue
            close = etf50CloseData.loc[(etf50CloseData.year == 2015) & (etf50CloseData.month == 3)].close[0]
            subData = (currentMonthData.loc[currentMonthData.name.str.contains("认购") & (currentMonthData.price == (close + 0.1))]).copy()
            if not subData.empty:
                subData['date'] = str(year) + "{:0>2}".format(str(month))
            putData = (currentMonthData.loc[currentMonthData.name.str.contains("认沽") & (currentMonthData.price == (close - 0.1))]).copy()
            if not putData.empty:
                putData['date'] = str(year) + "{:0>2}".format(str(month))
            if backData.empty:
                backData = subData
            else:
                backData = pd.merge(backData, subData, how='outer')
            backData = pd.merge(backData, putData, how='outer')
    return backData


def main():
    df_basic = pd.read_csv("data.csv")
    df_basic = df_basic.loc[df_basic.name.str.contains("50ETF")]
    fromYear = 2015
    endYear = 2016
    etf50CloseData = pd.read_csv("data1.csv")
    data = getBackData(df_basic, fromYear, endYear, etf50CloseData)
    data = data.drop(["ts_code", "name"], axis=1)
    print(data)
    im_surface(data)


if __name__ == "__main__":
    main()
