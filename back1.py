import pandas as pd

def getBackData(baseData, fromYear, endYear, etf50CloseData):
    backData = {}
    for year in range(fromYear, endYear + 1):
        backData[year] = {}
        currentYearData = baseData.loc[baseData.name.str.contains(str(year) + "年")]
        for month in range(1, 4):
            backData[year][month] = {}
            currentMonthData = currentYearData.loc[currentYearData.name.str.contains(str(month) + "月")]
            close = etf50CloseData.loc[(etf50CloseData.year == 2015) & (etf50CloseData.month == 3)].close[0]
            backData[year][month]['认购'] = currentMonthData.loc[currentMonthData.name.str.contains("认购") & (currentMonthData.price == (close + 0.1))]
            backData[year][month]['认沽'] = currentMonthData.loc[currentMonthData.name.str.contains("认沽") & (currentMonthData.price == (close - 0.1))]

    return backData


def main():
    df_basic = pd.read_csv("data.csv")
    df_basic = df_basic.loc[df_basic.name.str.contains("50ETF")]
    fromYear = 2015
    endYear = 2015
    etf50CloseData = pd.read_csv("data1.csv")
    print(getBackData(df_basic, fromYear, endYear, etf50CloseData))


if __name__ == "__main__":
    main()
