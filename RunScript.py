from FutureMMPricing import *
from Utility import *

class params:
    multiQuoteSource = False ##运用几路行情， False表示一路，True表示多路

    product = CZCEProduct.TA

    #period related
    startdatestr = "20200701"
    enddatestr   = "20200716"

    #basis model related
    alpha = 0.02
    alphalow = 0.02
    alphahigh = 0.24
    lowend = 0.08
    highend = 0.24

    #backtest related
    window = 480
    tickcredit = 0.8
    elastic = 0.2

readinfile = Utility.getReadInFile(params.product)
datelist = Utility.getTradeDates(params.startdatestr, params.enddatestr)
datelist = Utility.getFinalTradeList(datelist, readinfile)
solver = FutureMMPricing(params)
solver.BackTestRun(datelist, readinfile)



df_adpt = pd.DataFrame()
df_prod = pd.DataFrame()
result = pd.DataFrame()
for date, spec in readinfile.iterrows():
    print(date)
    maininst = spec.maininst
    unmaininst = spec.unmaininst
    pricetick = pricetickdt[spec['product']]
    filename = "mkdata\\" + str(date) + ".txt"
    rawdata = readdata(filename, [unmaininst, maininst], pricetick)
    data = findspottheo(rawdata, unmaininst, maininst)
    data = calcbasis(data, pricetick, alpha, alphalow, alphahigh, lowend, highend, window, unmaininst, maininst)

    # df_forbt = data_unmain[['timestamp','bidprice1', 'askprice1', 'lastprice','totalvolume','vol','barymid','owntheo_prod','owntheo_adpt']]
    df_forbt = data.copy()
    # df_forbt = df_forbt.dropna(how='any')  ## 剔除掉 index < window的数据（snapbasis_zscore为nan）
    df_forbt.reset_index(drop=True, inplace=True)

    df_forbt_result = plbacktest_onlyhitter(df_forbt, tickcredit, pricetick, elastic, unmaininst)
    smpdf = smpcols(df_forbt_result)
    resultdate = daystats_onlyhitter(df_forbt_result)
    resultdate.name = date
    result = result.append(resultdate)

    outdataname = spec['product'] + "_backtestdata_" + str(date) + ".csv"
    df_forbt_result.to_csv(outdataname)

    buy_adpt, sell_adpt, buy_prod, sell_prod, plt = plotgraph(df_forbt_result, date, maininst, unmaininst)

    if len(readinfile) > 1:
        plt.close('all')
result.to_csv("result.csv")
stats = statsandhist(result)
stats.to_csv("stats.csv", encoding='gbk')