import os
from WindPy import *
w.start()

definitionDir = "definition"
mktDataDir  = "mkdata\\CZCE"

class TickSize:
    CF = 5
    SR = 1
    MA = 1
    TA = 2

class Multiplier:
    CF = 5
    SR = 10
    MA = 10
    TA = 5

class CZCEProduct(enum.Enum):
    CF = 1
    SR = 2
    MA = 3
    TA = 4

class LitePriceColName:
    timestamp = 'timestamp'
    srcip = 'srcip'
    symbol = 'symbol'

    bidqty = 'bidqty'
    bidprice = 'bidprice'
    askqty = 'askqty'
    askprice = 'askprice'

    bidprice1 = 'bidprice1'
    bidqty1 = 'bidqty1'
    bidprice2 = 'bidprice2'
    bidqty2 = 'bidqty2'
    bidprice3 = 'bidprice3'
    bidqty3 = 'bidqty3'
    bidprice4 = 'bidprice4'
    bidqty4 = 'bidqty4'
    bidprice5 = 'bidprice5'
    bidqty5 = 'bidqty5'
    askprice1 = 'askprice1'
    askqty1 = 'askqty1'
    askprice2 = 'askprice2'
    askqty2 = 'askqty2'
    askprice3 = 'askprice3'
    askqty3 = 'askqty3'
    askprice4 = 'askprice4'
    askqty4 = 'askqty4'
    askprice5 = 'askprice5'
    askqty5 = 'askqty5'
    lastprice = 'lastprice'
    totalvolume = 'totalvolume'
    vol = 'volume'
    openinterest = 'openinterest'

    bary = 'bary'
    mid  = 'mid'
    barymid = 'barymid'

    spottheo = 'spottheo'
    unmainbary = 'unmainbary'




class ExchTime:
    dayopentime  = datetime.time(9,0,0)
    dayclosetime = datetime.time(15,0,0)
    daystarttime = datetime.time(9,3,0)
    dayendtime   = datetime.time(14,57,0)
    nightopentime  = datetime.time(21,0,0)
    nightstarttime = datetime.time(21,3,0)

class Utility:
    def getTradeDates(self, startdatestr, enddatestr):
        try:
            tradesDates = pd.read_csv(dateListDir,index_col= 0, header = None).index.tolist()
            _tdaysdt    = pd.to_datetime(tradesDates)
            _latest = max(_tdaysdt)
            _earliest= min(_tdaysdt)
            if (pd.to_datetime(startdatestr) >= _earliest) & (pd.to_datetime(enddatestr) <= _latest):
                _tdaysdt = _tdaysdt[(_tdaysdt >= _earliest)&(_tdaysdt <= _lastet)]
            else:
                _tdaysdt = w.tdays(startdatestr, enddatestr).Data[0]
        except ValueError:
            assert 'TradeFILE NOT FOUND'
        else:
            _tdaysdt = w.tdays(startdatestr, enddatestr).Data[0]
        tradeDatesStr = [date.strftime("%Y%M%d") for date in _tdaysdt]

        retrun tradeDatesStr

    def getMktData(self, datestr,folderdir = mktDataDir):
        files = os.listdir(folderdir)
        files = [file for file in files if '.txt' in file]
        md = None
        filename = folderdir + datestr + ".txt"
        if filename in files:
            md  = pd.read_csv(filename,index_col= None, header = 0)
        else:
            Print("No MktData for ", datestr)
        return md

    def getReadInFile(self, product, folderdir = definitionDir):
        filename = folderdir + "\\" + "ReadInFile_all_" + product.upper() + ".csv"
        return pd.read_csv(filename, index_col = None, header = 0)

    def getFinalTradeList(self, datelist, readinfile):
        datelist_dt = pd.to_datetime(datelist)
        readinfile_dt = pd.to_datetime(readinfile['date'])
        crossdates = list(set(datelist_dt).intersection(readinfile_dt))
        return [date.strftime("%Y%M%d") for date in crossdates]

    def getInst(self, datestr, readinfile):
        maininst   =  readinfile.loc[readinfile['date']== datestr, 'maininst']
        unmaininst =  readinfile.loc[readinfile['date'] == datestr, 'unmaininst']
        return maininst, unmaininst

    def getPriceTick(self, product:CZCEProduct):
        if product == CZCEProduct.CF:
            pricetick = TickSize.CF
        elif product == CZCEProduct.SR:
            pricetick = TickSize.SR
        elif product == CZCEProduct.MA:
            pricetick = TickSize.MA
        elif product == CZCEProduct.TA:
            pricetick = TickSize.TA
        else:
            pricetick = None
        return pricetick

    def getMultiplier(self, product:CZCEProduct):
        if product == CZCEProduct.CF:
            multi = Multiplier.CF
        elif product == CZCEProduct.SR:
            multi = Multiplier.SR
        elif product == CZCEProduct.MA:
            multi = Multiplier.MA
        elif product == CZCEProduct.TA:
            multi = Multiplier.TA
        else:
            multi= None
        return multi








