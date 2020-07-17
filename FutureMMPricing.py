# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:31:58 2020

@author: HP
"""
import numpy as np
import pandas as pd
import datetime 

import time

import matplotlib.pyplot as plt
from Utility import *

class FutureMMPricing:
    def __init__(self, params):

        self.mutiQuoteSource = params.multiQuoteSource
        self.product = params.product

        # period related
        self.startdatestr = params.startdatestr
        self.enddatestr = params.enddatestr

        # basis model related
        self.alpha = params.alpha
        self.alphalow = params.alphalow
        self.alphahigh = params.alphahigh
        self.lowend = params.lowend
        self.highend = params.highend

        # backtest related
        self.window = params.window
        self.tickcredit = params.tickcredit
        self.elastic = params.elastic

        self.pricetick = Utility.getPriceTick(self.product)
        self.multiplier= Utility.getMultiplier(self.product)

    def backTestRun(self, datelist, readinfile):
        for date in dateslist:
            maininst, unmaininst = Utility.getInst(date, readinfile)
            mkdataday = Utility.getMktData(date)
            if mkdataday == None:
                continue
            rawdata = self.mkdPreClean(mkdataday, [maininst, unmaininst])
            mdday   = self.findSpotTheo(rawdata, unmaininst, maininst)



    def mkdPreClean(self, mkdataday, instlist):
        # rawdata  = mkdataday.loc[mkdataday.symbol.str.len()==5, :]
        # if len(mkdataday) > len(rawdata):
        #     rawdata.to_csv(filename, index= False)
        #rawdata.sort_values(by= 'timestamp', ascending=True, inplace = True)

        rawdata   = rawdata.loc[rawdata.symbol.isin(instlist), :]
        ##...因为生产接入了两路行情，此处只选其一以避免重复计算，以及乱序。
        if self.mutiQuoteSource == False:
            srip = '172.17.8.51'
            rawdata = rawdata.loc[rawdata.srcip == srip, :]
        else:
            pass
        droplist = []
        for inst in instlist:
            _df = rawdata.loc[rawdata.symbol.str.upper()==unmaininst.upper(), :].index.tolist()
            _df[LitePriceColName.vol] = _df[LitePriceColName.totalvolume].diff(1)
            todrop = _df.loc[_df[LitePriceColName.vol]<0,:].index.tolist()
            droplist += todrop
        droplist = list(set(droplist))
        rawdata.drop(droplist, inplace = True)
        rawdata.dropna(subset=instlist, how='any', inplace= True)
        rawdata.reset_index(drop= True, inplace= True)
        return rawdata

        ##...calc barymid price...

        # rawdata['bary']       = (rawdata.askprice1 * rawdata.bidqty1 + rawdata.bidprice1 * rawdata.askqty1) / (rawdata.bidqty1 + rawdata.askqty1)
        # rawdata['spreadtick'] = (rawdata.askprice1 - rawdata.bidprice1) / pricetick
        # step = 0.2
        # rawdata['baryweight'] = np.maximum(step, 1 - (rawdata.spreadtick - 1) * step )
        # rawdata['barymid']    = rawdata.bary * rawdata.baryweight + (1 - rawdata.baryweight) * (rawdata.askprice1 + rawdata.bidprice1) / 2
        # rawdata.loc[(rawdata.bidprice1==0)|(rawdata.askprice1==0), 'barymid'] = rawdata.loc[(rawdata.bidprice1==0)|(rawdata.askprice1==0), 'lastprice']

    def findSpotTheo(self, df, unmaininst, maininst):
        unmainindexlist = df.loc[df.symbol.str.upper()==unmaininst.upper(), :].index.tolist()
        mainindexlist   = df.loc[df.symbol.str.upper()==maininst.upper(), :].index.tolist()
        #找到最近的spot和非主力月价格（barymid）
        df.loc[mainindexlist, LitePriceColName.spottheo] = df.loc[mainindexlist, LitePriceColName.barymid]
        df[LitePriceColName.spottheo] = df[LitePriceColName.spottheo].fillna(method ='ffill')
        df.loc[unmainindexlist, LitePriceColName.unmainbary] = df.loc[unmainindexlist, LitePriceColName.barymid]
        df[LitePriceColName.unmainbary] = df[LitePriceColName.unmainbary].fillna(method ='ffill')
        return df


        # #df.reset_index(drop= True, inplace= True)
        # df['dt_timestamp'] = pd.to_datetime(df.timestamp)
        # df['dt_spottheo'] = df['dt_timestamp']
        # df['dt_unmain']   = df['dt_timestamp']
        #
        # unmainindexlist = df.loc[df.symbol.str.upper()==unmaininst.upper(), :].index.tolist()
        # mainindexlist   = df.loc[df.symbol.str.upper()==maininst.upper(), :].index.tolist()
        #
        # df.loc[unmainindexlist,'vol'] = df.loc[unmainindexlist,'totalvolume'].diff(1).fillna(0)
        # df.loc[mainindexlist,'vol']   = df.loc[mainindexlist,'totalvolume'].diff(1).fillna(0)
        #
        # df = df.loc[df.vol>=0, :]
        #
        #
        #
        # df.loc[unmainindexlist, 'dt_spottheo'] = np.nan
        # df['dt_spottheo'] = df['dt_spottheo'].fillna(method='ffill')
        # df['timegap_spottheo'] = (df.dt_timestamp - df.dt_spottheo).apply(lambda x: x.total_seconds())
        # df.loc[mainindexlist, 'dt_unmain'] = np.nan
        # df['dt_unmain'] = df['dt_unmain'].fillna(method='ffill')
        # df['timegap_unmain'] = (df.dt_timestamp - df.dt_unmain).apply(lambda x: x.total_seconds())
        # df['timegap_lastline'] = (df.dt_timestamp.diff(1)).apply(lambda x: x.total_seconds())
        # del df['dt_timestamp']
        # del df['dt_spottheo']
        # del df['dt_unmain']
        #
        # df.loc[mainindexlist, 'spottheo'] = df.loc[mainindexlist, 'barymid']
        # df['spottheo'] = df.spottheo.fillna(method ='ffill')
        # df.loc[unmainindexlist, 'unmain'] = df.loc[unmainindexlist, 'barymid']
        # df['unmain'] = df.unmain.fillna(method ='ffill')
        #
        # df.loc[unmainindexlist, 'unmain_bid1'] = df.loc[unmainindexlist, 'bidprice1']
        # df['unmain_bid1'] = df.unmain_bid1.fillna(method ='ffill')
        #
        # df.loc[unmainindexlist, 'unmain_ask1'] = df.loc[unmainindexlist, 'askprice1']
        # df['unmain_ask1'] = df.unmain_ask1.fillna(method ='ffill')
        #
        # df.dropna(subset=['unmain','spottheo'], how='any', inplace= True)
        #
        # df.reset_index(drop= True, inplace= True)
        #
        # unmainindexlist = df.loc[df.symbol.str.upper()==unmaininst.upper(), :].index.tolist()
        # mainindexlist   = df.loc[df.symbol.str.upper()==maininst.upper(), :].index.tolist()
        #
        # df.loc[unmainindexlist,'dist'] = df.loc[unmainindexlist,'unmain'].diff(1).fillna(0).abs()
        # df.loc[mainindexlist,'dist']   = df.loc[mainindexlist,'spottheo'].diff(1).fillna(0).abs()
        #
        # return df

    def calcbasis(df, pricetick, alpha, alphalow,  alphahigh, lowend, highend, window, unmaininst, maininst):
        df = df.copy()
        df = df.loc[(df.spottheo > 0)&(df.unmain>0), :]
        df.reset_index(drop= True, inplace= True)
        df['snapbasis'] = df.unmain - df.spottheo
        df['spreadtrust'] = df.baryweight.fillna(1)
        df.loc[(df.askprice1==0)|(df.bidprice1==0),'spreadtrust'] = 1

        unmainindexlist = df.loc[df.symbol.str.upper()==unmaininst.upper(), :].index.tolist()
        mainindexlist   = df.loc[df.symbol.str.upper()==maininst.upper(), :].index.tolist()

        df.loc[unmainindexlist, 'ef_unmain'] = df.loc[unmainindexlist, 'unmain'].diff(window) / df.loc[unmainindexlist, 'dist'].rolling(window).sum()
        df.loc[mainindexlist, 'ef_main']     = df.loc[mainindexlist, 'spottheo'].diff(window) / df.loc[mainindexlist, 'unmain'].rolling(window).sum()
        df.loc[:, ['ef_unmain','ef_main']]   = df.loc[:, ['ef_unmain','ef_main']].fillna(method='ffill')
        df['efdiff'] = (df.ef_unmain - df.ef_main)

        df['snapbasis_smth'] = df.snapbasis.rolling(6).mean()
        df['Dspot'] = df.spottheo.diff(6)
        df['Dunmain'] = df.unmain.diff(6)
        df['Ddiff']  = (df.Dspot - df.Dunmain).abs() / pricetick
        df['Ddifftrust'] = 1
        df.loc[df.Ddiff<=0.5,'Ddifftrust'] = 0.5
        df.loc[df.Ddiff>=2,'Ddifftrust']   = 2


        df['snapbasis_max'] = df.snapbasis_smth.rolling(window).max()
        df['snapbasis_min'] = df.snapbasis_smth.rolling(window).min()
        df['snapbasis_maxchgpct'] = (df['snapbasis_max'] - df['snapbasis_min']) / df.spottheo *100

    #    df['snapbasis_std'] = df.snapbasis_smth.rolling(window).std()
    #    df['snapbasis_mean']= df.snapbasis_smth.rolling(window).mean()
    #    df['snapbasis_zscore'] = abs((df.snapbasis_smth - df.snapbasis_mean)) / df['snapbasis_std']

        slope = (alphahigh - alphalow) / (highend - lowend)
        df['maxchgpct_alpha'] = alphalow + slope * (df['snapbasis_maxchgpct'] - lowend)
        df.loc[df.snapbasis_maxchgpct>highend, 'maxchgpct_alpha'] = alphahigh
        df.loc[df.snapbasis_maxchgpct<lowend, 'maxchgpct_alpha']  = alphalow

        df['maxchgpct_alpha'] = df['maxchgpct_alpha'].fillna(alpha)
        df['alpha_adpt'] = (df['maxchgpct_alpha'] * df.spreadtrust).fillna(alpha)
        #df['alpha_adpt'] = (df['maxchgpct_alpha'] ).fillna(alpha)
        df.loc[df.alpha_adpt>0.4, 'alpha_adpt'] = 0.4

    #    df['Dsnapbasis'] =  df.snapbasis.diff(1).abs()
    #    df['Dspottheo']  =  df.spottheo.diff(1).abs()
    #    df['Dunmain']    =  df.unmain.diff(1).abs()
    #    df['changetrust']      = (df.Dunmain + pricetick) / (df.Dspottheo + pricetick)

    #    unmainindexlist = df.loc[df.symbol.str.upper()==unmaininst.upper(), :].index.tolist()
    #    df_unmain = df.loc[unmainindexlist, :]
    #    df_unmain['snapbasis_std'] = df_unmain.snapbasis.rolling(window).std()
    #    df_unmain['snapbasis_mean']= df_unmain.snapbasis.rolling(window).mean()
    #    df_unmain['snapbasis_zscore'] = abs((df_unmain.snapbasis - df_unmain.snapbasis_mean)) / df_unmain['snapbasis_std']
    #
    #    df_unmain['Dsnapbasis'] =  df_unmain.snapbasis.diff(1).abs()
    #    df_unmain['Dspottheo']  =  df_unmain.spottheo.diff(1).abs()
    #    df_unmain['Dunmain']    =  df_unmain.unmain.diff(1).abs()
    #
    #    df_unmain['changetrust']      = (df_unmain.Dunmain + 0.5*pricetick) / (df_unmain.Dspottheo + 0.5*pricetick)
    #
    #    collist = ['snapbasis_std','snapbasis_mean','snapbasis_zscore','Dsnapbasis','Dspottheo','Dunmain','changetrust']
    #    for col in collist:
    #        df.loc[unmainindexlist, col]    = df_unmain[col]
    #    df.loc[:,['snapbasis_std','snapbasis_mean','snapbasis_zscore']].fillna(method='ffill', inplace=True)
    #
    #    df['totaltrust'] = (df.spreadtrust * df.changetrust).fillna(1)
    #    df.loc[df.totaltrust>=5, 'totaltrust']   = 5
    #    df.loc[df.totaltrust<=1/5, 'totaltrust'] = 1/5

        length = len(df)
        snapbasis = df.snapbasis.values
        ewmabasis_prod = np.zeros(length)
        ewmabasis_adpt = np.zeros(length)
        alpha_adpt_array = np.zeros(length)
        alpha_adpt_array = df.alpha_adpt.values

        for index in range(length):
            alpha_prod = alpha
            alpha_adpt = alpha_adpt_array[index]
            if (index == 0):
                ewmabasis_prod[index] = snapbasis[index]
                ewmabasis_adpt[index] = snapbasis[index]
            else:
                indexpre = index - 1
                ewmabasis_prod[index] =  snapbasis[index] * alpha_prod + (1- alpha_prod)* ewmabasis_prod[indexpre]
                ewmabasis_adpt[index] =  snapbasis[index] * alpha_adpt + (1- alpha_adpt)* ewmabasis_adpt[indexpre]
        df['ewmabasis_prod'] =  ewmabasis_prod
        df['ewmabasis_adpt'] =  ewmabasis_adpt

        df['owntheo_prod'] = ewmabasis_prod + df.spottheo.values
        df['owntheo_adpt'] = ewmabasis_adpt + df.spottheo.values
        df['alpha_adpt']   = alpha_adpt_array

        return df


    #    df_unmain = df.loc[df.symbol.str.upper()==unmaininst.upper(), :]
    #    df_unmain = df_unmain.loc[df_unmain.spottheo > 0, :]
    #    df_unmain = df_unmain.loc[df_unmain.barymid > 0, :]
    #    df_unmain = df_unmain.drop_duplicates(subset='timestamp', keep='last')
    #
    #    #df_unmain['vol'] = df_unmain.totalvolume - df_unmain.totalvolume.shift(1)
    #    df_unmain = df_unmain.loc[df_unmain.vol>=0, :]
    #    df_unmain.reset_index(drop= True, inplace= True)
    #
    #    df_unmain['snapbasis']        = df_unmain.barymid - df_unmain.spottheo
    #    df_unmain['Dsnapbasis'] = df_unmain.snapbasis.diff(1)
    #    df_unmain['Dprice'] = df_unmain.barymid.diff(1)
    #    df_unmain['Dspot']  = df_unmain.spottheo.diff(1)
    #    df_unmain['Dprice/Dspot'] = abs(df_unmain.Dprice / df_unmain.Dspot.fillna(0))
    #    df_unmain['Dprice/Dspot'] = df_unmain['Dprice/Dspot'].fillna(1)
    #    df_unmain['multiadjust'] = df_unmain['Dprice/Dspot']
    #    df_unmain.loc[df_unmain['Dprice/Dspot']<0.5,'multiadjust'] = 0.5
    #    df_unmain.loc[df_unmain['Dprice/Dspot']>2,'multiadjust'] = 2
    #
    #
    #    df_unmain['snapbasis_std']    = df_unmain.snapbasis.rolling(window).std()
    #    df_unmain['Dsnapbasis_std']   = df_unmain.Dsnapbasis.rolling(window).std()
    #    df_unmain.loc[df_unmain['snapbasis_std']<10**(-7), 'snapbasis_std']    = 10**(-7)
    #    df_unmain.loc[df_unmain['Dsnapbasis_std']<10**(-7), 'Dsnapbasis_std']    = 10**(-7)
    #    df_unmain['snapbasis_mean']   = df_unmain.snapbasis.rolling(window).mean()
    #    df_unmain['Dsnapbasis_mean']   = df_unmain.Dsnapbasis.rolling(window).mean()
    #    #df_unmain['snapbasis_zscore'] = (df_unmain.snapbasis - df_unmain.snapbasis_mean) / df_unmain['snapbasis_std']
    #    #df_unmain['snapbasis_zscore'] = (df_unmain.snapbasis - df_unmain.snapbasis_mean) / 1.5
    #    df_unmain['Dsnapbasis_zscore'] = (df_unmain.Dsnapbasis - df_unmain.Dsnapbasis_mean) / df_unmain['Dsnapbasis_std'] * df_unmain.multiadjust
    #
    #    snapbasis = df_unmain.snapbasis.values
    #    Dsnapbasis_zscore = df_unmain.Dsnapbasis_zscore.values
    #
    #    length = len(df_unmain)
    #    ewmabasis_prod = np.zeros(length)
    #    ewmabasis_high = np.zeros(length)
    #    ewmabasis_adpt = np.zeros(length)
    #    alpha_adpt_array = np.zeros(length)
    #
    #
    #    for index in range(length):
    #        if (index == 0):
    #            ewmabasis_prod[index] = snapbasis[index]
    #            ewmabasis_high[index] = snapbasis[index]
    #            ewmabasis_adpt[index] = snapbasis[index]
    #            alpha_adpt_array[index] = alpha
    #        else:
    #            indexpre = index - 1
    #            if (index<= window):
    #                alpha_adpt = alpha
    #            else:
    #                if (abs(Dsnapbasis_zscore[index]) <= lowend):
    #                    alpha_adpt = alphalow
    #                elif (abs(Dsnapbasis_zscore[index]) >= highend):
    #                    alpha_adpt = alphahigh
    #                else:
    #                    slope = (alphahigh - alphalow) / (highend - lowend)
    #                    alpha_adpt = alphalow + slope * (abs(Dsnapbasis_zscore[index]) - lowend)
    #
    #            ewmabasis_prod[index] =  snapbasis[index] * alpha + (1- alpha)* ewmabasis_prod[indexpre]
    #            ewmabasis_high[index] =  snapbasis[index] * alphahigh + (1- alphahigh)* ewmabasis_high[indexpre]
    #            ewmabasis_adpt[index] =  snapbasis[index] * alpha_adpt + (1- alpha_adpt)* ewmabasis_adpt[indexpre]
    #            alpha_adpt_array[index] = alpha_adpt
    #    df_unmain['ewmabasis_prod'] =  ewmabasis_prod
    #    df_unmain['ewmabasis_high'] =  ewmabasis_high
    #    df_unmain['ewmabasis_adpt'] =  ewmabasis_adpt
    #
    #    df_unmain['owntheo_prod'] = ewmabasis_prod + df_unmain.spottheo.values
    #    df_unmain['owntheo_high'] = ewmabasis_high + df_unmain.spottheo.values
    #    df_unmain['owntheo_adpt'] = ewmabasis_adpt + df_unmain.spottheo.values
    #    df_unmain['alpha_adpt']   = alpha_adpt_array

    def preprocess(df):
        df2 = df.drop_duplicates(subset='timestamp', keep='last').set_index('timestamp')
        df2['vol'] = df2.totalvolume - df2.totalvolume.shift(1)
        df2 = df2.loc[df2.vol>=0, :]
        df2 = df2.iloc[500:-500,  ].dropna(how='any')

        return df2

    def plbacktest_onlyhitter(df, hittertickcredit, pricetick, elastic,unmaininst):
        df['timestampdt'] = pd.to_datetime(df.timestamp)
        df['timeofday']   = df.timestampdt.apply(lambda x: x.time())

        df['status'] = "notrading"
        df.loc[(df.timeofday>= dayopentime)&(df.timeofday<= dayclosetime),'status'] = 'trading'
        df.loc[(df.timeofday>= nightopentime), 'status']  = 'trading'

        df['strategy'] = 'off'
        df.loc[(df.timeofday>= daystarttime)&(df.timeofday<= dayendtime),'strategy'] = "on"
        df.loc[(df.timeofday>  dayendtime)&(df.timeofday<= dayclosetime),'strategy'] = "on2off"
        df.loc[(df.timeofday>= nightstarttime),'strategy'] = "on"
        df.strategy.iloc[-200:] = "on2off"

        creditprice = hittertickcredit * pricetick
        df['hitter_prod'] = 0
        df['hitter_adpt'] = 0
        df['hitterprice_prod'] = 0
        df['hitterprice_adpt'] = 0
        df['hitterpl_prod'] = 0
        df['hitterpl_adpt'] = 0
        df['pospl_prod'] = 0
        df['pospl_adpt'] = 0
        df['pos_prod'] = 0
        df['pos_adpt'] = 0
        df['elastmv_prod'] = 0
        df['elastmv_adpt'] = 0
        df['finaltheo_prod'] = 0
        df['finaltheo_adpt'] = 0

        df = df.loc[df.status=='trading', :]
        df.reset_index(drop=True, inplace=True)

        owntheo_prod  = df.owntheo_prod.values
        owntheo_adpt  = df.owntheo_adpt.values
        bidprice1     = df.unmain_bid1.values
        askprice1     = df.unmain_ask1.values
        timestamp_array = df.timestamp.values
        symbol_array    = df.symbol.values

        length = len(df)
        hitter_prod = np.zeros(length)
        hitter_adpt = np.zeros(length)
        hitterprice_prod = np.zeros(length)
        hitterprice_adpt = np.zeros(length)
        hitterpl_prod = np.zeros(length)
        hitterpl_adpt = np.zeros(length)
        pospl_prod    = np.zeros(length)
        pospl_adpt    = np.zeros(length)
        pos_prod      = np.zeros(length)
        pos_adpt      = np.zeros(length)
        elastmv_prod  = np.zeros(length)
        elastmv_adpt  = np.zeros(length)
        finaltheo_prod= np.zeros(length)
        finaltheo_adpt= np.zeros(length)

        rawbid_prod  = np.zeros(length)
        rawbid_adpt  = np.zeros(length)
        rawask_prod= np.zeros(length)
        rawask_adpt= np.zeros(length)
        mybid_prod  = np.zeros(length)
        mybid_adpt  = np.zeros(length)
        myask_prod= np.zeros(length)
        myask_adpt= np.zeros(length)

        timegap_array = df.timegap_lastline.values
        for index in range(length-1):
            content = df.loc[index, :]
            timestampnow = timestamp_array[index]
            timestampnext= timestamp_array[index+1]
            if (index == 0):
                elastmv_prod[index]   = 0
                elastmv_adpt[index]   = 0
                finaltheo_prod[index] = owntheo_prod[index] + elastmv_prod[index]
                finaltheo_adpt[index] = owntheo_adpt[index] + elastmv_adpt[index]
            else:
                preindex = index - 1
                elastmv_prod[index]       = -pos_prod[preindex] * elastic * pricetick
                elastmv_adpt[index]       = -pos_adpt[preindex] * elastic * pricetick
                finaltheo_prod[index]     =  owntheo_prod[index] + elastmv_prod[index]
                finaltheo_adpt[index]     =  owntheo_adpt[index] + elastmv_adpt[index]
                ##...以下为quoter报单价格，credit用的是hittercredit，后期可以改成quoter自己的credit
                rawbid_prod[index]        =  finaltheo_prod[index] - creditprice
                rawbid_adpt[index]        =  finaltheo_adpt[index] - creditprice
                rawask_prod[index]        =  finaltheo_prod[index] + creditprice
                rawask_adpt[index]        =  finaltheo_adpt[index] + creditprice
                mybid_prod[index]         =  np.floor(rawbid_prod[index] / pricetick) * pricetick
                mybid_adpt[index]         =  np.floor(rawbid_adpt[index] / pricetick) * pricetick
                myask_prod[index]         =  np.ceil(rawask_prod[index]  / pricetick) * pricetick
                myask_adpt[index]         =  np.ceil(rawask_adpt[index]  / pricetick) * pricetick

                hitterflag = True
                unmain_bidprice1 = bidprice1[index]
                unmain_askprice1 = askprice1[index]
                if (content.strategy =="on"):
                    if (timestampnow == timestampnext):
                        if (symbol_array[index].upper()==maininst.upper()) & (symbol_array[index+1].upper()==unmaininst.upper()):
                            unmain_bidprice1 = bidprice1[index+1]
                            unmain_askprice1 = askprice1[index+1]
                    if (symbol_array[index].upper()==unmaininst.upper()) & (timegap_array[index]>0):
                            hitterflag = False

                    if hitterflag == True:
                        hitternum_prod, tdprice_prod = tdengine_onlyhitter(finaltheo_prod[index],creditprice,unmain_bidprice1,unmain_askprice1)
                        hitternum_adpt, tdprice_adpt = tdengine_onlyhitter(finaltheo_adpt[index],creditprice,unmain_bidprice1,unmain_askprice1)
                    else:
                        hitternum_prod = 0
                        tdprice_prod   = 0
                        hitternum_adpt = 0
                        tdprice_adpt   = 0
                    hitter_prod[index] = hitternum_prod
                    hitter_adpt[index] = hitternum_adpt
                    hitterprice_prod[index] = tdprice_prod
                    hitterprice_adpt[index] = tdprice_adpt
                    hitterpl_prod[index] =  hitternum_prod * (finaltheo_prod[index] - tdprice_prod)
                    hitterpl_adpt[index] =  hitternum_adpt * (finaltheo_adpt[index] - tdprice_adpt)
                    pospl_prod[index]    =  pos_prod[preindex] * (finaltheo_prod[index] - finaltheo_prod[preindex])
                    pospl_adpt[index]    =  pos_adpt[preindex] * (finaltheo_adpt[index] - finaltheo_adpt[preindex])

                    pos_prod[index]     = pos_prod[preindex] + hitter_prod[index]
                    pos_adpt[index]     = pos_adpt[preindex] + hitter_adpt[index]

                else:
                    time5 = time.time()
                    hitter_prod[index] = 0
                    hitter_adpt[index] = 0
                    hitterprice_prod[index] = 0
                    hitterprice_adpt[index] = 0

                    hitterpl_prod[index] =  0
                    hitterpl_adpt[index] =  0

                    pospl_prod[index]    =  pos_prod[preindex] * (finaltheo_prod[index] - finaltheo_prod[preindex])
                    pospl_adpt[index]    =  pos_adpt[preindex] * (finaltheo_adpt[index] - finaltheo_adpt[preindex])

                    pos_prod[index]      =  pos_prod[preindex] + hitter_prod[index]
                    pos_adpt[index]      =  pos_adpt[preindex] + hitter_adpt[index]

                if (content.strategy == "on2off"):
                    covernum_prod = pos_prod[index]
                    if (covernum_prod > 0):
                        if bidprice1[index] > 0:
                            coverprice_prod = bidprice1[index]
                        else:
                            coverprice_prod = askprice1[index] - pricetick
                        coverpl_prod = (coverprice_prod - finaltheo_prod[index]) * covernum_prod
                        hitter_prod[index] = -covernum_prod
                        hitterpl_prod[index] =  coverpl_prod
                        hitterprice_prod[index] = coverprice_prod
                        pos_prod[index]      = 0

                    elif (covernum_prod < 0):
                        if askprice1[index] > 0:
                            coverprice_prod = askprice1[index]
                        else:
                            coverprice_prod = bidprice1[index] + pricetick
                        coverpl_prod = (coverprice_prod - finaltheo_prod[index]) * covernum_prod
                        hitter_prod[index] = -covernum_prod
                        hitterpl_prod[index]   =  coverpl_prod
                        hitterprice_prod[index] = coverprice_prod
                        pos_prod[index]    = 0

                    covernum_adpt = pos_adpt[index]
                    if (covernum_adpt > 0):
                        if bidprice1[index] > 0:
                            coverprice_adpt = bidprice1[index]
                        else:
                            coverprice_adpt = askprice1[index] - pricetick
                        coverpl_adpt = (coverprice_adpt - finaltheo_adpt[index]) * covernum_adpt
                        hitter_adpt[index] = -covernum_adpt
                        hitterpl_adpt[index]  =  coverpl_adpt
                        hitterprice_adpt[index] = coverprice_adpt
                        pos_adpt[index] = 0
                    elif (covernum_adpt < 0):
                        if askprice1[index] > 0:
                            coverprice_adpt = askprice1[index]
                        else:
                            coverprice_adpt = bidprice1[index] + pricetick
                        coverpl_adpt = (coverprice_adpt - finaltheo_adpt[index]) * covernum_adpt
                        hitter_adpt[index] = -covernum_adpt
                        hitterpl_adpt[index] =  coverpl_adpt
                        hitterprice_adpt[index] = coverprice_adpt
                        pos_adpt[index]      = 0

        df['hitter_prod'] = hitter_prod
        df['hitter_adpt'] = hitter_adpt
        df['hitterprice_prod'] = hitterprice_prod
        df['hitterprice_adpt'] = hitterprice_adpt
        df['hitterpl_prod'] = hitterpl_prod
        df['hitterpl_adpt'] = hitterpl_adpt
        df['pospl_prod'] = pospl_prod
        df['pospl_adpt'] = pospl_adpt
        df['pos_prod'] = pos_prod
        df['pos_adpt'] = pos_adpt
        df['elastmv_prod'] = elastmv_prod
        df['elastmv_adpt'] = elastmv_adpt
        df['finaltheo_prod'] = finaltheo_prod
        df['finaltheo_adpt'] = finaltheo_adpt
        df['rawbid_prod'] = rawbid_prod
        df['rawbid_adpt'] = rawbid_adpt
        df['rawask_prod'] = rawask_prod
        df['rawask_adpt'] = rawask_adpt
        df['mybid_prod']  = mybid_prod
        df['mybid_adpt']  = mybid_adpt
        df['myask_prod']  = myask_prod
        df['myask_adpt']  = myask_adpt

        return df


    def tdengine_onlyhitter(mytheo, credit, bidprice1, askprice1):
        hitter = 0
        hitterprice = 0
        if ((mytheo + credit <= bidprice1)&(bidprice1>0)):
            hitter = -1
            hitterprice = bidprice1
        elif ((mytheo - credit >= askprice1) & (askprice1 > 0)):
            hitter = 1
            hitterprice = askprice1


        return hitter, hitterprice


    def daystats_onlyhitter(df):
        hittertrades_prod = df.hitter_prod.abs().sum()
        hitterpl_prod     = df.hitterpl_prod.sum()
        pospl_prod        = df.pospl_prod.sum()
        totalpl_prod = hitterpl_prod + pospl_prod

        hittertrades_adpt = df.hitter_adpt.abs().sum()
        hitterpl_adpt     = df.hitterpl_adpt.sum()
        pospl_adpt        = df.pospl_adpt.sum()
        totalpl_adpt      = hitterpl_adpt + pospl_adpt

        result       = pd.Series({'hitternum_prod': hittertrades_prod, 'hitternum_adpt': hittertrades_adpt, \
                                  'hitterpl_prod': hitterpl_prod, 'hitterpl_adpt': hitterpl_adpt, \
                                  'pospl_prod': pospl_prod, 'pospl_adpt': pospl_adpt, \
                                  'totalpl_prod': totalpl_prod, 'totalpl_adpt': totalpl_adpt })
        return result

    def statsandhist(result):
        stats = pd.DataFrame(columns=['adpt','prod'])
        totalplmean_adpt = result.totalpl_adpt.mean()
        totalplmean_prod = result.totalpl_prod.mean()
        stats = stats.append(pd.Series({'adpt':totalplmean_adpt, 'prod':totalplmean_prod}, name=u'日均盈亏'))
        totalplmedian_adpt = result.totalpl_adpt.median()
        totalplmedian_prod = result.totalpl_prod.median()
        stats = stats.append(pd.Series({'adpt':totalplmedian_adpt, 'prod':totalplmedian_prod}, name=u'日均盈亏中位数'))
        tradesperday_adpt = result.hitternum_adpt.mean()
        tradesperday_prod = result.hitternum_prod.mean()
        stats = stats.append(pd.Series({'adpt':tradesperday_adpt, 'prod':tradesperday_prod}, name=u'日均交易次数'))

        gainratio_adpt = np.count_nonzero(result.totalpl_adpt>0) / len(result)
        lossratio_adpt = np.count_nonzero(result.totalpl_adpt<0) / len(result)
        gainratio_prod = np.count_nonzero(result.totalpl_prod>0) / len(result)
        lossratio_prod = np.count_nonzero(result.totalpl_prod<0) / len(result)
        stats = stats.append(pd.Series({'adpt':gainratio_adpt, 'prod':gainratio_prod}, name='盈利天数比例'))
        stats = stats.append(pd.Series({'adpt':lossratio_adpt, 'prod':lossratio_prod}, name='亏损天数比例'))

        totalplpertrade_adpt = result.totalpl_adpt.sum() / result.hitternum_adpt.sum()
        totalplpertrade_prod = result.totalpl_prod.sum() / result.hitternum_prod.sum()
        stats = stats.append(pd.Series({'adpt':totalplpertrade_adpt, 'prod':totalplpertrade_prod}, name='每次盈利'))
        tradeplpertrade_adpt = result.hitterpl_adpt.sum() / result.hitternum_adpt.sum()
        tradeplpertrade_prod = result.hitterpl_prod.sum() / result.hitternum_prod.sum()
        stats = stats.append(pd.Series({'adpt':tradeplpertrade_adpt, 'prod':tradeplpertrade_prod}, name='每次交易盈利'))
        posplpertrade_adpt = result.pospl_adpt.sum() / result.hitternum_adpt.sum()
        posplpertrade_prod = result.pospl_prod.sum() / result.hitternum_prod.sum()
        stats = stats.append(pd.Series({'adpt':posplpertrade_adpt, 'prod':posplpertrade_prod}, name='每次持仓盈利'))

        result['totalpldiff(adpt-diff)'] = result.totalpl_adpt - result.totalpl_prod
        winratio  = np.count_nonzero(result['totalpldiff(adpt-diff)']>0) / len(result)
        lossratio = np.count_nonzero(result['totalpldiff(adpt-diff)']<0) / len(result)
        stats = stats.append(pd.Series({'adpt':winratio, 'prod':None}, name='adpt战胜prod的比例'))
        stats = stats.append(pd.Series({'adpt':lossratio, 'prod':None}, name='adpt输给prod的比例'))

        return stats

    def plotgraph(df, date, maininst, unmaininst):

        df = df.copy()
        df = df.drop_duplicates(subset='timestamp', keep='first')
        ax1 = plt.subplot(1,1,1)
        ax1.plot(df.index.tolist(), df.owntheo_adpt, 'r',label='theo_adpt')
        ax1.plot(df.index.tolist(), df.owntheo_prod, 'b', label='theo_prod')
        ax1.plot(df.index.tolist(), df.spottheo, 'y',label='spot')
    #    plt.plot(df.index.tolist(), df.unmain_ask1.replace(0,np.nan),'g')
    #    plt.plot(df.index.tolist(), df.unmain_bid1.replace(0,np.nan),'y')

        ax2 = plt.twinx()
        ax2.plot(df.index.tolist(), df.efdiff, 'c--', label='efdiff')
        #ax2.plot(df.index.tolist(), df.pos_prod, 'b--', label='pos_prod')

        buy_adpt  = df.loc[df.hitter_adpt>0,:]
        sell_adpt = df.loc[df.hitter_adpt<0,:]
        buy_prod  = df.loc[df.hitter_prod>0,:]
        sell_prod = df.loc[df.hitter_prod<0,:]
        ax1.plot(buy_adpt.index.tolist(),  buy_adpt.hitterprice_adpt, 'ro')
        ax1.plot(sell_adpt.index.tolist(), sell_adpt.hitterprice_adpt, 'yo')
        ax1.plot(buy_prod.index.tolist(),  buy_prod.hitterprice_prod, 'r^')
        ax1.plot(sell_prod.index.tolist(), sell_prod.hitterprice_prod, 'y^')

        ax1.legend(loc ='upper left')
        ax2.legend(loc ='upper right')

        savename = "NewGraph\\" + str(date) + "_" + "(maininst=" + maininst + "," + "unmaininst=" + unmaininst +").png"
        plt.savefig(savename)

        fignew = plt.figure()
        ax1 = fignew.add_subplot(1,1,1)
        ax1.plot(df.index.tolist(), df.efdiff, 'r--',label='efdiff')

        ax2 = ax1.twinx()
        ax2.plot(df.index.tolist(), df.snapbasis_smth, 'c--', label='snapbasis_smth')
        ax1.legend(loc ='upper left')
        ax2.legend(loc ='upper right')


        fig2 = plt.figure()
        ax = fig2.add_subplot(1,1,1)
        ax.plot(df.index.tolist(), df.snapbasis_maxchgpct, 'r', label ='snapbasis_maxchgpct')
        ax2 = ax.twinx()
        ax2.plot(df.index.tolist(), df.alpha_adpt, 'b', label ='alpha_adpt')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        return buy_adpt, sell_adpt, buy_prod, sell_prod, plt

    def basisstdstats(product):
        files = os.listdir()
        files = [file for file in files if "backtestdata" in file and product in file]
        files.sort()
        mergedf = pd.DataFrame()
        for name in files:
            print(name)
            time1 = time.time()
            tempdf = pd.read_csv(name, usecols=['timestamp','snapbasis_std'])
            time2 = time.time()
            mergedf = pd.concat([mergedf, tempdf], axis= 0)
            time3 = time.time()
            print('time2-time1', time2-time1)
            print('time3-time2', time3-time2)
        return mergedf

    def smpcols(df):
        df['totalpl_adpt'] =  df.hitterpl_adpt.cumsum() + df.pospl_adpt.cumsum()
        df['totalpl_prod'] =  df.hitterpl_prod.cumsum() + df.pospl_prod.cumsum()

        colsforshow = ['symbol','timestamp','bidprice1','askprice1','vol','barymid','spottheo','unmain',\
        'unmain_bid1','unmain_ask1','snapbasis','snapbasis_smth','snapbasis_max','snapbasis_min','snapbasis_maxchgpct','maxchgpct_alpha',\
        'spreadtrust','Ddiff','Ddifftrust','dist','ef_main','ef_unmain','efdiff','alpha_adpt','owntheo_adpt','owntheo_prod','status','strategy','pos_adpt','pos_prod',\
        'totalpl_adpt','totalpl_prod']

        return df[colsforshow]



    

