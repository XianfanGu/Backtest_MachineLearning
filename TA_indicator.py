import numpy as np
import math
import pandas as pd
from talib import RSI, BBANDS, OBV, EMA, MA, MACD,STOCH, CCI, AD


class TA:

    def __init__(self,dates, high, low, price,volume):
        self.training_window = pd.DataFrame({'timestamp':dates.date})
        #self.training_window = self.training_window.set_index('timestamp')

    def appendDate(self,dates):
        for i in dates:
            timestamp_ = self.training_window['timestamp'].values
            if not i.date() in timestamp_:
                self.training_window = self.training_window.append({'timestamp':i.date()},ignore_index=True)

    def calcMA(self,dates,price):
        ma = pd.DataFrame(columns=['timestamp','price','data_sma5','data_sma10','data_sma15','data_sma20','data_sma60'])
        #ma = ma.set_index('timestamp')
        index = 0
        data_sma5 = np.array(MA(price, 5))
        data_sma10 = np.array(MA(price, 10))
        data_sma15 = np.array(MA(price, 15))
        data_sma20 = np.array(MA(price, 20))
        data_sma60 = np.array(MA(price, 60))

        for data,date in zip(price,dates):
            timestamp_ = self.training_window['timestamp'].values
            if math.isnan(data) or (date.date() in timestamp_ and 'price' in self.training_window.columns):
                continue
            ma = ma.append({'timestamp':date.date(), 'price':price[index], 'data_sma5':data_sma5[index], 'data_sma10':data_sma10[index], 'data_sma15':data_sma15[index], 'data_sma20':data_sma20[index],
                       'data_sma60':data_sma60[index]},ignore_index=True)
            index = index + 1
        self.mergeMatrice(ma)
        print(self.training_window.loc[self.training_window['timestamp'] == date.date()])
        print("result: ",self.training_window.loc[self.training_window['timestamp'] == date.date()]['price'].values == np.NaN)
        return ma

    def calcEMA(self,volume, mov_date):
        real = EMA(volume, mov_date)
        self.mergeMatrice(real)
        return real

    def calcPVO(self,volume):
        PVO = []
        EMA_12 = calcEMA(volume, 12)
        EMA_26 = calcEMA(volume, 26)
        for date in range(len(volume)):
            if math.isnan(EMA_12[date]) or math.isnan(EMA_26[date]):
                PVO.append([date + 1, np.nan])
            else:
                PVO.append([date + 1, (EMA_12[date] - EMA_26[date]) / EMA_12[date] * 100])
        # print(PVO)
        self.mergeMatrice(PVO)
        return PVO

    def calcSTOCH(self,high, low, price):
        date = 1
        STOCH_ = []
        fastk, fastd = STOCH(high, low, price)
        for ele, ele1 in zip(fastk, fastd):
            STOCH_.append([date, ele, ele1])
            date = date + 1
        return STOCH_

    def calcMACD(self,price):
        date = 1
        MACD_ = []
        macd, macdsignal, macdhist = MACD(price)
        for ele, ele1, ele2 in zip(macd, macdsignal, macdhist):
            MACD_.append([date, ele, ele1, ele2])
            date = date + 1
        return MACD_

    def calcCCI(self,high, low, price):
        date = 1
        CCI_ = []
        real = CCI(high, low, price)
        for ele in real:
            CCI_.append([date, ele])
            date = date + 1
        return CCI_

    def calcPM(self,price):
        date = 1
        PM = []
        loss = 0
        gain = 0
        price_prev = 0
        price_prev_5 = 0
        price_prev_10 = 0
        price_prev_15 = 0
        price_prev_60 = 0
        is_delta_5 = 1
        is_delta_10 = 1
        is_delta_15 = 1
        is_delta_60 = 1

        for data in price:
            # print(data)
            if (math.isnan(data)):
                continue

            # price change: one day ratio
            if (price_prev == 0):
                price_prev = data
                price_delta = 0
                price_ratio = 0
            else:
                price_next = data
                price_delta = price_next - price_prev
                price_ratio = price_next / price_prev
                price_prev = price_next

            # price change: five days ratio
            if (is_delta_5 >= 4 and not math.isnan(price[date - 5 + 1])):
                price_next_5 = data
                price_prev_5 = price[date - 5 + 1]
                price_delta_5 = price_next_5 - price_prev_5
                price_ratio_5 = price_next_5 / price_prev_5
            else:
                price_delta_5 = 0
                price_ratio_5 = 0

            # price change: ten days ratio
            if (is_delta_10 >= 9 and not math.isnan(price[date - 10 + 1])):
                price_next_10 = data
                price_prev_10 = price[date - 10 + 1]
                price_delta_10 = price_next_10 - price_prev_10
                price_ratio_10 = price_next_10 / price_prev_10
            else:
                price_delta_10 = 0
                price_ratio_10 = 0

            # price change: 15 days ratio
            if (is_delta_15 >= 14 and not math.isnan(price[date - 15 + 1])):
                price_next_15 = data
                price_prev_15 = price[date - 15 + 1]
                price_delta_15 = price_next_15 - price_prev_15
                price_ratio_15 = price_next_15 / price_prev_15
            else:
                price_delta_15 = 0
                price_ratio_15 = 0

            # price change: 60 days ratio
            if (is_delta_60 >= 59 and not math.isnan(price[date - 60 + 1])):
                price_next_60 = data
                price_prev_60 = price[date - 60 + 1]
                price_delta_60 = price_next_60 - price_prev_60
                price_ratio_60 = price_next_60 / price_prev_60
            else:
                price_delta_60 = 0
                price_ratio_60 = 0

            PM.append([date, data, price_ratio, price_ratio_5, price_ratio_10, price_ratio_15, price_ratio_60])
            date = date + 1
            is_delta_5 = is_delta_5 + 1
            is_delta_10 = is_delta_10 + 1
            is_delta_15 = is_delta_15 + 1
            is_delta_60 = is_delta_60 + 1

        return PM

    def calcAD(self,high, low, price, volume):
        date = 1
        AD_ = []
        real = AD(high, low, price, volume)
        for ele in real:
            AD_.append([date, ele])
            date = date + 1
        return AD_

    def calcOBV(self,price, volume):
        date = 1
        OBV_ = []
        obv = OBV(price, volume)
        for ele in obv:
            OBV_.append([date, ele])
            date = date + 1
        return OBV_

    def calcRSI(self,price):
        date = 1
        RSI_ = []
        rsi = RSI(price, timeperiod=14)
        for ele in rsi:
            RSI_.append([date, ele])
            date = date + 1
        return RSI_

    def mergeMatrice(self, Matrix_B):
        self.training_window = pd.merge(self.training_window,Matrix_B,on='timestamp',how='inner')

    def removeNAN(self):
        for data in input_data_set:
            # print(len(input_data_set))
            # print(len(tar))
            if (np.isnan(data).any() or np.isnan(tar[date - 1]).any()):
                input_data_set = np.delete(input_data_set, date - 1, 0)
                tar = np.delete(tar, date - 1, 0)
            else:
                date = date + 1

