import numpy as np
import math
import pandas as pd
from toollib.Data.Data import Data
from talib import RSI, OBV, EMA, MA, MACD,STOCH, CCI, AD

class TA(Data):

    def __init__(self,dates):
        self.training_window = pd.DataFrame({'timestamp': dates.date})
        super().__init__(dates)

    def appendDate(self,dates):
        for i in dates:
            timestamp_ = self.training_window['timestamp'].values
            if not i.date() in timestamp_:
                self.training_window = self.training_window.append({'timestamp':i.date()},ignore_index=True)

    def addFeature(self, ta_list, dates, inputs):
        price = inputs[0]
        volume = inputs[1]
        high = inputs[2]
        low = inputs[3]
        if 'PM' in ta_list:
            re,exist_ = self.calcPM(dates, price)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'MA' in ta_list:
            re,exist_ = self.calcMA(dates, price)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'RSI' in ta_list:
            re,exist_ = self.calcRSI(dates, price)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'OBV' in ta_list:
            re,exist_ = self.calcOBV(dates,price, volume)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'EMA' in ta_list:
            _,re, exist_ = self.calcEMA(dates, volume, 12)
            # print(re)
            if (exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'MACD' in ta_list:
            re,exist_ = self.calcMACD(dates, price)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'STOCH' in ta_list:
            re,exist_ = self.calcSTOCH(dates, high, low, price)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'CCI' in ta_list:
            re,exist_ = self.calcCCI(dates,high, low, price)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)
        if 'AD' in ta_list:
            re,exist_ = self.calcAD(dates,high, low, price, volume)
            #print(re)
            if(exist_ == 1):
                self.updataFeature(re)
            else:
                self.mergeMatrice(re)

    def calcMA(self,dates,price):
        #ma = pd.DataFrame(columns=['timestamp','price','data_sma5','data_sma10','data_sma15','data_sma20','data_sma60'])
        data_sma5 = np.array(MA(price, 5))
        data_sma10 = np.array(MA(price, 10))
        data_sma15 = np.array(MA(price, 15))
        data_sma20 = np.array(MA(price, 20))
        data_sma60 = np.array(MA(price, 60))
        """
        isExist = 0
        for data,date in zip(price,dates):
            timestamp_ = self.training_window['timestamp'].values
            if(all(ele in self.training_window.columns.values for ele in ma.columns.values)):
                isExist = 1
            if math.isnan(data) or (date.date() in timestamp_ and all(ele in self.training_window.columns.values for ele in ma.columns.values) and not any(np.isnan(ele) for ele in self.training_window.loc[self.training_window['timestamp'] == date.date()].values[0][1:])):
                index = index + 1
                continue
            ma = ma.append({'timestamp':date.date(), 'price':price[index], 'data_sma5':data_sma5[index], 'data_sma10':data_sma10[index], 'data_sma15':data_sma15[index], 'data_sma20':data_sma20[index],
                       'data_sma60':data_sma60[index]},ignore_index=True)
            index = index + 1
        """
        ma,isExist = self.createDateFrame(dates,np.column_stack((data_sma5,data_sma10,data_sma15,data_sma20,data_sma60)),['timestamp','data_sma5','data_sma10','data_sma15','data_sma20','data_sma60'])
        #print(ma)
        return ma,isExist

    def calcEMA(self,dates,volume, mov_date):
        real = EMA(volume, mov_date)
        EMA_,isExist = self.createDateFrame(dates,real,['timestamp','EMA'])
        return real,EMA_,isExist

    def calcPVO(self,dates,volume):
        PVO = []
        EMA_12,_,_ = self.calcEMA(volume, 12)
        EMA_26,_,_ = self.calcEMA(volume, 26)
        for date in range(len(volume)):
            if math.isnan(EMA_12[date]) or math.isnan(EMA_26[date]):
                PVO.append(np.nan)
            else:
                PVO.append((EMA_12[date] - EMA_26[date]) / EMA_12[date] * 100)
        # print(PVO)
        PVO,isExist = self.createDateFrame(dates,PVO,['timestamp','PVO'])
        return PVO,isExist

    def calcSTOCH(self,dates, high, low, price):
        STOCH_ = []
        fastk, fastd = STOCH(high, low, price)
        for ele, ele1 in zip(fastk, fastd):
            STOCH_.append([ele, ele1])
        STOCH_, isExist = self.createDateFrame(dates, STOCH_, ['timestamp', 'fastk','fastd'])
        return STOCH_, isExist

    def calcMACD(self,dates,price):
        MACD_ = []
        macd, macdsignal, macdhist = MACD(price)
        for ele, ele1, ele2 in zip(macd, macdsignal, macdhist):
            MACD_.append([ele, ele1, ele2])
        MACD_, isExist = self.createDateFrame(dates, MACD_, ['timestamp', 'macd', 'macdsignal', 'macdhist'])
        return MACD_, isExist

    def calcCCI(self,dates,high, low, price):
        real = CCI(high, low, price)
        CCI_, isExist = self.createDateFrame(dates, real, ['timestamp', 'CCI'])
        return CCI_, isExist

    def calcPM(self,dates,price):
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
                PM.append([np.nan,np.nan,np.nan,np.nan,np.nan])
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

            PM.append([price_ratio, price_ratio_5, price_ratio_10, price_ratio_15, price_ratio_60])
            date = date + 1
            is_delta_5 = is_delta_5 + 1
            is_delta_10 = is_delta_10 + 1
            is_delta_15 = is_delta_15 + 1
            is_delta_60 = is_delta_60 + 1
        PM, isExist = self.createDateFrame(dates, PM, ['timestamp', 'price_ratio','price_ratio_5','price_ratio_10','price_ratio_15','price_ratio_60'])
        return PM, isExist

    def calcAD(self,dates,high, low, price, volume):
        real = AD(high, low, price, volume)
        AD_, isExist = self.createDateFrame(dates, real, ['timestamp', 'AD'])
        return AD_, isExist

    def calcOBV(self,dates,price, volume):
        obv = OBV(price, volume)
        OBV_, isExist = self.createDateFrame(dates, obv, ['timestamp', 'OBV'])
        return OBV_, isExist

    def calcRSI(self,dates,price):
        rsi = RSI(price, timeperiod=14)
        RSI_, isExist = self.createDateFrame(dates, rsi, ['timestamp', 'RSI'])
        return RSI_, isExist

    def mergeMatrice(self, Matrix_B):
        #print(Matrix_B)
        self.training_window = pd.merge(self.training_window,Matrix_B,on='timestamp',how='inner')

    def updataFeature(self, df):
        for index, row in df.iterrows():
            #print(row.values)
            if not any(np.isnan(ele) for ele in row.values[1:]):
                self.training_window.loc[self.training_window['timestamp'] == row['timestamp'],df.columns.values] = row.values
        #self.training_window.update(Matrix_B, overwrite = True)

    def createDateFrame(self,dates,datalist,tags):
        df = pd.DataFrame(columns=tags)
        # ma = ma.set_index('timestamp')
        index = 0
        isExist = 0
        for data, date in zip(datalist, dates):
            timestamp_ = self.training_window['timestamp'].values
            if (all(ele in self.training_window.columns.values for ele in df.columns.values)):
                isExist = 1
            if date.date() in timestamp_ and all(ele in self.training_window.columns.values for ele in df.columns.values) \
                    and not any(np.isnan(ele) for ele in self.training_window.loc[self.training_window['timestamp'] == date.date()].values[0][1:]):
                index = index + 1
                continue
            row = list(np.append(date.date(), data))
            #print(row)
            df = df.append(pd.Series(row,index=df.columns[:len(row)]), ignore_index=True)
            index = index + 1
        return df, isExist

    def removeNAN(self,input_data_set):
        for data in input_data_set:
            # print(len(input_data_set))
            # print(len(tar))
            if (np.isnan(data).any() or np.isnan(tar[date - 1]).any()):
                input_data_set = np.delete(input_data_set, date - 1, 0)
                tar = np.delete(tar, date - 1, 0)
            else:
                date = date + 1

    def getInputMatrix(self):
        input_window = self.training_window.drop('timestamp',1)
        return input_window.as_matrix()