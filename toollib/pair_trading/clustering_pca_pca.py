"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from zipline.api import (history,order, record, symbol,order_target_percent,set_benchmark,set_long_only,schedule_function,sid,date_rules,time_rules)
from zipline import run_algorithm
import math
import numpy as np
# Pandas library: https://pandas.pydata.org/
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import urllib
from io import StringIO
import csv
import datetime as dt
N_PRIN_COMPONENTS = 7
SYMBOL = 'SPY'
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    feature_num = 11
    context.orders_submitted = False
    large_num = 9999999
    least_num = 0
    context.n_components = 11
    context.n_components = 6
    context.SP500_symbol = ['AAPL', 'ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL',
                    'AMG', 'A', 'GAS', 'ARE', 'APD', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO',
                    'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI',
                    'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY',
                    'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK.B', 'BBY', 'BLX', 'HRB',
                    'BA', 'BWA', 'BXP', 'BSX', 'BMY', 'BRCM', 'BF.B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB', 'COF',
                    'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK',
                    'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH',
                    'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST',
                    'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO',
                    'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB',
                    'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG',
                    'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB',
                    'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL',
                    'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD',
                    'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP',
                    'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR',
                    'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI',
                    'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR',
                    'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L',
                    'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT',
                    'MKC', 'MCD', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP',
                    'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP',
                    'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS',
                    'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO',
                    'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI',
                    'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG',
                    'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG',
                    'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLD', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX',
                    'SEE', 'SRE', 'SHW', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK',
                    'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE',
                    'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJX', 'TMK',
                    'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS',
                    'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT',
                    'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN',
                    'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']

    context.model2 = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2, C=10.0, verbose=True)  # 8.05 for SVM model
    context.model3 = KNeighborsClassifier(n_neighbors=feature_num, p=3, metric='minkowski')  # 7.05 for  model
    context.model5 = DecisionTreeClassifier(criterion='entropy', max_depth=feature_num, random_state=0)
    context.model4 = RandomForestClassifier(criterion='entropy', n_estimators=feature_num, random_state=1,
                                            n_jobs=2)  # 5.2 for randomforest
    context.model1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    context.model = KMeans(n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=0)
    #context.model = DBSCAN(eps=0.2,min_samples=3,metric='euclidean')
    context.lookback = 350  # Look back 62 days
    context.history_range = 350  # Only consider the past 400 days' history
    context.threshold = 4.05
    context.longprices = large_num
    context.shortprices = least_num
    context.times = 0
    set_long_only()
    # Generate a new model every week
    schedule_function(create_model, date_rules.week_end(), time_rules.market_close(minutes=10))
    """
    # Generate a new model every week
    schedule_function(create_model1, date_rules.week_end(), time_rules.market_close(minutes=10))
    """

    # Trade at the start of every day
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes=1))

def handle_data(context, data):
    pass
def create_model(context, data):
    # Get the relevant daily prices
    X = {}
    if(context.times == 1):
        return 0
    print("test")
    info = []
    id = 0
    for symbol_ in context.SP500_symbol:
        try:
            recent_prices = data.history(symbol(symbol_), 'price', context.history_range, '1d').values
            recent_volume = data.history(symbol(symbol_), 'volume', context.history_range, '1d').values
            info.append({'vol': sum(recent_volume) / context.history_range, 'stock': symbol_})
            train_, target_ = getTrainingWindow(recent_prices, recent_volume)
            X_normalized_ = preprocessing.normalize(train_, norm='l2')
            y = np.delete(target_, 0, 1)
            y = np.ravel(y)
            sc = StandardScaler()
            sc.fit(X_normalized_)
            X_std = sc.transform(X_normalized_)
            pca = PCA(n_components=N_PRIN_COMPONENTS)
            X_pca = pca.fit_transform(X_std)
            X_train, y_train = X_pca, y
            X[symbol_] = []
            X[symbol_].append(X_train)
        except Exception as error:
            pass
        id = id + 1
    #print(info[0]['stock'])
    train_1 = []
    label = []
    print(X['AAPL'][0])
    print("\n")
    print(X['AAPL'][0].T)
    for symbol_ in context.SP500_symbol:
        try:

            X_perstock = (X[symbol_][0]).T

            pca_1 = PCA(n_components=1)
            X_pca_per_stock = pca_1.fit_transform(X_perstock)
            #print(X_pca_per_stock.T[0])
            label.append(symbol_)
            train_1.append(X_pca_per_stock.T[0])
        except Exception as error:
            pass
    #print(train_1)
    X_normalized_1 = preprocessing.normalize(train_1, norm='l2')
    sc = StandardScaler()
    sc.fit(X_normalized_1)
    X_std_1 = sc.transform(X_normalized_1)
    print(X_std_1)
    y_pred = context.model.fit_predict(X_std_1)
    labels = context.model.labels_
    n_clusters_ = len(set(labels))
    print("Clusters discovered: ", n_clusters_)
    #print(X_perday)



    label_list = unique(y_pred)
    print(label_list)
    pca = PCA(n_components=2)
    cluster = []
    X_axis = pca.fit_transform(X_std_1)
    X_axis = np.array(X_axis)
    for y_i in label_list:
        color = np.random.rand(3, )
        print(color)
        plt.scatter(X_axis[y_pred == y_i,0], X_axis[y_pred == y_i,1], c=color, marker='o', s=40)
        cluster.append(np.where(y_pred==y_i))

    for j in range(0,len(y_pred),20):
        plt.annotate(
            label[j],
            xy=(X_axis[j,0], X_axis[j,1]), xytext=(-0.2, 0.2),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    plt.show()
    max_list = []
    print(cluster[0])
    for k in range(n_clusters_):
        max_value = 0
        max_stock = ''
        for ele in (cluster[k][0]):
            print(info[ele]['vol'])
            avg_vol = float(info[ele]['vol'])
            if(avg_vol>max_value):
                max_value = avg_vol
                max_stock = info[ele]['stock']
        max_list.append(max_stock)
    print(max_list)
    context.times = context.times+1
    return 1


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n - 2] = np.nan
    return ret[:] / n


def getMA(price):
    date = 1
    MA = []
    data_sma5 = moving_average(price, 5)
    data_sma10 = moving_average(price, 10)
    data_sma15 = moving_average(price, 15)
    data_sma20 = moving_average(price, 20)
    data_sma60 = moving_average(price, 60)

    for data in price:
        if (math.isnan(data)):
            continue
        MA.append([date, data_sma5[date - 1], data_sma10[date - 1], data_sma15[date - 1], data_sma20[date - 1],
                   data_sma60[date - 1]])
        date = date + 1
    return MA


def getEMA(volume, mov_date):
    date = 1
    EMA = []
    vol_ema_prev = 0
    k = 2.0 / (mov_date + 1.0)
    for data in volume:
        if (math.isnan(data)):
            continue
        if (date < mov_date):
            vol_ema = np.nan
            EMA.append(vol_ema)
        elif (date == mov_date):
            vol_sma_ = moving_average(volume, mov_date)
            vol_sma = vol_sma_[date - 1]
            EMA.append(vol_sma)
            vol_ema_prev = vol_sma
            # print(vol_ema_prev)
        else:
            vol_ema = data * k + vol_ema_prev * (1 - k)
            EMA.append(vol_ema)
            vol_ema_prev = vol_ema
        date = date + 1
    return EMA


def getPVO(volume):
    PVO = []
    EMA_12 = getEMA(volume, 12)
    EMA_26 = getEMA(volume, 26)
    for date in range(len(volume)):
        if math.isnan(EMA_12[date]) or math.isnan(EMA_26[date]):
            PVO.append([date + 1, np.nan])
        else:
            PVO.append([date + 1, (EMA_12[date] - EMA_26[date]) / EMA_12[date] * 100])
    # print(PVO)
    return PVO


def getPM(price):
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


def getOBV(price, volume):
    date = 1
    OBV_ = []
    Volume = []
    price_prev = 0
    OBV = 0
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

        if (price_delta < 0):
            OBV = OBV - volume[date - 1]
            Volume.append(-volume[date - 1])
        elif (price_delta > 0):
            OBV = OBV + volume[date - 1]
            Volume.append(volume[date - 1])
        else:
            Volume.append(0)

        OBV_.append([date, OBV])
        date = date + 1
    """              
    plt.plot([elem for elem in range(date-1)],[elem[-1] for elem in OBV_],'b-')
    plt.title('OBV')
    plt.show()               
    plt.bar([elem for elem in range(date-1)], Volume, width= 0.8, bottom=None, align='center', data=None)
    plt.title('volume(in postive or negative state)')
    plt.show()
    """
    return OBV_


def getRSI(price):
    date = 1
    window_width = 14.0
    is_delta_WW = 1
    loss = 0
    gain = 0
    price_prev = 0
    RSI = 0
    RSI_ = []
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

            # RSI calculation
        if (is_delta_WW <= window_width):
            if (price_delta < 0):
                loss += abs(price_delta)
            elif (price_delta > 0):
                gain += price_delta
            if (is_delta_WW == window_width):
                avg_loss = loss / window_width
                avg_gain = gain / window_width

        elif (is_delta_WW > window_width):
            if (price_delta < 0):
                loss = abs(price_delta)
                gain = 0
            elif (price_delta > 0):
                loss = 0
                gain = price_delta
            avg_loss = (avg_loss * (window_width - 1) + loss) / window_width
            avg_gain = (avg_gain * (window_width - 1) + gain) / window_width
            RS = avg_gain / avg_loss
            RSI = 100 - 100 / (1 + RS)
        RSI_.append([date, RSI])
        is_delta_WW = is_delta_WW + 1
        date = date + 1

    """                
    plt.plot([elem for elem in range(date-1)],[elem[-1] for elem in RSI_],'r-')
    plt.title('RSI')
    plt.show()
    """
    return RSI_


def mergeMatrice(Matrix_A, Matrix_B):
    return np.concatenate((np.delete(Matrix_A, 0, 1), np.delete(Matrix_B, 0, 1)), axis=1)


def getTarget(price, threshold, horizon):
    date = 1
    dataset_prices = []
    price_prev = 0
    labeled_target = []
    for data in price:
        # print(data)
        if (math.isnan(data)):
            continue
        # price change: one day ratio
        dataset_prices.append(data)

    for data_price in dataset_prices:
        if (price_prev == 0):
            price_prev = data_price
            price_delta = 0
        else:
            price_next = data_price
            price_delta = price_next - price_prev
            price_prev = price_next

        if not ((date + horizon) > np.size(dataset_prices)):
            if (price_delta > 0):
                if ((dataset_prices[date - 1 + horizon] / dataset_prices[date - 1]) >= (threshold + 1)):
                    target = 1  # 1 means buy

                elif ((dataset_prices[date - 1 + horizon] / dataset_prices[date - 1]) <= (1 - threshold)):
                    target = -1  # -1 means sell

                else:
                    target = 0  # 0 means hold

            else:
                if ((dataset_prices[date - 1] / dataset_prices[date - 1 + horizon]) >= (1 + threshold)):
                    target = 1  # 1 means buy

                elif ((dataset_prices[date - 1] / dataset_prices[date - 1 + horizon]) <= (1 - threshold)):
                    target = -1  # -1 means sell

                else:
                    target = 0  # 0 means sell
        else:
            target = np.nan
        labeled_target.append([date, target])
        date = date + 1
    return labeled_target


def getTrainingWindow(prices, volume):
    # Query historical pricing data for AAPL
    date = 1

    MA_ = getMA(prices)
    PM_ = getPM(prices)
    OBV_ = getOBV(prices, volume)
    RSI_ = getRSI(prices)
    PVO_ = getPVO(volume)
    input_data_set_0 = mergeMatrice(MA_, PM_)
    input_data_set_1 = mergeMatrice(input_data_set_0, OBV_)
    input_data_set = mergeMatrice(input_data_set_1, RSI_)
    # input_data_set_2 = mergeMatrice(input_data_set_2,PVO_)

    tar = getTarget(prices, 0.015, 4)
    for data in input_data_set:
        # print(len(input_data_set))
        # print(len(tar))
        if (np.isnan(data).any() or np.isnan(tar[date - 1]).any()):
            input_data_set = np.delete(input_data_set, date - 1, 0)
            tar = np.delete(tar, date - 1, 0)
        else:
            date = date + 1
    return input_data_set, tar


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """





start = pd.to_datetime('2016-01-01').tz_localize('US/Eastern')
end = pd.to_datetime('2018-01-01').tz_localize('US/Eastern')
# Create algorithm object passing in initialize and
# handle_data functions
perf_manual = run_algorithm(start = start, end = end, capital_base = 10000000.0,  initialize=initialize, handle_data=rebalance, bundle = 'custom-na-csvdir-bundle')

# Print
#perf_manual.to_csv(SYMBOL+'_output.csv')

