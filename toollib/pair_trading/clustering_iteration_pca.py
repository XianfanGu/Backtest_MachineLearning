"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from zipline.api import (history,order, record, symbol,order_target_percent,set_benchmark,set_long_only,schedule_function,sid,date_rules,time_rules)
from zipline import run_algorithm
import math
import csv
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
from talib import RSI, BBANDS, OBV, EMA, MA, MACD,STOCH, CCI, AD
import matplotlib.pyplot as plt
import numpy as np
import difflib as dif
import urllib
from io import StringIO
import csv
import datetime as dt
N_PRIN_COMPONENTS = 7
SYMBOL = 'SPY'
pathname = 'doc/constituents.csv'
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    feature_num = 14
    context.orders_submitted = False
    large_num = 9999999
    least_num = 0
    context.n_components = 11
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
    #context.model = KMeans(n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=0)
    context.model = DBSCAN(eps=0.2,min_samples=3,metric='euclidean')
    context.sector = readCSV(pathname)
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
    label = []
    info = []
    id = 0
    price_stocks_list = {}
    sectors = iterateCreateID(context.sector)
    for symbol_ in context.SP500_symbol:
        try:
            recent_prices = data.history(symbol(symbol_), 'price', context.history_range, '1d').values
            recent_volume = data.history(symbol(symbol_), 'volume', context.history_range, '1d').values
            recent_high = data.history(symbol(symbol_), 'high', context.history_range, '1d').values
            recent_low = data.history(symbol(symbol_), 'low', context.history_range, '1d').values
            info.append({'vol': sum(recent_volume) / context.history_range, 'stock': symbol_})
            price_stocks_list[symbol_] = recent_prices
	#get the moving training window
            train_, target_ = getTrainingWindow(recent_high,recent_low,recent_prices, recent_volume, sectors[symbol_])
            X_normalized_ = preprocessing.normalize(train_, norm='l2')
            y = np.delete(target_, 0, 1)
            y = np.ravel(y)
            sc = StandardScaler()
            sc.fit(X_normalized_)
            X_std = sc.transform(X_normalized_)
	# PCA to input features (N_PRIN_COMPONENTS)
            pca = PCA(n_components=N_PRIN_COMPONENTS)
            X_pca = pca.fit_transform(X_std)
            X_train, y_train = X_pca, y
            X[symbol_] = []
            X[symbol_].append(X_train)
            label.append(symbol_)

        except Exception as error:
            pass
        id = id + 1

    #print(info[0]['stock'])
    stock_cluster_id = {}
    print("feature number:", train_.shape[1])
    print("PCA feature number:", X_train.shape[1])
#transfer 300 days 3D matrx to 2D matrix by merge days axis with PCA
    for i in range(len(y_train)):
        train_1 = []
        for symbol_ in context.SP500_symbol:
            try:
                X_perday = (X[symbol_][0][i])
                #print(X_pca_per_stock.T[0])
                if(i == 0):
                    stock_cluster_id[symbol_] = []
                train_1.append(X_perday)
            except Exception as error:
                pass
        #print(train_1)
        #X_normalized_1 = preprocessing.normalize(train_1, norm='l2')
        #sc = StandardScaler()
        #sc.fit(X_normalized_1)
        #X_std_1 = sc.transform(X_normalized_1)
        X_std_1 = train_1
        #print(X_std_1)
# DBScan clustering model
        y_pred = context.model.fit_predict(X_std_1)
        labels = context.model.labels_
        n_clusters_ = len(set(labels))
        date_i = 0
        for symbol_ in context.SP500_symbol:
            try:
                stock_cluster_id[symbol_].append(y_pred[date_i])
                date_i = date_i + 1
            except Exception as error:
                pass

        print("Clusters discovered: ", n_clusters_)
    symbol_cluster_id = {}
    label_list_2 = []
    label_list_2_id = 0
    new_clusters = cluster_base_simlilarity(context, stock_cluster_id, 0.55)
    #print(len(stock_cluster_id['AAPL']),len(y_train))
# iteration of clustering and create the lists of id of clusters in every clustering
    i = 0
    for cluster_ in new_clusters:
        label_list_2.append(label_list_2_id)
        print(cluster_)
        i = i + 1
        for ele in cluster_:
            symbol_cluster_id[ele] = label_list_2_id
        label_list_2_id = label_list_2_id + 1
    n_clusters_2 = i
    print("Iterated Clusters discovered: ", n_clusters_2)
    #print(dif.SequenceMatcher(None,stock_cluster_id['AAPL'],stock_cluster_id['DIS']).ratio())
    #
    y_pred_2 = []
    for symbol_ in context.SP500_symbol:
        try:
            y_pred_2.append(symbol_cluster_id[symbol_])
        except Exception as error:
            pass
    y_pred_2 = np.array(y_pred_2)

# list the label of clusters
    label_list = unique(y_pred)
    print(len(y_pred))
    print(label_list)
    pca = PCA(n_components=2)
    cluster = []
    X_axis = pca.fit_transform(X_std_1)
    X_axis = np.array(X_axis)
#scatter plot iteration of clustering
    for y_i in label_list:
        color = np.random.rand(3, )
        print(color)
        plt.scatter(X_axis[y_pred == y_i,0], X_axis[y_pred == y_i,1], c=color, marker='o', s=40)
        plt.suptitle('the number of clusters: '+str(n_clusters_))
        plt.xlabel('pca_feature_1')
        plt.ylabel('pca_feature_2')
        cluster.append(np.where(y_pred==y_i))

    for j in range(0,len(y_pred),20):
        plt.annotate(
            label[j],
            xy=(X_axis[j,0], X_axis[j,1]), xytext=(-0.2, 0.2),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    plt.show()
    print(label_list_2)
    print(len(y_pred_2))
    cluster2 = []
#scatter plot pca clustering
    for y_i in label_list_2:
        color = np.random.rand(3, )
        print(color)
        #print(X_axis[y_pred_2 == y_i,0])
        plt.scatter(X_axis[y_pred_2 == y_i,0], X_axis[y_pred_2 == y_i,1], c=color, marker='o', s=40)
        plt.suptitle('the number of clusters: '+str(n_clusters_2))
        plt.xlabel('pca_feature_1')
        plt.ylabel('pca_feature_2')
        cluster2.append(np.where(y_pred_2==y_i))

    for j in range(0,len(y_pred),20):
        plt.annotate(
            label[j],
            xy=(X_axis[j,0], X_axis[j,1]), xytext=(-0.2, 0.2),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    plt.show()

    max_list = getMaxVolumeInClusters(n_clusters_, cluster, info)
    print(max_list)


    max_iterate_list = getMaxVolumeInClusters(n_clusters_2, cluster2, info)
    print(max_iterate_list)

    #plotDataByDate(new_clusters,price_stocks_list,context.history_range)
    
    context.times = context.times + 1
    return 1

#create dictionary for sectors, read sectors from csv
def readCSV(pathname):
    n = 0
    mydict = {}
    with open(pathname, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if not n == 0:
                mydict[rows[0]] = rows[2]
            else:
                n = 1
    return mydict
def iterateCreateID(namelist):
    idDict= {}
    valDict = {}
    id = 0
    for key, value in namelist.items():
        if value in valDict:
            idDict[key] = valDict[value]
        else:
            valDict[value] = id
            idDict[key] = id
            id = id + 1
    return idDict

def plotDataByDate(cluster_tags,data,time_range):
    print(len(cluster_tags))
    cmap = plt.get_cmap('jet_r')
    N = len(cluster_tags)
    num = 0
    labels = []
    MAX_ITERATE = 6
    j = 0

    for tags in cluster_tags:
        color = cmap(float(num)/N)
        num = num + 2
        range_num = np.random.randint(-300,300)
        k = 0

        for i in tags:
            if(k > MAX_ITERATE):
                break
            if(k==0):
                rand_ = np.random.randint(0, len(data[i]))
                list_range = [(range_num - rand_) for sort in range(len(data[i]))]
            data[i] = list_range+data[i]
            plt.plot(list(range(1, time_range+1)),data[i], c=color)
            labels.append(i)
            k = k + 1
        # I'm basically just demonstrating several different legend options here...
    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)

    plt.show()

def getMaxVolumeInClusters(n_clusters, stocks_cluster_pos, info_):
    stocks_list = []
    for k in range(n_clusters):
        max_value = 0
        max_stock = ''
        for ele in (stocks_cluster_pos[k][0]):
            #print(info[ele]['vol'])
            avg_vol = float(info_[ele]['vol'])
            if(avg_vol>max_value):
                max_value = avg_vol
                max_stock = info_[ele]['stock']
        stocks_list.append(max_stock)
    return stocks_list

def cluster_base_simlilarity(context,stocks_cluster_id, threshold):
    clusters = []
    symbol_list = context.SP500_symbol.copy()
    symbol_list_1 = context.SP500_symbol.copy()
    for symbol_ in symbol_list:
        try:
            cluster_row = []
            factor_1 = stocks_cluster_id[symbol_]
            factor_1.sort()
            cluster_row.append(symbol_)
            #print(factor_1)
            for symbol_2 in symbol_list_1:
                try:
                    if(symbol_2 == symbol_):
                        continue
                    factor_2 = stocks_cluster_id[symbol_2]
                    factor_2.sort()
                    ratio = dif.SequenceMatcher(None,factor_1,factor_2).ratio()
                    #print(ratio)
                    if(ratio >= threshold):
                        cluster_row.append(symbol_2)
                        symbol_list_1.remove(symbol_2)
                        symbol_list.remove(symbol_2)
                except Exception as error:
                    pass
            clusters.append(cluster_row)
            symbol_list_1.remove(symbol_)
        except Exception as error:
            pass
    print(clusters)
    return clusters

def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def getMA(price):
    date = 1
    ma = []
    data_sma5 = np.array(MA(price, 5))
    data_sma10 = np.array(MA(price, 10))
    data_sma15 = np.array(MA(price, 15))
    data_sma20 = np.array(MA(price, 20))
    data_sma60 = np.array(MA(price, 60))

    for data in price:
        if (math.isnan(data)):
            continue
        ma.append([date, data_sma5[date - 1], data_sma10[date - 1], data_sma15[date - 1], data_sma20[date - 1],
                   data_sma60[date - 1]])
        date = date + 1
    return ma


def getEMA(volume, mov_date):
    real = EMA(volume, mov_date)
    return real


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

def getSTOCH(high,low,price):
    date = 1
    STOCH_ = []
    fastk, fastd = STOCH(high,low,price)
    for ele, ele1 in zip(fastk, fastd):
        STOCH_.append([date, ele, ele1])
        date = date + 1
    return STOCH_

def getMACD(price):
    date = 1
    MACD_ = []
    macd, macdsignal, macdhist = MACD(price)
    for ele,ele1,ele2 in zip(macd, macdsignal, macdhist):
        MACD_.append([date, ele,ele1,ele2])
        date = date + 1
    return MACD_

def getCCI(high,low,price):
    date = 1
    CCI_ = []
    real = CCI(high, low, price)
    for ele in real:
        CCI_.append([date, ele])
        date = date + 1
    return CCI_

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

def getAD(high,low,price,volume):
    date = 1
    AD_ = []
    real = AD(high, low, price, volume)
    for ele in real:
        AD_.append([date, ele])
        date = date + 1
    return AD_

def getOBV(price, volume):
    date = 1
    OBV_ = []
    obv = OBV(price, volume)
    for ele in obv:
        OBV_.append([date, ele])
        date = date + 1
    return OBV_


def getRSI(price):
    date = 1
    RSI_ = []
    rsi = RSI(price, timeperiod=14)
    for ele in rsi:
        RSI_.append([date, ele])
        date = date + 1
    return RSI_

def getSector(length, sector):
    date = 1
    Sector_ = []
    for i in range(length):
        Sector_.append([date, sector])
        date = date + 1
    return Sector_
#merge two matrice to one in x axis
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


def getTrainingWindow(high, low, prices, volume, sector):
    # Query historical pricing data
    date = 1
    MA_ = getMA(prices)
    PM_ = getPM(prices)
    OBV_ = getOBV(prices, volume)
    RSI_ = getRSI(prices)
    PVO_ = getPVO(volume)
    STOCH_ = getSTOCH(high, low, prices)
    MACD_ = getMACD(prices)
    CCI_ = getCCI(high, low, prices)
    AD_ = getAD(high, low, prices,volume)


    input_data_set_0 = mergeMatrice(MA_, PM_)
    input_data_set_1 = mergeMatrice(input_data_set_0, OBV_)
    input_data_set_6 = mergeMatrice(input_data_set_1, RSI_)
    input_data_set_2 = mergeMatrice(input_data_set_6,PVO_)
    input_data_set_3 = mergeMatrice(input_data_set_2, STOCH_)
    input_data_set_4 = mergeMatrice(input_data_set_3, MACD_)
    input_data_set_5 = mergeMatrice(input_data_set_4, CCI_)
    input_data_set = mergeMatrice(input_data_set_5, AD_)

    tar = getTarget(prices, 0.015, 4)
    for data in input_data_set:
        # print(len(input_data_set))
        # print(len(tar))
        if (np.isnan(data).any() or np.isnan(tar[date - 1]).any()):
            input_data_set = np.delete(input_data_set, date - 1, 0)
            tar = np.delete(tar, date - 1, 0)
        else:
            date = date + 1
    Sector_ = getSector(len(input_data_set), sector)
    input_data_set = mergeMatrice(input_data_set, Sector_)
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

