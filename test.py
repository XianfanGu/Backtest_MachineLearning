"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from zipline.api import (history,order, record, symbol,order_target_percent,set_benchmark,set_long_only,schedule_function,sid,date_rules,time_rules)
from zipline import run_algorithm
import os.path
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
import urllib
from io import StringIO
import csv
import datetime as dt
MODEL_NAME = ''
SYMBOL = ''
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    feature_num = 11
    context.orders_submitted = False
    large_num = 9999999
    least_num = 0
    context.n_components = 6
    context.security = symbol(SYMBOL)  # Trade SPY
    set_benchmark(symbol(SYMBOL))  # Set benchmarks
    context.model2 = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2, C=10.0, verbose=True)  # 8.05 for SVM model
    context.model3 = KNeighborsClassifier(n_neighbors=feature_num, p=3, metric='minkowski')  # 7.05 for  model
    context.model = DecisionTreeClassifier(criterion='entropy', max_depth=feature_num, random_state=0)
    context.model4 = RandomForestClassifier(criterion='entropy', n_estimators=feature_num, random_state=1,
                                            n_jobs=2)  # 5.2 for randomforest
    context.model1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    context.modellist = {'SVM':context.model2,'KNeighbors':context.model3,'DecisionTree':context.model,'RandomForest':context.model4,'LogisticRegression':context.model1}
    context.lookback = 350  # Look back 62 days
    context.history_range = 350  # Only consider the past 400 days' history
    context.threshold = 4.05
    context.longprices = large_num
    context.shortprices = least_num
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
    model = context.modellist[MODEL_NAME]
    recent_prices = data.history(context.security, 'price', context.history_range, '1d').values
    recent_volume = data.history(context.security, 'volume', context.history_range, '1d').values
    train_, target_ = getTrainingWindow(recent_prices, recent_volume)
    X_normalized_ = preprocessing.normalize(train_, norm='l2')
    y = np.delete(target_, 0, 1)
    y = np.ravel(y)
    sc = StandardScaler()
    sc.fit(X_normalized_)
    X_std = sc.transform(X_normalized_)
    X_train, y_train = X_std, y

    model.fit(X_train, y_train)


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
    # Get recent prices
    model = context.modellist[MODEL_NAME]
    recent_prices = data.history(context.security, 'price', context.lookback + 1, '1d').values
    recent_volume = data.history(context.security, 'volume', context.lookback + 1, '1d').values
    test_, _ = getTrainingWindow(recent_prices, recent_volume)
    X_normalized_ = preprocessing.normalize(test_, norm='l2')
    sc = StandardScaler()
    sc.fit(X_normalized_)
    X_std = sc.transform(X_normalized_)
    X_test = X_std[-1, :]
    for stock in context.portfolio.positions:
        print(context.portfolio.positions[stock].amount)
    if not context.orders_submitted:
        order(context.security, 10000)
        print('Initial orders submitted')
        context.orders_submitted = True
    try:
        if model:  # Check if our model is generated
            # Predict using our model and the recent prices
            X_test_ = X_test.reshape(1, -1)
            prediction = model.predict(X_test_)
            prediction_accuracy = model.predict(X_std[-10:, :])  # predict in past 10 days
            accuracy = accuracy_score(np.ravel(np.delete(_, 0, 1))[-10:], prediction_accuracy)
            print('Accuracy: %.2f' % accuracy)
            record(accuracy=accuracy)
            # print(prediction," x_test: ",X_test)
            record(prediction=prediction)
            decision_order = prediction[0]
            order_target_percent(context.security, decision_order)
    except Exception as error:
        print('Caught this error: ' + repr(error))

    """        
    if context.model1: # Check if our model is generated

        # Predict using our model and the recent prices
        prediction = context.model1.predict(X_test)
        record(prediction = prediction)

        # Go long if we predict the price will rise, short otherwise
        if prediction == 1:
            order_target_percent(context.security, 1.0)
        elif prediction == 0:
            order_target_percent(context.security, 0.0)
        else:
            order_target_percent(context.security, -1.0)
    """

"""
QUERY_URL_JSON = "https://www.alphavantage.co/query?function={REQUEST_TYPE}&outputsize=full&datatype=csv&apikey={KEY}&symbol={SYMBOL}"
API_KEY = "VKNYIAEYDFJGF1RS"
def _request_csv(symbol, req_type):
    with urllib.request.urlopen(QUERY_URL_JSON.format(REQUEST_TYPE=req_type, KEY=API_KEY, SYMBOL=symbol)) as req:
        data = req.read().decode('utf-8')
    return data

def get_daily_csv_data(symbol):
    return _request_csv(symbol, 'TIME_SERIES_DAILY')

start = pd.to_datetime('2016-01-01').tz_localize('US/Eastern')
end = pd.to_datetime('2018-01-01').tz_localize('US/Eastern')
# getting amd data from alpha vantage via api
csv_= get_daily_csv_data('SPY')
print(type(csv_))
dataframe_ = pd.read_csv(StringIO(csv_))
print(dataframe_)
"""

test_string = ['AAPL', 'ABT', 'ACN', 'ADBE', 'AAP', 'AET', 'AMG', 'ARE', 'AKAM', 'AGN', 'ADS', 'MO', 'AEE', 'AEP', 'AIG',
                   'AMP', 'AME', 'APH', 'ADI', 'APA', 'AMAT', 'AIZ', 'ADSK', 'AZO', 'AVB', 'BLL', 'BK', 'BAX', 'BDX', 'BRK.B',
                   'HRB', 'BWA', 'BSX', 'CHRW', 'COG', 'CPB', 'CAH', 'KMX', 'CAT', 'CBS', 'CNP', 'CERN', 'SCHW', 'CVX', 'CB', 'XEC',
                   'CTAS', 'C', 'CLX', 'CMS', 'KO', 'CTSH', 'CMCSA', 'CAG', 'ED', 'GLW', 'CCI', 'CMI', 'DHI', 'DRI', 'DE', 'XRAY',
                   'DISCA', 'DLTR', 'DOV', 'DTE', 'DUK', 'ETFC', 'ETN', 'EIX', 'EA', 'EMR', 'ETR', 'EQT', 'EQIX', 'ESS', 'ES', 'EXPE',
                   'ESRX', 'FFIV', 'FAST', 'FIS', 'FE', 'FLIR', 'FLR', 'FTI', 'BEN', 'GPS', 'GD', 'GGP', 'GPC', 'GILD', 'GT', 'GWW', 'HBI',
                   'HRS', 'HAS', 'HCP', 'HP', 'HPQ', 'HON', 'HST', 'HUM', 'ITW', 'INTC', 'IBM', 'IPG', 'INTU', 'IVZ', 'JEC', 'JNJ', 'JPM',
                   'KSU', 'KEY', 'KMB', 'KLAC', 'KR', 'LLL', 'LRCX', 'LEG', 'LUK', 'LNC', 'LMT', 'LOW', 'MTB', 'M', 'MRO', 'MAR', 'MLM', 'MA',
                   'MKC', 'MCK', 'MDT', 'MET', 'MCHP', 'MSFT', 'TAP', 'MON', 'MCO', 'MOS', 'MYL', 'NOV', 'NTAP', 'NWL', 'NEM', 'NEE', 'NKE', 'NBL',
                   'NSC', 'NOC', 'NUE', 'ORLY', 'OMC', 'ORCL', 'PCAR', 'PH', 'PAYX', 'PBCT', 'PEP', 'PRGO', 'PCG', 'PNW', 'PNC', 'PPG', 'PX', 'PCLN',
                   'PG', 'PLD', 'PEG', 'PHM', 'PWR', 'DGX', 'RTN', 'RHT', 'RF', 'RHI', 'COL', 'ROST', 'CRM', 'SCG', 'STX', 'SRE', 'SPG', 'SLG', 'SNA',
                   'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYMC', 'TROW', 'TXN', 'HSY', 'TMO', 'TWX', 'TJX', 'TSS', 'TSN', 'VLO', 'VTR', 'VZ', 'VIAB', 'VNO',
                   'WMT', 'DIS', 'WAT', 'WFC', 'WY', 'WMB', 'WYN', 'XEL', 'XLNX', 'YUM', 'ZION',
                   'AES', 'AFL', 'APD', 'ALXN', 'ALL', 'AMZN', 'AXP', 'ABC', 'APC', 'AIV', 'T', 'AVY', 'BAC', 'BBT', 'BBY', 'BA',
                   'BMY', 'CA', 'COF', 'CCL', 'CELG', 'CF', 'CMG', 'CINF', 'CTXS', 'CMA', 'COP', 'STZ', 'CSX', 'DHR', 'DVN', 'D',
                   'EMN', 'EW', 'EOG', 'EQR', 'EXC', 'XOM', 'FDX', 'FISV', 'FMC', 'FCX', 'GRMN', 'GIS', 'GS', 'HAL', 'HIG', 'HCN',
                   'HD', 'HBAN', 'ICE', 'IFF', 'IRM', 'JCI', 'JNPR', 'KIM', 'KSS', 'LH', 'LEN', 'LLY', 'MAC', 'MMC', 'MAT', 'MRK',
                   'MU', 'MDLZ', 'MS', 'NDAQ', 'NFLX', 'NI', 'JWN', 'NRG', 'OXY', 'PDCO', 'PKI', 'PXD', 'RL', 'PFG', 'PRU', 'PVH',
                   'QCOM', 'O', 'RSG', 'ROK', 'SLB', 'SEE', 'SWKS', 'SO', 'SRCL', 'SYY', 'TXT', 'TIF', 'TMK', 'VAR', 'VRTX', 'VMC',
                   'WM', 'WDC', 'WEC', 'XRX', 'ZBH',
                   'A', 'AAL', 'AMGN', 'ADM', 'BXP', 'HSIC', 'CTL', 'CI', 'CME', 'COST', 'DVA', 'EBAY', 'EFX', 'EXPD', 'FITB', 'FLS', 'GE', 'GOOGL',
                   'HOG', 'HES', 'IR', 'ISRG', 'K', 'MAS', 'MHK', 'MSI', 'NFX', 'NTRS', 'OKE', 'PNR', 'PFE', 'PPL', 'PGR', 'RRC', 'ROP', 'SHW', 'STI', 'TRV', 'TSCO', 'VRSN', 'WBA', 'WYNN',
                    'AMT', 'AON', 'CBG', 'CSCO', 'CVS', 'EL', 'F', 'HRL', 'IP', 'MCD', 'MNST', 'NVDA', 'PSA', 'REGN', 'SJM', 'TGT', 'ANTM', 'XL',
                    'ADP', 'CHK', 'JBHT','CL', 'ECL', 'L','LB','WHR'
                   ]


start = pd.to_datetime('2014-01-01').tz_localize('US/Eastern')
end = pd.to_datetime('2017-01-01').tz_localize('US/Eastern')
# Create algorithm object passing in initialize and
# handle_data functions
#['BAC', 'GNW', 'IPG', 'HOG', 'JPM', 'HCN', 'KSS', 'MDLZ']
#['BAC', 'INTC', 'SPLS', 'PFE', 'HPQ', 'JPM', 'IPG', 'TROW']


#test_string = ['HPQ']
model_list = ['SVM','KNeighbors','DecisionTree','RandomForest','LogisticRegression']
for ele in test_string:
    SYMBOL = ele
    if(os.path.isfile(('output_csv/'+SYMBOL+'_SVM_output.csv'))):
        print('output_csv/'+SYMBOL+'_SVM_output.csv is exist')
        continue
    else:
        print('output_csv/' + SYMBOL + '_SVM_output.csv is not exist')
    for model_name in model_list:
        MODEL_NAME = model_name
        perf_manual = run_algorithm(start = start, end = end, capital_base = 10000000.0,  initialize=initialize, handle_data=rebalance, bundle = 'custom-na-csvdir-bundle')

        # Print
        perf_manual.to_csv('output_csv/'+SYMBOL+'_'+MODEL_NAME+'_output.csv')




