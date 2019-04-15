import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv


def readCSV(pathname,index_):
    n = 0
    myList = []
    with open(pathname, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if n==0:
                pos = 0
                for ele in rows:
                    if ele == index_:
                        break
                    else:
                        pos = pos +1

            if not n == 0:
                myList.append(rows[pos])
            else:
                n = 1
    return myList

def plotDataByDate(model,data,title,ylabel,xlabel,colors):
    print(len(model))

    labels = []

    fig = plt.figure()
    for tags,color in zip(model,colors):
        plt.plot(list(range(1, len(data[tags])+1)),data[tags], c=color)
        plt.suptitle(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        #plt.yticks(np.arange(min(data[tags]), max(data[tags]) + 1, 100000.0))
        labels.append(tags)

        # I'm basically just demonstrating several different legend options here...
    plt.legend(labels, ncol=4, loc='upper left',
               bbox_to_anchor=[0.045, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)
    fig.savefig('output_img/'+title+'.png')



test_list = list([['AAPL', 'ABT', 'ACN', 'ADBE', 'AAP', 'AET', 'AMG', 'ARE', 'AKAM', 'AGN', 'ADS', 'MO', 'AEE', 'AEP', 'AIG', 'AMP', 'AME', 'APH', 'ADI', 'APA', 'AMAT', 'AIZ', 'ADSK', 'AZO', 'AVB', 'BLL', 'BK', 'BAX', 'BDX', 'BRK.B', 'HRB', 'BWA', 'BSX', 'CHRW', 'COG', 'CPB', 'CAH', 'KMX', 'CAT', 'CBS', 'CNP', 'CERN', 'SCHW', 'CVX', 'CB', 'XEC', 'CTAS', 'C', 'CLX', 'CMS', 'KO', 'CTSH', 'CMCSA', 'CAG', 'ED', 'GLW', 'CCI', 'CMI', 'DHI', 'DRI', 'DE', 'XRAY', 'DISCA', 'DLTR', 'DOV', 'DTE', 'DUK', 'ETFC', 'ETN', 'EIX', 'EA', 'EMR', 'ETR', 'EQT', 'EQIX', 'ESS', 'ES', 'EXPE', 'ESRX', 'FFIV', 'FAST', 'FIS', 'FE', 'FLIR', 'FLR', 'FTI', 'BEN', 'GPS', 'GD', 'GGP', 'GPC', 'GILD', 'GT', 'GWW', 'HBI', 'HRS', 'HAS', 'HCP', 'HP', 'HPQ', 'HON', 'HST', 'HUM', 'ITW', 'INTC', 'IBM', 'IPG', 'INTU', 'IVZ', 'JEC', 'JNJ', 'JPM', 'KSU', 'KEY', 'KMB', 'KLAC', 'KR', 'LLL', 'LRCX', 'LEG', 'LUK', 'LNC', 'LMT', 'LOW', 'MTB', 'M', 'MRO', 'MAR', 'MLM', 'MA', 'MKC', 'MCK', 'MDT', 'MET', 'MCHP', 'MSFT', 'TAP', 'MON', 'MCO', 'MOS', 'MYL', 'NOV', 'NTAP', 'NWL', 'NEM', 'NEE', 'NKE', 'NBL', 'NSC', 'NOC', 'NUE', 'ORLY', 'OMC', 'ORCL', 'PCAR', 'PH', 'PAYX', 'PBCT', 'PEP', 'PRGO', 'PCG', 'PNW', 'PNC', 'PPG', 'PX', 'PCLN', 'PG', 'PLD', 'PEG', 'PHM', 'PWR', 'DGX', 'RTN', 'RHT', 'RF', 'RHI', 'COL', 'ROST', 'CRM', 'SCG', 'STX', 'SRE', 'SPG', 'SLG', 'SNA', 'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYMC', 'TROW', 'TXN', 'HSY', 'TMO', 'TWX', 'TJX', 'TSS', 'TSN', 'VLO', 'VTR', 'VZ', 'VIAB', 'VNO', 'WMT', 'DIS', 'WAT', 'WFC', 'WY', 'WMB', 'WYN', 'XEL', 'XLNX', 'YUM', 'ZION']
,['AES', 'AFL', 'APD', 'ALXN', 'ALL', 'AMZN', 'AXP', 'ABC', 'APC', 'AIV', 'T', 'AVY', 'BAC', 'BBT', 'BBY', 'BA', 'BMY', 'CA', 'COF', 'CCL', 'CELG', 'CF', 'CMG', 'CINF', 'CTXS', 'CMA', 'COP', 'STZ', 'CSX', 'DHR', 'DVN', 'D', 'EMN', 'EW', 'EOG', 'EQR', 'EXC', 'XOM', 'FDX', 'FISV', 'FMC', 'FCX', 'GRMN', 'GIS', 'GS', 'HAL', 'HIG', 'HCN', 'HD', 'HBAN', 'ICE', 'IFF', 'IRM', 'JCI', 'JNPR', 'KIM', 'KSS', 'LH', 'LEN', 'LLY', 'MAC', 'MMC', 'MAT', 'MRK', 'MU', 'MDLZ', 'MS', 'NDAQ', 'NFLX', 'NI', 'JWN', 'NRG', 'OXY', 'PDCO', 'PKI', 'PXD', 'RL', 'PFG', 'PRU', 'PVH', 'QCOM', 'O', 'RSG', 'ROK', 'SLB', 'SEE', 'SWKS', 'SO', 'SRCL', 'SYY', 'TXT', 'TIF', 'TMK', 'VAR', 'VRTX', 'VMC', 'WM', 'WDC', 'WEC', 'XRX', 'ZBH']
,['A', 'AAL', 'AMGN', 'ADM', 'BXP', 'HSIC', 'CTL', 'CI', 'CME', 'COST', 'DVA', 'EBAY', 'EFX', 'EXPD', 'FITB', 'FLS', 'GE', 'GOOGL', 'HOG', 'HES', 'IR', 'ISRG', 'K', 'MAS', 'MHK', 'MSI', 'NFX', 'NTRS', 'OKE', 'PNR', 'PFE', 'PPL', 'PGR', 'RRC', 'ROP', 'SHW', 'STI', 'TRV', 'TSCO', 'VRSN', 'WBA', 'WYNN']
,['AMT', 'AON', 'CBG', 'CSCO', 'CVS', 'EL', 'F', 'HRL', 'IP', 'MCD', 'MNST', 'NVDA', 'PSA', 'REGN', 'SJM', 'TGT', 'ANTM', 'XL'],
['ADP', 'CHK', 'JBHT'],
['CL', 'ECL', 'L'],
['LB'],
['WHR']])

"""


test_string = ['BAC', 'INTC', 'SPLS', 'PFE', 'HPQ', 'JPM', 'IPG', 'TROW']"""
model_list = ['SVM','KNeighbors','DecisionTree','RandomForest','LogisticRegression']

def getLastRow(pathname,index_):
    n = 0
    with open(pathname, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if n == 0:
                pos = 0
                for ele in rows:
                    if ele == index_:
                        n = 1
                        break
                    else:
                        pos = pos + 1
            else:
                break
        for rows in reversed(list(reader)):
            if not n == 0:
                return float(rows[pos])
    return np.nan



def daily_return(input_):
    prev = np.nan
    output_ = []
    print(input_[0])
    for ele in input_:
        if(math.isnan(prev)):
            prev = ele
            output_.append(1.0)
        else:
            latter_ = ele
            if(latter_<0 and prev >0):
                output_.append((1.0 + np.log(prev / prev-latter_)))
            elif(latter_>0 and prev <0):
                output_.append((1.0 + np.log(latter_ -prev / latter_)))
            else:
                output_.append((1.0-np.log(latter_/prev)))
            prev = latter_
    return output_

def deleteSpace(list_):
    result = []
    for ele in list_:
        if ele == '':
            continue
        else:
            result.append(ele)
    return result

"""
id = 0
for test_string in test_list:
    for model_ in model_list:
        y_ = {}
        print(model_)
        print("cluster ",id)
        index_ = 'algorithm_period_return'
        index_1 = 'benchmark_period_return'
        index_2 = 'benchmark_volatility'
        index_3 = 'algo_volatility'
        index_4 = 'sharpe'
        index_5 = 'max_drawdown'
        num = 0
        num2 = 0
        num3 = 0
        num4 = 0
        num5 = 0
        num6 = 0
        for test_ in test_string:
            pathname = 'output/' + test_ + '_' + model_ + '_output.csv'
            num = num + getLastRow(pathname,index_)
            num2 = num2 + getLastRow(pathname,index_1)
            num3 = num3 + getLastRow(pathname,index_2)
            num4 = num4 + getLastRow(pathname,index_3)
            num5 = num5 + getLastRow(pathname,index_4)
            num6 = num6 + getLastRow(pathname,index_5)
        print(index_," ",index_1," ",index_2," ",index_3," ",index_4," ",index_5)
        print(num/len(test_string), " ",num2/len(test_string), " ",num3/len(test_string), " ",num4/len(test_string), " ",num5/len(test_string), " ",num6/len(test_string), " ")
        print("\n\n\n\n")
    id = id + 1

"""



id = 0
for test_string in test_list:

    y_ = {}
    model_ = 'BUY_HOLD_'
    print('BUY_HOLD_')
    print("cluster ",id)
    index_ = 'algorithm_period_return'
    index_1 = 'benchmark_period_return'
    index_2 = 'benchmark_volatility'
    index_3 = 'algo_volatility'
    index_4 = 'sharpe'
    index_5 = 'max_drawdown'
    num = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0
    num6 = 0
    for test_ in test_string:
        pathname = 'output/' + test_ + '_' + model_ + '_output.csv'
        num = num + getLastRow(pathname,index_)
        num2 = num2 + getLastRow(pathname,index_1)
        num3 = num3 + getLastRow(pathname,index_2)
        num4 = num4 + getLastRow(pathname,index_3)
        num5 = num5 + getLastRow(pathname,index_4)
        num6 = num6 + getLastRow(pathname,index_5)
    print(index_," ",index_1," ",index_2," ",index_3," ",index_4," ",index_5)
    print(num/len(test_string), " ",num2/len(test_string), " ",num3/len(test_string), " ",num4/len(test_string), " ",num5/len(test_string), " ",num6/len(test_string), " ")
    print("\n\n\n\n")
    id = id + 1



"""
colors = cm.rainbow(np.linspace(0, 1, len(model_list)))
index_ = 'portfolio_value'
for test_ in test_string:
    y_ = {}
    for model_ in model_list:
        pathname = 'output/' + test_ + '_' + model_ + '_output.csv'
        #print(len(list(np.float_(readCSV(pathname,index_)))))
        y_[model_] = list(np.float_(readCSV(pathname,index_)))
    print(y_['SVM'])
    plotDataByDate(model_list,y_,test_,index_,'days',colors)
    # print(y_)

"""

"""
colors = cm.rainbow(np.linspace(0, 1, len(model_list)))
index_ = 'accuracy'
index_1 = 'algorithm_period_return'
Y={}
X={}
for model_ in model_list:
    y_ = {}
    index_list = []
    index_list_1 = []
    for test_ in test_list[1]:
        pathname = 'output/' + test_ + '_' + model_ + '_output.csv'
       # print(len(deleteSpace(readCSV(pathname,index_))))
        size_ = len(list(np.float_(deleteSpace(readCSV(pathname,index_)))))
        size_1 = len(list(np.float_(deleteSpace(readCSV(pathname,index_1)))))
        index_list.append(list(np.float_(deleteSpace(readCSV(pathname,index_)))))
        index_list_1.append(list(np.float_(deleteSpace(readCSV(pathname,index_1))))[(size_1-size_):])
    x = []
    y = []
    for i in range(len(index_list[0])):
        sum_acc = 0
        sum_ret = 0
        for ele,ele1 in zip(index_list, index_list_1):
            sum_acc = sum_acc + ele[i]
            sum_ret = sum_ret + ele1[i]
        #print(len(index_list),len(index_list_1))
        avg_acc = sum_acc/len(index_list)
        avg_ret = sum_ret/len(index_list_1)
        print(avg_acc)
        x.append(avg_acc)
        y.append(avg_ret)
    Y[model_] = y
    X[model_] = x

fig=plt.figure()
cmap = plt.get_cmap('jet_r')
colors = cm.rainbow(np.linspace(0, 1, len(model_list)))
for model_,color in zip(model_list,colors):
    plt.scatter(X[model_], Y[model_], c=color, marker='.', s=40)
    plt.ylabel('period return')
    plt.xlabel('accuracy')

plt.legend(model_list, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)
plt.show()
fig.savefig('output_img/1.png')
"""
#print(Y)
    #plotDataByDate(model_list,y_,test_,index_,'days',colors)
    # print(y_)