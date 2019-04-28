import urllib
import urllib.request
import os
import csv
import progressbar
from time import sleep
from datetime import datetime

DT_FORMAT = "%Y-%m-%d %hh:%mm:%ss"
min_size = 5591
max_size = 90591
QUERY_URL_CSV = "https://www.alphavantage.co/query?function={REQUEST_TYPE}&from_symbol={FROM_SYM}&to_symbol={TO_SYM}&interval={TIME}&outputsize=full&datatype=csv&apikey={KEY}"
API_KEY = "VKNYIAEYDFJGF1RS"
root_path = 'csv/currency/minute/'
def _request_csv(req_type, from_symbol, to_symbol, interval):
    try:
        urllib.request.urlretrieve(QUERY_URL_CSV.format(REQUEST_TYPE=req_type, KEY=API_KEY, FROM_SYM = from_symbol, TO_SYM = to_symbol, TIME = interval), root_path +from_symbol+to_symbol+'.csv')
    except urllib.error.HTTPError as ex:
        print('Problem:', ex)



def format_file(symbol):
    body = []
    try:
        with open(root_path+symbol+'.csv', "r") as inp:
            #START_DATE = datetime.strptime("2019-04-02 22:00:00", DT_FORMAT)
            #END_DATE = datetime.strptime("2019-04-18 20:00:00", DT_FORMAT)
            #print(START_DATE)

            inp.seek(0)
            lastrow = None
            n = 0
            for lastrow in csv.reader(inp):
                if(n==0):
                    firstrow = lastrow
                n = n + 1
            print(firstrow)
            if (firstrow[0] != 'timestamp'):
                os.remove(root_path + symbol + '.csv')
                print("File Removed!", '  ' + root_path + symbol + '.csv')
                print('Caught this error: format is not correct')
                return False

            inp.seek(0)
            for row in csv.reader(inp):
                if(row[0]=='timestamp'):
                    row.append('volume')
                    body.append(row)
                    continue
                row.append(0.0)
                body.append(row)

        with open(root_path+symbol+'.csv', "w") as out:
            writer = csv.writer(out)
            for row in body:
                writer.writerow(row)
        return True

    except Exception as error:
        pass

    os.remove(root_path + symbol + '.csv')
    print("File Removed!", '  '+ root_path + symbol + '.csv')
    print('Caught this error: format is not correct')
    return False

def checkFileSize(symbol):
    try:
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        statinfo = os.path.getsize(root_path + symbol +'.csv')
        print(root_path + symbol +'.csv :'+statinfo.__str__())
        if(statinfo <= min_size) or (statinfo >= max_size):
            return False
        else:
            return True
    except Exception as error:
        pass
    print("this file is not exist: " + root_path + symbol + '.csv')
    return False



def download():
    currency_symbol = ['EUR','CHF','GBP','USD','AUD','JPY','NZD']
    bar = progressbar.ProgressBar(maxval=(len(currency_symbol)*(len(currency_symbol)-1)), \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    try:
        i = 0
        bar.start()
        for from_symbol in currency_symbol:
            for to_symbol in currency_symbol:
                if(to_symbol==from_symbol):
                    continue
                bar.update(i + 1)
                sleep(0.1)
                if not checkFileSize(from_symbol+to_symbol):
                    _request_csv('FX_INTRADAY',from_symbol, to_symbol, '1min')
                    format_file(from_symbol+to_symbol)
                else:
                    i = i + 1
                    continue
                i = i + 1
        bar.finish()
    except Exception as error:
        print('Caught this error1: ' + repr(error))

