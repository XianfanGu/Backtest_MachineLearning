import urllib
import urllib.request
import os
import csv
import progressbar
from time import sleep
from datetime import datetime
from toollib.Data.info import Info


SnP500 = Info(pathname='data/constituents.csv').get_symbol_list()
ASX200 = Info(pathname='data/asx200.csv').get_symbol_list()

DT_FORMAT = "%Y-%m-%d"
min_size = 200000
max_size = 300000

QUERY_URL_CSV = "https://www.alphavantage.co/query?function={REQUEST_TYPE}&outputsize=full&datatype=csv&apikey={KEY}&symbol={SYMBOL}"
API_KEY = "VKNYIAEYDFJGF1RS"
root_path = 'csv/stocks/daily/'

def _request_csv(symbol, req_type):
    try:
        urllib.request.urlretrieve(QUERY_URL_CSV.format(REQUEST_TYPE=req_type, KEY=API_KEY, SYMBOL=symbol), root_path+symbol+'.csv')
    except urllib.error.HTTPError as ex:
        print('Problem:', ex)



def format_file(symbol):
    body = []
    try:
        with open(root_path+symbol+'.csv', "r") as inp:
            START_DATE = datetime.strptime("2006-10-01", DT_FORMAT)
            END_DATE = datetime.strptime("2019-04-18", DT_FORMAT)
            #print(START_DATE)

            inp.seek(0)
            lastrow = None
            n = 0
            for lastrow in csv.reader(inp):
                if(n==1):
                    firstrow = lastrow
                n = n + 1
            print(firstrow)
            if(datetime.strptime(lastrow[0], DT_FORMAT)>START_DATE or datetime.strptime(firstrow[0], DT_FORMAT)<END_DATE):
                os.remove(root_path+symbol+'.csv')
                print("File Removed!",'  '+root_path+symbol+'.csv')
                return False
            inp.seek(0)
            for row in csv.reader(inp):
                if(row[0]=='timestamp'):
                    body.append(row)
                    continue
                DATE = datetime.strptime(row[0], DT_FORMAT)
                #print(DATE)
                if(DATE >= START_DATE):
                    body.append(row)

        with open(root_path+symbol+'.csv', "w") as out:
            writer = csv.writer(out)
            for row in body:
                writer.writerow(row)
        return True

    except Exception as error:
        pass

    os.remove(root_path + symbol + '.csv')
    print("File Removed!", '  '+root_path + symbol + '.csv')
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
    bar = progressbar.ProgressBar(maxval=len(SnP500), \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    try:
        i = 0
        bar.start()
        for symbol in SnP500:
            bar.update(i + 1)
            sleep(0.1)
            if not checkFileSize(symbol):
                _request_csv(symbol,'TIME_SERIES_DAILY_ADJUSTED')
                format_file(symbol)
            else:
                i = i + 1
                continue
            i = i + 1
        bar.finish()
    except Exception as error:
        print('Caught this error1: ' + repr(error))

