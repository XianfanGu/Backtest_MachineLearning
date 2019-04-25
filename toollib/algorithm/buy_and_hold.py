from zipline.api import (history,order, record, symbol,order_target_percent,set_benchmark,set_long_only,schedule_function,sid,date_rules,time_rules)
from zipline import run_algorithm
from  zipline.api import calendars
import math
import numpy as np
# Pandas library: https://pandas.pydata.org/
import pandas as pd
stocks_bundle = 'custom-stocks-csvdir-bundle'
currency_bundle = 'custom-currency-csvdir-bundle'
SYMBOL = ""
# Called once at the start of the simulation.
def initialize(context):

    context.security = symbol(SYMBOL)  # Trade
    set_benchmark(symbol(SYMBOL))  # Set benchmarks
    #print(context.security)
    context.start = True
    set_long_only()
    schedule_function(market_open, date_rules.every_day(), time_rules.market_open(minutes=1))
    context.orders_submitted = False

def market_open(context, data):
    # Put all my money in SPY.
    if not context.orders_submitted:
        order(context.security, 10000)
        print('Initial orders submitted')
        context.orders_submitted = True
    if context.start == True:
        order_target_percent(context.security, 1.0)
        context.start = False
        print("test")
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

for ele in test_string:
    SYMBOL = ele

    perf_manual = run_algorithm(start = start, end = end, capital_base = 10000000.0,  initialize=initialize, handle_data=market_open, bundle = stocks_bundle)

    # Print
    perf_manual.to_csv('output/'+SYMBOL+'_BUY_HOLD_'+'_output.csv')