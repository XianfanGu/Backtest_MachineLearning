import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities




register(
    'custom-currency-csvdir-bundle',
    csvdir_equities(
        ['minute'],
	'/home/sustechcs/test/Backtest_MachineLearning/csv/currency',
    ),
	calendar_name='CME',  # US equities

)