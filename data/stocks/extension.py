import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities




register(
    'custom-stocks-csvdir-bundle',
    csvdir_equities(
        ['daily'],
	'/home/sustechcs/test/Backtest_MachineLearning/csv/stocks',
    ),
    calendar_name='NYSE',  # US equities

)