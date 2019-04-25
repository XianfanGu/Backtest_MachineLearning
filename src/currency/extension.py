import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

start = pd.Timestamp('2006-10-02 22:00:00', tz='utc')
end = pd.Timestamp('2019-04-18 20:00:00', tz='utc')

register(
    'custom-currency-csvdir-bundle',
    csvdir_equities(
        ['intraday'],
	'/home/sustechcs/Backtest_MachineLearning/csv/currency',
    ),
    calendar_name='NYSE',  # US equities
    start_session=start,
    end_session=end

)