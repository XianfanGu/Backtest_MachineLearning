import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

start = pd.Timestamp('2006-10-02', tz='utc')
end = pd.Timestamp('2019-04-18', tz='utc')

register(
    'custom-na-csvdir-bundle',
    csvdir_equities(
        ['daily'],
	'/home/sustechcs/Backtest_MachineLearning/csv',
    ),
    calendar_name='NYSE',  # US equities
    start_session=start,
    end_session=end

)