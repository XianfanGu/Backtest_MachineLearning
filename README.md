# Backtest_MachineLearning
tool in adavanced financial analysis for the stocks prediction

the project is base on Zipline backtest platform, see the Zipline: http://www.zipline.io/
, Sklearn library: https://scikit-learn.org/
, and TA library: https://www.ta-lib.org/

### Step 1: Install Package  ###

`python setup.py install && export PYTHONPATH='pwd'`

### Step 2: Download Data(csv file)  ###
 
`python3 toollib/Download/download_csv.py`
 
### Step 3: Create Data Bundle  ###
 
instruction: http://www.zipline.io/bundles.html
  
`gedit ~/.zipline/extension.py`

copy these codes to extension.py (remember to change your path)

```
import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities


start = pd.Timestamp('2006-10-02', tz='utc')
end = pd.Timestamp('2019-3-25', tz='utc')

register(
    'custom-na-csvdir-bundle',
    csvdir_equities(
        ['daily'],
        '/home/sustechcs/Backtest_MachineLearning/csv',
    ),
    calendar_name='NYSE', # US equities
    start_session=start,
    end_session=end
    
)
```


run this command line in terminal
`zipline ingest -b custom-na-csvdir-bundle`

### Step 3: Run Backtest  ###
`python3 toollib/Backtest/test.py`