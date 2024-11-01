

## Installation

Pre-requisites
```
sudo apt install gcc python3-pip
pip install setuptools[core] build
```

Go to folder and build module
```
cd LogAnalysis/indiflight_log_tools/
python -m build
```

Install it in your environment
```
pip install dist/indiflight_log_tools-0.1.0-cp310-cp310-linux_x86_64.whl
```

Test it 
```
cd ~/    # or anywhere else, but not the current folder
ipython -i -c "from indiflight_log_tools import IndiflightLog"
```

If no error, then it should have worked!


## Usage

```
from indiflight_log_tools import IndiflightLog

log = IndiflightLog("/path/to/log.bfl", resetTime=False)                          # resetTime maintains the flight controller time as the time basis in the dataframe (default True)
log = IndiflightLog("/path/to/log.bfl", resetTime=False, timeRange=(1000, 3000))  # crop the data to timeRange (given in ms)

log.raw                                                                           # pandas dataframe in raw units
log.data                                                                          # pandas dataframe in SI units
log.modeToText(log.data["flightModeFlags"])                                       # convert flight mode flag(s) into human-readable format
log.flags                                                                         # flight mode changes with timestamp
[(t, log.modeToText(x), log.modeToText(y)) for t, x, y in zip(log.flags["timeUs"], log.flags["enable"], log.flags["disable"])]  # flight mode changes in human readable format
```
