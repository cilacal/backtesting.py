import pandas as pd
import numpy as np
import warnings

from functools import partial
from typing import Optional, Type

from backtesting import Strategy
from backtesting import _Broker
from ._util import _Indicator, _Data

class Monitor:
    """
    Monitor the market using a particular (parameterized) strategy
    on particular data.

    Upon initialization, call method
    `backtesting.monitoring.Monitor.run` to run a monitoring
    instance.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy], 
                 window: int = 1):
        """
        Initialize a monitoring. Requires data and a strategy to monitor.
        Using the parameter `window` you can specify the number of last candles to consider for the strategy. 

        `data` is a `pd.DataFrame` with columns:
        `Open`, `High`, `Low`, `Close`, and (optionally) `Volume`.
        If any columns are missing, set them to what you have available,
        e.g.

            df['Open'] = df['High'] = df['Low'] = df['Close']

        The passed data frame can contain additional columns that
        can be used by the strategy (e.g. sentiment info).
        DataFrame index can be either a datetime index (timestamps)
        or a monotonic range index (i.e. a sequence of periods).

        `strategy` is a `backtesting.backtesting.Strategy`
        _subclass_ (not an instance).

        The following parameters to initialize the _Broker were defaulted as follows:
         - `cash`: 9999
         - `commission`: 0
         - `margin`: 1.
         - `trade_on_close`: False (following the default value of the backtesting.Backtest)
         - `hedging`: False
         - `exclusive_orders`: False
        Broker is needed to initialize the strategy, even though it is not used for monitoring.
        """

        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")

        data = data.copy(deep=False)

        # Convert index to datetime index
        if (not isinstance(data.index, pd.DatetimeIndex) and
            not isinstance(data.index, pd.RangeIndex) and
            # Numeric index with most large numbers
            (data.index.is_numeric() and
             (data.index > pd.Timestamp('1975').timestamp()).mean() > .8)):
            try:
                data.index = pd.to_datetime(data.index, infer_datetime_format=True)
            except ValueError:
                pass

        if 'Volume' not in data:
            data['Volume'] = np.nan

        if len(data) == 0:
            raise ValueError('OHLC `data` is empty')
        if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
            raise ValueError("`data` must be a pandas.DataFrame with columns "
                             "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'")
        if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
            raise ValueError('Some OHLC values are missing (NaN). '
                             'Please strip those lines with `df.dropna()` or '
                             'fill them in with `df.interpolate()` or whatever.')
        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted in ascending order. Sorting.',
                          stacklevel=2)
            data = data.sort_index()
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn('Data index is not datetime. Assuming simple periods, '
                          'but `pd.DateTimeIndex` is advised.',
                          stacklevel=2)
        self._data: pd.DataFrame = data

        self._broker = partial(
            _Broker, cash=9999, commission=0, margin=1.,
            trade_on_close=False, hedging=False,
            exclusive_orders=True, index=data.index,
        )
        self._strategy = strategy
        self._results: Optional[pd.Series] = None
        self.window = window

    def run(self, **kwargs) -> pd.Series:
        """
        Run the monitoring. Returns a signal of type of string with possible values: "BUY", "SELL", None
        """
        # Get the maximum value from strategy parameters and keep the necessary data only
        max_param_value = max([getattr(self._strategy, attr) for attr in [attr for attr in dir(self._strategy) if attr.startswith("param")]])
        self._data = self._data.iloc[-(max_param_value+1):]

        data = _Data(self._data.copy(deep=False))
        broker: _Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)

        strategy.init()
        data._update()  # Strategy.init might have changed/added to data.df

        # Indicators used in Strategy.next()
        indicator_attrs = {attr: indicator
                           for attr, indicator in strategy.__dict__.items()
                           if isinstance(indicator, _Indicator)}.items()

        # Skip first few candles where indicators are still "warming up"
        # +1 to have at least two entries available
        start = 1 + max((np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                         for _, indicator in indicator_attrs), default=0)

        # Disable "invalid value encountered in ..." warnings. Comparison
        # np.nan >= 3 is not invalid; it's False.
        with np.errstate(invalid='ignore'):

            for i in range(start, len(self._data)):
                # Prepare data and indicators for `next` call
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs:
                    # Slice indicator on the last dimension (case of 2d indicator)
                    setattr(strategy, attr, indicator[..., :i + 1])

                # Next tick, a moment before bar close
                self._signal = strategy.next(get_signal=True)

            # Set data back to full length
            # for future `indicator._opts['data'].index` calls to work
            data._set_length(len(self._data))

        return self._signal