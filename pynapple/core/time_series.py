import pandas as pd
import numpy as np
import warnings
from .time_units import TimeUnits
from pandas.core.internals import SingleBlockManager, BlockManager
from .interval_set import IntervalSet


def _get_restrict_method(align):
    if align in ('closest', 'nearest'):
        method = 'nearest'
    elif align in ('next', 'bfill', 'backfill'):
        method = 'bfill'
    elif align in ('prev', 'ffill', 'pad'):
        method = 'pad'
    else:
        raise ValueError('Unrecognized restrict align method')
    return method

def gaps_func(data, min_gap, method='absolute'):
    """
    finds gaps in a tsd
    :param data: a Tsd/TsdFrame
    :param min_gap: the minimum gap that will be considered
    :param method: 'absolute': min gap is expressed in time (us), 'median',
    min_gap expressed in units of the median inter-sample event
    :return: an IntervalSet containing the gaps in the TSd
    """
    dt = np.diff(data.times(units='us'))

    if method == 'absolute':
        pass
    elif method == 'median':
        md = np.median(dt)
        min_gap *= md
    else:
        raise ValueError('unrecognized method')

    ix = np.where(dt > min_gap)
    t = data.times()
    st = t[ix] + 1
    en = t[(np.array(ix) + 1)] - 1
    from neuroseries.interval_set import IntervalSet
    return IntervalSet(st, en)

def support_func(data, min_gap, method='absolute'):
    """
    find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd
    :param data: a Tsd/TsdFrame
    :param min_gap: the minimum gap that will be considered
    :param method: 'absolute': min gap is expressed in time (us), 'median',
    min_gap expressed in units of the median inter-sample event
    :return: an IntervalSet
    """

    here_gaps = data.gaps(min_gap, method=method)
    t = data.times('us')
    from neuroseries.interval_set import IntervalSet
    span = IntervalSet(t[0] - 1, t[-1] + 1)
    support_here = span.set_diff(here_gaps)
    return support_here


class Tsd(pd.Series):
    """
    A subclass of :func:`pandas.Series` specialized for neurophysiology time series.

    Tsd provides standardized time representation, plus functions for restricting and realigning time series
    """

    def __init__(self, t, d=None, time_units=None, time_support=None, **kwargs):
        """
        Tsd Initializer.

        Args:
            t: an object transformable in a time series, or a :func:`~pandas.Series` equivalent (if d is None)
            d: the data in the time series
            time_units: the time units in which times are specified (has no effect if a Pandas object
            is provided as the first argument
            **kwargs: arguments that will be passed to the :func:`~pandas.Series` initializer.
        """
        if isinstance(t, SingleBlockManager):
            d = t.array
            t = t.index.values
            if 'index' in kwargs: kwargs.pop('index')            
        elif isinstance(t, pd.Series):
            d = t.values
            t = t.index.values

        t = TimeUnits.format_timestamps(t, time_units)

        if time_support is not None:
            bins = time_support.values.ravel()
            ix = np.array(pd.cut(t, bins, labels=np.arange(len(bins) - 1, dtype=np.float64)))
            ix[np.floor(ix / 2) * 2 != ix] = np.NaN
            ix = np.floor(ix/2)
            ix = ~np.isnan(ix)
            super().__init__(index=t[ix],data=d[ix])
        else:
            time_support = IntervalSet(start = t[0], end = t[-1])
            super().__init__(index=t, data=d)

        self.time_support = time_support
        self.rate = len(t)/self.time_support.tot_length('s')
        self.index.name = "Time (us)"
        self._metadata.append("nts_class")
        self.nts_class = self.__class__.__name__


    def __repr__(self):
        return self.as_units('s').__repr__()

    def __str__(self): return self.__repr__()

    def times(self, units=None):
        """
        The times of the Tsd, returned as np.double in the desired time units

        Args:
            units: the desired time units

        Returns:
            ts: the times vector

        """
        return TimeUnits.return_timestamps(self.index.values.astype(np.float64), units)

    def as_series(self):
        """
        The Tsd as a :func:`pandas:pandas.Series` object

        Returns:
            ss: the series object

        """

        return pd.Series(self, copy=True)

    def as_units(self, units=None):
        """
        Returns a Series with time expressed in the desired unit.

        :param units: us, ms, or s
        :return: Series with adjusted times
        """
        ss = self.as_series()
        t = self.index.values
        t = TimeUnits.return_timestamps(t, units)
        ss.index = t
        units_str = units
        if not units_str:
            units_str = 'us'
        ss.index.name = "Time (" + units_str + ")"
        return ss

    def data(self):
        """
        The data in the Series object

        Returns: the data

        """
        return self.values

    def value_from(self, tsd, ep, align='closest'):
        """
        Replace the value with the closest value from tsd
        """
        method = _get_restrict_method(align)
        ix = TimeUnits.format_timestamps(self.restrict(ep).index.values)
        tsd = tsd.restrict(ep)
        new_tsd = tsd.reindex(ix, method=method)
        return Tsd(new_tsd, time_support = ep)

    def restrict(self, ep, keep_labels=False):
        """
        Restricts the Tsd to a set of times delimited by a :func:`~neuroseries.interval_set.IntervalSet`

        Args:
            iset: the restricting interval set
            keep_labels:

        Returns:
        # changed col to 0
        """
        ix = ep.in_interval(self)
        tsd_r = pd.DataFrame(self, copy=True)
        col = tsd_r.columns[0]
        tsd_r['interval'] = ix
        ix = ~np.isnan(ix)
        tsd_r = tsd_r[ix]
        return Tsd(tsd_r[col], time_support=ep)

    def count(self, bin_size, ep = None, time_units = 's'):
        """
        Count occurences of events within bin size 
        bin_size should be seconds unless specified     
        If no epochs is passed, the data will be binned based on the time support.
        """     
        if not isinstance(ep, IntervalSet):
            ep = self.time_support
            
        bin_size_us = TimeUnits.format_timestamps(np.array([bin_size]), time_units)[0]

        # bin for each epochs
        time_index = []
        count = []
        for i in ep.index:
            bins = np.arange(ep.start[i], ep.end[i] + bin_size_us, bin_size_us)
            count.append(np.histogram(self.index.values, bins)[0])
            time_index.append(bins[0:-1] + np.diff(bins)/2)
        time_index = np.hstack(time_index)
        count = np.hstack(count)
        return Tsd(t=time_index, d=count, time_support=ep)

    def threshold(self):
        pass


    def gaps(self, min_gap, method='absolute'):
        """
        finds gaps in a tsd
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet containing the gaps in the TSd
        """
        return gaps_func(self, min_gap, method)

    def support(self, min_gap, method='absolute'):
        """
        find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet
        """
        return support_func(self, min_gap, method)

    def start_time(self, units='us'):
        return self.times(units=units)[0]

    def end_time(self, units='us'):
        return self.times(units=units)[-1]

    @property
    def _constructor(self):
        return Tsd


# noinspection PyAbstractClass
class TsdFrame(pd.DataFrame):

    def __init__(self, t, d=None, time_units=None, time_support=None, **kwargs):
        if isinstance(t, pd.DataFrame):
            d = t.values
            c = t.columns.values
            t = t.index.values
        else:
            if 'columns' in kwargs:
                c = kwargs['columns']
            else:
                c = np.arange(d.shape[1])

        t = TimeUnits.format_timestamps(t, time_units)

        if time_support is not None:
            bins = time_support.values.ravel()
            ix = np.array(pd.cut(t, bins, labels=np.arange(len(bins) - 1, dtype=np.float64)))
            ix[np.floor(ix / 2) * 2 != ix] = np.NaN
            ix = np.floor(ix/2)
            ix = ~np.isnan(ix)
            super().__init__(index=t[ix],data=d[ix], columns = c)
        else:
            time_support = IntervalSet(start = t[0], end = t[-1])
            super().__init__(index=t, data=d, columns=c)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.time_support = time_support

        self.rate = len(t)/self.time_support.tot_length('s')
        self.index.name = "Time (us)"
        self._metadata.append("nts_class")
        self.nts_class = self.__class__.__name__

    def __repr__(self):
        return self.as_units('s').__repr__()

    def __str__(self): return self.__repr__()

    def __getitem__(self, key):
        """
        Override to pass time_support
        """
        result = super().__getitem__(key)
        time_support = self.time_support
        if isinstance(result, pd.Series):
            return Tsd(result, time_support=time_support)
        elif isinstance(result, pd.DataFrame):
            return TsdFrame(result, time_support=time_support)

    def times(self, units=None):
        return TimeUnits.return_timestamps(self.index.values.astype(np.float64), units)

    def as_dataframe(self, copy=True):
        """
        :return: copy of the data in a DataFrame (strip Tsd class label)
        """
        return pd.DataFrame(self, copy=copy)

    def as_units(self, units=None):
        """
        returns a DataFrame with time expressed in the desired unit
        :param units: us (s), ms, or s
        :return: DataFrame with adjusted times
        """
        t = self.index.values.copy()
        t = TimeUnits.return_timestamps(t, units)
        df = pd.DataFrame(index=t, data=self.values)
        units_str = units
        if not units_str:
            units_str = 'us'
        df.index.name = "Time (" + units_str + ")"
        df.columns = self.columns.copy()
        return df

    def data(self):
        if len(self.columns) == 1:
            return self.values.ravel()
        return self.values

    def realign(self, t, align='closest'):
        method = _get_restrict_method(align)
        ix = TimeUnits.format_timestamps(t)

        rest_t = self.reindex(ix, method=method, columns=self.columns.values)
        return rest_t

    def restrict(self, iset, keep_labels=False):
        ix = iset.in_interval(self)
        tsd_r = pd.DataFrame(self, copy=True)
        tsd_r['interval'] = ix
        ix = ~np.isnan(ix)
        tsd_r = tsd_r[ix]
        if not keep_labels:
            del tsd_r['interval']
        return TsdFrame(tsd_r, time_support=iset, copy=True)

    def gaps(self, min_gap, method='absolute'):
        """
        finds gaps in a tsd
        :param self: a Tsd/TsdFrame
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet containing the gaps in the TSd
        """
        return gaps_func(self, min_gap, method)

    def support(self, min_gap, method='absolute'):
        """
        find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet
        """
        return support_func(self, min_gap, method)

    def start_time(self, units='us'):
        return self.times(units=units)[0]

    def end_time(self, units='us'):
        return self.times(units=units)[-1]


# noinspection PyAbstractClass
class Ts(Tsd):
    def __init__(self, t, time_units=None, **kwargs):
        super().__init__(t, None, time_units=time_units, **kwargs)
        self.nts_class = self.__class__.__name__

