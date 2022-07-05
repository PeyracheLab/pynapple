# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-27 18:33:31
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-07 11:24:31

import pandas as pd
import numpy as np
import warnings
from .time_units import TimeUnits
from pandas.core.internals import SingleBlockManager, BlockManager
from .interval_set import IntervalSet

def _get_restrict_method(align):
    """
    Get method for alignment
    """
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
    """

    here_gaps = data.gaps(min_gap, method=method)
    t = data.times('us')
    from neuroseries.interval_set import IntervalSet
    span = IntervalSet(t[0] - 1, t[-1] + 1)
    support_here = span.set_diff(here_gaps)
    return support_here

class Tsd(pd.Series):
    # class Tsd():
    """
    A subclass of pandas.Series specialized for neurophysiology time series.
    
    Tsd provides standardized time representation, plus various functions for manipulating times series.

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """
    
    def __init__(self, t, d=None, time_units='s', time_support=None, **kwargs):
        """
        Tsd Initializer.
                
        Parameters
        ----------
        t : numpy.ndarray or pandas.Series
            An object transformable in a time series, or a pandas.Series equivalent (if d is None)
        d : numpy.ndarray, optional
            The data of the time series
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default])
        time_support : IntervalSet, optional
            The time support of the tsd object
        **kwargs
            Arguments that will be passed to the pandas.Series initializer.
        """
        if isinstance(t, SingleBlockManager):
            d = t.array
            t = t.index.values
            if 'index' in kwargs: kwargs.pop('index')            
        elif isinstance(t, pd.Series):
            d = t.values
            t = t.index.values

        t = TimeUnits.format_timestamps(t, time_units)

        if len(t) > 0:
            if time_support is not None:
                bins = time_support.values.ravel()
                # Because yes there is no funtion with both bounds closed as an option
                ix = np.vstack((
                    np.array(pd.cut(t, bins, labels=np.arange(len(bins) - 1, dtype=np.float64))),
                    np.array(pd.cut(t, bins, labels=np.arange(len(bins) - 1, dtype=np.float64), right=False))
                    )).T
                ix[np.floor(ix / 2) * 2 != ix] = np.NaN
                ix = np.floor(ix/2)
                ix[np.isnan(ix[:,0]),0] = ix[np.isnan(ix[:,0]),1]
                ix = ~np.isnan(ix[:,0])
                if d is not None:
                    super().__init__(index=t[ix], data=d[ix])
                else:
                    super().__init__(index=t[ix], data=None, dtype=np.float64)
            else:
                time_support = IntervalSet(start = t[0], end = t[-1], time_units = 's')
                if d is not None:
                    super().__init__(index=t, data=d)
                else:
                    super().__init__(index=t, data=d, dtype=np.float64)
        else:
            time_support = IntervalSet(pd.DataFrame(columns=['start','end']))
            super().__init__(index=t, data=d, dtype=np.float64)

        self.time_support = time_support
        if self.time_support.tot_length('s') > 0:
            self.rate = len(t) / self.time_support.tot_length('s')
        else:
            self.rate = np.nan
        self.index.name = "Time (s)"
        self._metadata.append("nap_class")
        self.nap_class = self.__class__.__name__

    def __lt__(self, value):
        return self.as_series().__lt__(value)

    def __gt__(self, value):
        return self.as_series().__gt__(value)

    def __le__(self, value):
        return self.as_series().__le__(value)

    def __ge__(self, value):
        return self.as_series().__ge__(value)

    def __ne__(self, value):
        return self.as_series().__ne__(value)

    def __eq__(self, value):        
        return self.as_series().__eq__(value)

    def __repr__(self):
        return self.as_series().__repr__()

    def __str__(self): return self.__repr__()

    def times(self, units='s'):
        """
        The time index of the Tsd, returned as np.double in the desired time units.
        
        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        out: numpy.ndarray
            the time indexes
        """
        times = TimeUnits.return_timestamps(self.index.values.astype(np.float64), units)
        return times

    def as_series(self):
        """
        Convert the Ts/Tsd object to a pandas.Series object.
        
        Returns
        -------
        out: pandas.Series
            _
        """
        return pd.Series(self, copy=True)

    def as_units(self, units='s'):
        """
        Returns a Series with time expressed in the desired unit.
        
        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])
        
        Returns
        -------
        pandas.Series
            the series object with adjusted times
        """
        ss = self.as_series()
        t = self.index.values
        t = TimeUnits.return_timestamps(t, units)
        if units == 'us':
            t = t.astype(np.int64)
        ss.index = t
        units_str = units
        if not units_str:
            units_str = 's'
        ss.index.name = "Time (" + units_str + ")"
        return ss

    def data(self):
        """
        The data in the Tsd object
        
        Returns
        -------
        out: numpy.ndarray
            _
        """
        return self.values

    def value_from(self, tsd, ep=None, align='closest'):
        """
        Replace the value with the closest value from tsd argument
        
        Parameters
        ----------
        tsd : Tsd
            The Tsd object holding the values to replace
        ep : IntervalSet (optional)
            The IntervalSet object to restrict the operation. 
            If None, the time support of the tsd input object is used.
        align : str, optional
            The method to align (closest/prev/next)
        
        Returns
        -------
        out : Tsd
            Tsd object with the new values
        
        Example
        -------
        In this example, the ts object will receive the closest values in time from tsd.
        
        >>> import pynapple as nap
        >>> import numpy as np
        >>> t = np.unique(np.sort(np.random.randint(0, 1000, 100))) # random times
        >>> ts = nap.Ts(t=t, time_units='s')             
        >>> tsd = nap.Tsd(t=np.arange(0,1000), d=np.random.rand(1000), time_units='s')
        >>> ep = nap.IntervalSet(start = 0, end = 500, time_units = 's')
        
        The variable ts is a time series object containing only nan.
        The tsd object containing the values, for example the tracking data, and the epoch to restrict the operation.
        
        >>> newts = ts.value_from(tsd, ep)
        
        newts is the same size as ts restrict to ep.
        
        >>> print(len(ts.restrict(ep)), len(newts))
            52 52
        """
        if ep is None:
            ep = tsd.time_support
        method = _get_restrict_method(align)
        ix = TimeUnits.format_timestamps(self.restrict(ep).index.values)
        tsd = tsd.restrict(ep)
        tsd = tsd.as_series()
        new_tsd = tsd.reindex(ix, method=method)
        return Tsd(new_tsd, time_support = ep)

    def restrict(self, ep):
        """
        Restricts a Tsd object to a set of time intervals delimited by an IntervalSet object
        
        Parameters
        ----------
        ep : IntervalSet
            the IntervalSet object 
        
        Returns
        -------
        out: Tsd
            Tsd object restricted to ep
        
        Example
        -------
        The Ts object is restrict to the intervals defined by ep.
    
        >>> import pynapple as nap
        >>> import numpy as np
        >>> t = np.unique(np.sort(np.random.randint(0, 1000, 100)))
        >>> ts = nap.Ts(t=t, time_units='s')
        >>> ep = nap.IntervalSet(start=0, end=500, time_units='s')
        >>> newts = ts.restrict(ep)
        
        The time support of newts automatically inherit the epochs defined by ep.
        
        >>> newts.time_support
        >>>    start    end
        >>> 0    0.0  500.0
        
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
        Count occurences of events within bin_size. 
        bin_size should be seconds unless specified.
        If no epochs is passed, the data will be binned based on the time support.
        
        Parameters
        ----------
        bin_size : float
            The bin size (default is second)
        
        ep : None or IntervalSet, optional
            IntervalSet to restrict the operation
        
        time_units : str, optional
            Time units of bin size ('us', 'ms', 's' [default])
        
        Returns
        -------
        out: Tsd
            A Tsd object indexed by the center of the bins.
        
        Example
        -------
        This example shows how to count events within bins of 0.1 second.
        
        >>> import pynapple as nap
        >>> import numpy as np
        >>> t = np.unique(np.sort(np.random.randint(0, 1000, 100)))
        >>> ts = nap.Ts(t=t, time_units='s')
        >>> bincount = ts.count(0.1)
        
        An epoch can be specified:
        
        >>> ep = nap.IntervalSet(start = 100, end = 800, time_units = 's')
        >>> bincount = ts.count(0.1, ep=ep)

        And bincount automatically inherit ep as time support: 

        >>> bincount.time_support
        >>>    start    end
        >>> 0  100.0  800.0
        """
        if not isinstance(ep, IntervalSet):
            ep = self.time_support
            
        bin_size = TimeUnits.format_timestamps(np.array([bin_size]), time_units)[0]

        # bin for each epochs
        time_index = []
        count = []
        for i in ep.index:
            bins = np.arange(ep.start[i], ep.end[i] + bin_size, bin_size)
            count.append(np.histogram(self.index.values, bins)[0])
            time_index.append(bins[0:-1] + np.diff(bins)/2)
        time_index = np.hstack(time_index)
        count = np.hstack(count)
        return Tsd(t=time_index, d=count, time_support=ep)

    def threshold(self, thr, method='above'):
        """
        Apply a threshold function to the tsd to return a new tsd 
        with the time support being the epochs above/below the threshold
        
        Parameters
        ----------
        thr : float
            The threshold value
        method : str, optional
            The threshold method (above/below)
        
        Returns
        -------
        out: Tsd
            All the time points below or above the threshold

        Raises
        ------
        ValueError
            Raise an error if method is not 'below' or 'above'
        RuntimeError
            Raise an error if thr is too high/low and no epochs is found.
          
        Example
        -------
        This example finds all epoch above 0.5 within the tsd object.
        
        >>> import pynapple as nap
        >>> tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100))
        >>> newtsd = tsd.threshold(0.5)
        
        The epochs with the times above/below the threshold can be accessed through the time support:
        
        >>> tsd = nap.Tsd(t=np.arange(100), d=np.arange(100), time_units='s')
        >>> tsd.threshold(50).time_support
        >>>    start   end
        >>> 0   50.5  99.0
        
        """
        d = self.values
        t = self.index.values
        idx_rising = np.where(np.logical_and(d[:-1] <= thr, d[1:] > thr))[0]
        idx_falling = np.where(np.logical_and(d[:-1] >= thr, d[1:] < thr))[0]

        if method == 'above':
            starts = t[idx_rising] + (t[idx_rising+1]-t[idx_rising])/2
            ends = t[idx_falling] + (t[idx_falling+1]-t[idx_falling])/2
            if d[0] > thr:
                starts = np.hstack((t[0], starts))                
            if d[-1] > thr:
                ends = np.hstack((ends, t[-1]))
        elif method == 'below':
            starts = t[idx_falling] + (t[idx_falling+1]-t[idx_falling])/2
            ends = t[idx_rising] + (t[idx_rising+1]-t[idx_rising])/2
            if d[0] < thr:
                starts = np.hstack((t[0], starts))                
            if d[-1] < thr:
                ends = np.hstack((ends, t[-1]))
        else:
            raise ValueError("Method {} for thresholding is not accepted.".format(method))

        if (len(starts)==0 and len(ends)==0) or len(starts)!=len(ends):
            raise RuntimeError("Threshold {} with method {} returned empty tsd.".format(thr, method))
        else:
            time_support = IntervalSet(start = starts, end = ends)
            time_support = time_support.drop_short_intervals(0)
            time_support = self.time_support.intersect(time_support)
            tsd = self.restrict(time_support)
            return tsd

    def gaps(self, min_gap, method='absolute'):
        return gaps_func(self, min_gap, method)

    def support(self, min_gap, method='absolute'):
        return support_func(self, min_gap, method)

    def start_time(self, units='s'):
        """
        The first time index in the Ts/Tsd object
        
        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])
        
        Returns
        -------
        out: numpy.float64
            _
        """        
        return self.times(units=units)[0]

    def end_time(self, units='s'):
        """
        The last time index in the Ts/Tsd object
        
        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])
        
        Returns
        -------
        out: numpy.float64
            _
        """        
        return self.times(units=units)[-1]

    @property
    def _constructor(self):
        return Tsd


# noinspection PyAbstractClass
class TsdFrame(pd.DataFrame):
    # class TsdFrame():
    """
    A subclass of pandas.DataFrame specialized for neurophysiological time series.
    
    TsdFrame provides standardized time representation, plus various functions for manipulating times series with identical sampling frequency.
    
    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """
    
    def __init__(self, t, d=None, time_units='s', time_support=None, **kwargs):
        """
        TsdFrame initializer
        
        Parameters
        ----------
        t : numpy.ndarray or pandas.DataFrame
            the time index t,  or a pandas.DataFrame (if d is None)
        d : numpy.ndarray
            The data
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default]).
        time_support : IntervalSet, optional
            The time support of the TsdFrame object
        **kwargs
            Arguments that will be passed to the pandas.DataFrame initializer.
        """
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
            # Because yes there is no funtion with both bounds closed as an option
            ix = np.vstack((
                np.array(pd.cut(t, bins, labels=np.arange(len(bins) - 1, dtype=np.float64))),
                np.array(pd.cut(t, bins, labels=np.arange(len(bins) - 1, dtype=np.float64), right=False))
                )).T
            ix[np.floor(ix / 2) * 2 != ix] = np.NaN
            ix = np.floor(ix/2)
            ix[np.isnan(ix[:,0]),0] = ix[np.isnan(ix[:,0]),1]
            ix = ~np.isnan(ix[:,0])
            super().__init__(index=t[ix],data=d[ix], columns = c)
        else:
            time_support = IntervalSet(start = t[0], end = t[-1])
            super().__init__(index=t, data=d, columns=c)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.time_support = time_support

        self.rate = len(t)/self.time_support.tot_length('s')
        self.index.name = "Time (s)"
        self._metadata.append("nap_class")
        self.nap_class = self.__class__.__name__

    def __repr__(self):
        return self.as_units('s').__repr__()

    def __str__(self): return self.__repr__()

    def __getitem__(self, key):
        result = super().__getitem__(key)
        time_support = self.time_support
        if isinstance(result, pd.Series):
            return Tsd(result, time_support=time_support)
        elif isinstance(result, pd.DataFrame):
            return TsdFrame(result, time_support=time_support)

    def times(self, units='s'):
        """
        The time index of the TsdFrame, returned as np.double in the desired time units.
        
        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])
        
        Returns
        -------
        out: numpy.ndarray        
            _
        """        
        return TimeUnits.return_timestamps(self.index.values.astype(np.float64), units)

    def as_dataframe(self, copy=True):
        """
        Convert the TsdFrame object to a pandas.DataFrame object.
        
        Returns
        -------
        out: pandas.DataFrame
            _
        """
        return pd.DataFrame(self, copy=copy)

    def as_units(self, units='s'):
        """
        Returns a DataFrame with time expressed in the desired unit.
        
        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])
        
        Returns
        -------
        out: pandas.DataFrame
            the series object with adjusted times
        """
        t = self.index.values.copy()
        t = TimeUnits.return_timestamps(t, units)
        if units == 'us':
            t = t.astype(np.int64)

        df = pd.DataFrame(index=t, data=self.values)
        units_str = units
        if not units_str:
            units_str = 's'
        df.index.name = "Time (" + units_str + ")"
        df.columns = self.columns.copy()
        return df

    def data(self):
        """
        The data in the TsdFrame object
        
        Returns
        -------
        out: numpy.ndarray
            _
        """        
        if len(self.columns) == 1:
            return self.values.ravel()
        return self.values

    def realign(self, t, align='closest'):
        method = _get_restrict_method(align)
        ix = TimeUnits.format_timestamps(t)

        rest_t = self.reindex(ix, method=method, columns=self.columns.values)
        return rest_t

    def value_from(self, tsd, ep=None, align='closest'):
        """
        Replace the value with the closest value from tsd argument
        
        Parameters
        ----------
        tsd : Tsd
            The Tsd object holding the values to replace
        ep : IntervalSet, optional
            The IntervalSet object to restrict the operation.
            If None, ep is taken from the tsd of the time support
        align : str, optional
            The method to align (closest/prev/next)
        
        Returns
        -------
        out: Tsd
            Tsd object with the new values
        
        Example
        -------
        In this example, the ts object will receive the closest values in time from tsd.

        >>> import pynapple as nap
        >>> import numpy as np
        >>> t = np.unique(np.sort(np.random.randint(0, 1000, 100))) # random times
        >>> ts = nap.Ts(t=t, time_units='s')
        >>> tsd = nap.TsdFrame(t=np.arange(0,1000), d=np.random.rand(1000), time_units='s')
        >>> ep = nap.IntervalSet(start = 0, end = 500, time_units = 's')
        
        The variable ts is a time series object containing only nan.
        The tsd object containing the values, for example the tracking data, and the epoch to restrict the operation.
        
        >>> newts = ts.value_from(tsd, ep)
        
        newts is the same size as ts restrict to ep.
        
        >>> print(len(ts.restrict(ep)), len(newts))
            52 52
        """
        if ep is None:
            ep = tsd.time_support
        method = _get_restrict_method(align)
        ix = TimeUnits.format_timestamps(self.restrict(ep).index.values)
        tsd = tsd.restrict(ep)
        tsd = tsd.as_series()
        new_tsd = tsd.reindex(ix, method=method)
        return Tsd(new_tsd, time_support = ep)

    def restrict(self, iset, keep_labels=False):
        """
        Restricts a TsdFrame object to a set of time intervals delimited by an IntervalSet object`
        
        Parameters
        ----------
        iset : IntervalSet
            the IntervalSet object 
        keep_labels : bool, optional
            Wheter or not to drop the label of a column
        
        Returns
        -------
        TsdFrame
            TsdFrame object restricted to ep
        
        """
        ix = iset.in_interval(self)
        tsd_r = pd.DataFrame(self, copy=True)
        tsd_r['interval'] = ix
        ix = ~np.isnan(ix)
        tsd_r = tsd_r[ix]
        if not keep_labels:
            del tsd_r['interval']
        return TsdFrame(tsd_r, time_support=iset, copy=True)

    def gaps(self, min_gap, method='absolute'):
        return gaps_func(self, min_gap, method)

    def support(self, min_gap, method='absolute'):
        return support_func(self, min_gap, method)

    def start_time(self, units='s'):
        return self.times(units=units)[0]

    def end_time(self, units='s'):
        return self.times(units=units)[-1]


# noinspection PyAbstractClass
class Ts(Tsd):
    """
    A subclass of the Tsd object for a time series with only time index,
    By default, the values are set to nan.
    All the functions of a Tsd object are available in a Ts object.

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(self, t, time_units='s', time_support=None,**kwargs):
        """
        Ts Initializer
        
        Parameters
        ----------
        t : numpy.ndarray or pandas.Series
            An object transformable in a time series, or a pandas.Series equivalent (if d is None)
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default])
        time_support : IntervalSet, optional
            The time support of the Ts object
        **kwargs
            Arguments that will be passed to the pandas.Series initializer.
        """
        super().__init__(t, None, 
            time_units=time_units, 
            time_support=time_support, 
            dtype=np.float64,**kwargs)
        self.nts_class = self.__class__.__name__

