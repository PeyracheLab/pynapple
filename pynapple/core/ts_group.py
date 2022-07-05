# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-28 15:10:48
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-01 12:16:06


import numpy as np
import pandas as pd
import sys
import warnings
from collections import UserDict
from tabulate import tabulate
from .time_series import Ts, Tsd, TsdFrame
from .interval_set import IntervalSet
from .time_units import TimeUnits

def intersect_intervals(i_sets):
    """
    Helper to intersect intervals from ts_group
    """
    n_sets = len(i_sets)
    time1 = [i_set['start'] for i_set in i_sets]
    time2 = [i_set['end'] for i_set in i_sets]
    time1.extend(time2)
    time = np.hstack(time1)
    start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
                          -1 * np.ones(len(time)//2, dtype=np.int32)))

    df = pd.DataFrame({'time': time, 'start_end': start_end})
    df.sort_values(by='time', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['cumsum'] = df['start_end'].cumsum()
    ix = (df['cumsum']==n_sets).to_numpy().nonzero()[0]
    return IntervalSet(df['time'][ix], df['time'][ix+1])

def union_intervals(i_sets):
    """
    Helper to merge intervals from ts_group
    """
    time = np.hstack([i_set['start'] for i_set in i_sets] +
                     [i_set['end'] for i_set in i_sets])
    start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
                          -1 * np.ones(len(time)//2, dtype=np.int32)))
    df = pd.DataFrame({'time': time, 'start_end': start_end})
    df.sort_values(by='time', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['cumsum'] = df['start_end'].cumsum()
    ix_stop = (df['cumsum']==0).to_numpy().nonzero()[0]
    ix_start = np.hstack((0, ix_stop[:-1]+1))
    return IntervalSet(df['time'][ix_start], df['time'][ix_stop])


class TsGroup(UserDict):
    """
    The TsGroup is a dictionnary-like object to hold multiple [`Ts`][pynapple.core.time_series.Ts] 
    or [`Tsd`][pynapple.core.time_series.Tsd] objects
    with different time index.

    Attributes
    ----------
    time_support: IntervalSet
        The time support of the TsGroup
    """
    
    def __init__(self, data, time_support=None, time_units='s', **kwargs):
        """
        TsGroup Initializer
        
        Parameters
        ----------
        data : dict
            Dictionnary containing Ts/Tsd objects
        time_support : IntervalSet, optional
            The time support of the TsGroup. Ts/Tsd objects will be restricted to the time support if passed.
        time_units : str, optional
            Time units if data does not contain Ts/Tsd objects ('us', 'ms', 's' [default]).
        **kwargs
            Meta-info about the Ts/Tsd objects. Can be either pandas.Series or numpy.ndarray.
            Note that the index should match the index of the input dictionnary.
        
        Raises
        ------
        RuntimeError
            Raise error if the intersection of time support of Ts/Tsd object is empty.
        """
        self._initialized = False
        
        index = np.sort(list(data.keys()))

        self._metadata = pd.DataFrame(index=index, columns = ['freq'])
        
        # Transform elements to Ts/Tsd objects
        for k in index:
            if isinstance(data[k], (np.ndarray,list)):
                warnings.warn('Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object.', stacklevel=2)
                data[k] = Ts(t = data[k], time_support = time_support, time_units = time_units)

        # If time_support is passed, all elements of data are restricted prior to init
        if isinstance(time_support, IntervalSet):
            self.time_support = time_support
            data = {k:data[k].restrict(self.time_support) for k in index}
        else:
            # Otherwise do the union of all time supports            
            time_support = union_intervals([data[k].time_support for k in index])
            if len(time_support) == 0:
                raise RuntimeError("Intersection of time supports is empty. Consider passing a time support as argument.")
            self.time_support = time_support                
            data = {k:data[k].restrict(self.time_support) for k in index}
        #print(data)
        UserDict.__init__(self, data)
        
        # Making the TsGroup non mutable
        self._initialized = True

        # Trying to add argument as metainfo
        self.set_info(**kwargs)
        
    """
    Base functions
    """
    def __setitem__(self, key, value):
        if self._initialized:
            raise RuntimeError("TsGroup object is not mutable.")
        if self.__contains__(key):
            raise KeyError("Key {} already in group index.".format(key))
        else:
            if isinstance(value, (Ts, Tsd)):
                self._metadata.loc[int(key),'freq'] = value.rate
                super().__setitem__(int(key), value)
            elif isinstance(value, (np.ndarray, list)):
                warnings.warn('Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object.', stacklevel=2)
                tmp = Ts(t = value, time_units = 's')
                self._metadata.loc[int(key),'freq'] = tmp.rate
                super().__setitem__(int(key), tmp)
            else:
                raise ValueError("Value with key {} is not an iterable.".format(key))

    def __getitem__(self, key):
        if key.__hash__:
            if self.__contains__(key):
                return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))
        else:           
            metadata = self._metadata.loc[key,self._metadata.columns.drop('freq')]
            return TsGroup({k:self[k] for k in key}, time_support=self.time_support, **metadata)
    
    def __repr__(self):
        cols = self._metadata.columns.drop('freq')
        headers = ['Index', 'Freq. (Hz)'] + [c for c in cols]       
        lines = []      
    
        for i in self.data.keys():
            lines.append([str(i), '%.2f' % self._metadata.loc[i,'freq']] + [self._metadata.loc[i,c] for c in cols])
        return tabulate(lines, headers = headers)       
        
    def __str__(self):
        return self.__repr__()

    def keys(self): 
        """
        Return index/keys of TsGroup
        
        Returns
        -------
        list
            List of keys
        """
        return list(self.data.keys())

    def items(self):
        """
        Return a list of key/object.
        
        Returns
        -------
        list
            List of tuples
        """
        return list(self.data.items())

    def values(self):
        """
        Return a list of all the Ts/Tsd objects in the TsGroup
        
        Returns
        -------
        list
            List of Ts/Tsd objects
        """
        return list(self.data.values())

    #######################
    # Metadata 
    #######################
    def set_info(self, *args, **kwargs):        
        """
        Add metadata informations about the TsGroup.
        Metadata are saved as a DataFrame.

        Parameters
        ----------
        *args
            pandas.Dataframe or list of pandas.DataFrame
        **kwargs
            Can be either pandas.Series or numpy.ndarray
        Raises
        ------
        RuntimeError
            Raise an error if
                no column labels are found when passing simple arguments,
                indexes are not equals for a pandas series,
                not the same length when passing numpy array.

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
        
        To add metadata with a pandas.DataFrame:

        >>> import pandas as pd
        >>> structs = pd.DataFrame(index = [0,1,2], data=['pfc','pfc','ca1'], columns=['struct'])
        >>> tsgroup.set_info(structs)
        >>> tsgroup
          Index    Freq. (Hz)  struct
        -------  ------------  --------
              0             1  pfc
              1             2  pfc
              2             4  ca1
        
        To add metadata with a pd.Series or numpy.ndarray:

        >>> hd = pd.Series(index = [0,1,2], data = [0,1,1])
        >>> tsgroup.set_info(hd=hd)
        >>> tsgroup
          Index    Freq. (Hz)  struct      hd
        -------  ------------  --------  ----
              0             1  pfc          0
              1             2  pfc          1
              2             4  ca1          1

        """
        if len(args):
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    if pd.Index.equals(self._metadata.index, arg.index):
                        self._metadata = self._metadata.join(arg)
                    else:
                        raise RuntimeError("Index are not equals")
                elif isinstance(arg, (pd.Series, np.ndarray)):
                    raise RuntimeError("Columns needs to be labelled for metadata")
        if len(kwargs):
            for k, v in kwargs.items():
                if isinstance(v, pd.Series):
                    if pd.Index.equals(self._metadata.index, v.index):
                        self._metadata[k] = v
                    else:
                        raise RuntimeError("Index are not equals")
                elif isinstance(v, np.ndarray):
                    if len(self._metadata) == len(v):
                        self._metadata[k] = v
                    else:
                        raise RuntimeError("Array is not the same length.")
        return

    def get_info(self, key):
        """
        Returns the metainfo located in one column
        
        Parameters
        ----------
        key : str
            One of the metainfo columns name
        
        Returns
        -------
        pandas.Series
            The metainfo
        """
        return self._metadata[key]

    def _union_time_support(self):
        """
        Helper
        """
        idx = list(self.data.keys())
        i_sets = [self.data[i].time_support for i in idx]
        return union_intervals(i_sets)

    def _intersect_time_support(self):
        """
        Helper
        """
        idx = list(self.data.keys())
        i_sets = [self.data[i].time_support for i in idx]
        return intersect_intervals(i_sets)


    #################################
    #Generic functions of Tsd objects
    #################################
    def restrict(self, ep):
        """
        Restricts a TsGroup object to a set of time intervals delimited by an IntervalSet object
        
        Parameters
        ----------
        ep : IntervalSet
            the IntervalSet object 
        
        Returns
        -------
        TsGroup
            TsGroup object restricted to ep
        
        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')
        >>> newtsgroup = tsgroup.restrict(ep)
        
        All objects within the TsGroup automatically inherit the epochs defined by ep.
                
        >>> newtsgroup.time_support
           start    end
        0    0.0  100.0
        >>> newtsgroup[0].time_support
           start    end
        0    0.0  100.0
        """
        newgr = {}
        for k in self.data:
            newgr[k] = self.data[k].restrict(ep)
        cols = self._metadata.columns.drop('freq')

        return TsGroup(newgr, time_support = ep, **self._metadata[cols])

    def value_from(self, tsd, ep=None, align='closest'):
        """
        Replace the value of each Ts/Tsd object within the Ts group with the closest value from tsd argument
        
        Parameters
        ----------
        tsd : Tsd
            The Tsd object holding the values to replace
        ep : IntervalSet
            The IntervalSet object to restrict the operation.
            If None, the time support of the tsd input object is used.
        align : str, optional
            The method to align (closest/prev/next)
        
        Returns
        -------
        TsGroup
            TsGroup object with the new values
        
        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')      
        
        The variable tsd is a time series object containing the values to assign, for example the tracking data:
        
        >>> tsd = nap.Tsd(t=np.arange(0,100), d=np.random.rand(100), time_units='s')
        >>> ep = nap.IntervalSet(start = 0, end = 100, time_units = 's')
        >>> newtsgroup = tsgroup.value_from(tsd, ep)
            
        """
        if ep is None:
            ep = tsd.time_support        
        tsd = tsd.restrict(ep)
        newgr = {}
        for k in self.data:
            newgr[k] = self.data[k].value_from(tsd, ep, align)

        cols = self._metadata.columns.drop('freq')
        return TsGroup(newgr, time_support = ep, **self._metadata[cols])

    def count(self, bin_size, ep = None, time_units = 's'):
        """
        Count occurences of events within bin_size.
        bin_size should be seconds unless specified.
        If no epochs is passed, the data will be binned based on the time support.

        Parameters
        ----------
        bin_size : float
            The bin size (default is second)            
        ep : IntervalSet, optional
            IntervalSet to restrict the operation
            If None, the time support of self is used.
        time_units : str, optional
            Time units of bin size ('us', 'ms', 's' [default])
        
        Returns
        -------
        TsdFrame
            A TsdFrame with the columns being the index of each item in the TsGroup.

        Example
        -------
        This example shows how to count events within bins of 0.1 second for the first 100 seconds.
        
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')          
        >>> bincount = tsgroup.count(0.1, ep)
        >>> bincount
                  0  1  2
        Time (s)         
        0.05      0  0  0
        0.15      0  0  0
        0.25      0  0  1
        0.35      0  0  0
        0.45      0  0  0
        ...      .. .. ..
        99.55     0  1  1
        99.65     0  0  0
        99.75     0  0  1
        99.85     0  0  0
        99.95     1  1  1
        [1000 rows x 3 columns]     
        
        """
        if not isinstance(ep, IntervalSet):
            ep = self.time_support
            
        bin_size = TimeUnits.format_timestamps(np.array([bin_size]), time_units)[0]

        # bin for each epochs
        time_index = []
        count = []
        for i in ep.index:
            bins = np.arange(ep.start[i], ep.end[i] + bin_size, bin_size)
            tmp = np.array([np.histogram(self.data[n].index.values, bins)[0] for n in self.keys()])
            count.append(np.transpose(tmp))
            time_index.append(bins[0:-1] + np.diff(bins)/2)

        count = np.vstack(count)
        time_index = np.hstack(time_index)

        toreturn = TsdFrame(
            t = time_index, 
            d = count, 
            time_support = ep, 
            columns = list(self.keys()))
        return toreturn

    """
    Special slicing of metadata
    """
    def getby_threshold(self, key, thr, op = '>'):
        """
        Return a TsGroup with all Ts/Tsd objects with values above threshold for metainfo under key.
        
        Parameters
        ----------
        key : str
            One of the metainfo columns name
        thr : float
            THe value for thresholding
        op : str, optional
            The type of operation. Possibilities are '>', '<', '>=' or '<='.
        
        Returns
        -------
        TsGroup
            The new TsGroup
        
        Raises
        ------
        RuntimeError
            Raise eror is operation is not recognized.

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
          Index    Freq. (Hz)
        -------  ------------
              0             1
              1             2
              2             4   

        This exemple shows how to get a new TsGroup with all elements for which the metainfo frequency is above 1.
        >>> newtsgroup = tsgroup.getby_threshold('freq', 1, op = '>')
          Index    Freq. (Hz)
        -------  ------------
              1             2
              2             4

        """
        if op == '>':
            ix = list(self._metadata.index[self._metadata[key] > thr])
            return self[ix]
        elif op == '<':
            ix = list(self._metadata.index[self._metadata[key] < thr])
            return self[ix]
        elif op == '>=':
            ix = list(self._metadata.index[self._metadata[key] >= thr])
            return self[ix]
        elif op == '<=':
            ix = list(self._metadata.index[self._metadata[key] <= thr])
            return self[ix]
        else:
            raise RuntimeError("Operation {} not recognized.".format(op))


    def getby_intervals(self, key, bins):
        """
        Return a list of TsGroup binned.
        
        Parameters
        ----------
        key : str
            One of the metainfo columns name
        bins : numpy.ndarray or list
            The bin intervals
        
        Returns
        -------
        list
            A list of TsGroup
        
        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp, alpha = np.arange(3))
          Index    Freq. (Hz)    alpha                                                                    
        -------  ------------  -------
              0             1        0
              1             2        1
              2             4        2

        This exemple shows how to bin the TsGroup according to one metainfo key.
        >>> newtsgroup, bincenter = tsgroup.getby_intervals('alpha', [0, 1, 2])
        >>> newtsgroup
        [  Index    Freq. (Hz)    alpha
         -------  ------------  -------
               0             1        0,
           Index    Freq. (Hz)    alpha
         -------  ------------  -------
               1             2        1]

        By default, the function returns the center of the bins.
        >>> bincenter
        array([0.5, 1.5])
        """
        idx = np.digitize(self._metadata[key], bins)-1
        groups = self._metadata.index.groupby(idx)
        ix = np.unique(list(groups.keys()))
        ix = ix[ix>=0]
        ix = ix[ix<len(bins)-1]
        xb = bins[0:-1] + np.diff(bins)/2
        sliced = [self[list(groups[i])] for i in ix]  
        return sliced, xb[ix]           

    def getby_category(self, key):
        """
        Return a list of TsGroup grouped by category.
        
        Parameters
        ----------
        key : str
            One of the metainfo columns name
        
        Returns
        -------
        dict
            A dictionnary of TsGroup
        
        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp, group = [0,1,1])
          Index    Freq. (Hz)    group
        -------  ------------  -------
              0             1        0
              1             2        1
              2             4        1

        This exemple shows how to group the TsGroup according to one metainfo key.
        >>> newtsgroup = tsgroup.getby_category('group')
        >>> newtsgroup
        {0:   Index    Freq. (Hz)    group
         -------  ------------  -------
               0             1        0,
         1:   Index    Freq. (Hz)    group
         -------  ------------  -------
               1             2        1
               2             4        1}        
        """
        groups = self._metadata.groupby(key).groups
        sliced = {k:self[list(groups[k])] for k in groups.keys()}
        return sliced

