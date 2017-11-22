"""
A module containing classes and methods for gridded data

Author: Andre
Adapted from load_gridded_data, LMR_prior, LMR_calibrate
"""

from abc import abstractmethod, ABCMeta
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import izip
import numpy as np
import os
from os.path import join
import random
import tables as tb

import LMR_config
from LMR_utils2 import regrid_sphere_gridded_object, var_to_hdf5_carray, \
    empty_hdf5_carray, regrid_esmpy_grid_object
from LMR_utils2 import fix_lon, regular_cov_infl
_LAT = 'lat'
_LON = 'lon'
_LEV = 'lev'
_TIME = 'time'

_DEFAULT_DIM_ORDER = [_TIME, _LEV, _LAT, _LON]
_ALT_DIMENSION_DEFS = {'latitude': _LAT,
                       'longitude': _LON,
                       'plev': _LEV}

_ftypes = LMR_config.Constants.data['file_types']


class GriddedVariable(object):

    __metaclass__ = ABCMeta

    PRE_PROCESSED_FILETAG = '.pre_{}.h5'
    PRE_PROCESSED_FILEDIR = 'pre_proc_files'
    PRE_PROCESSED_OBJ_NODENAME = 'grid_object'
    PRE_PROCESSED_DATA_NODENAME = 'grid_object_data'

    def __init__(self, name, dims_ordered, data, time=None,
                 lev=None, lat=None, lon=None, fill_val=None,
                 sampled=None, avg_interval=None, regrid_method=None,
                 regrid_grid=None, esmpy_interp=None, lat_grid=None,
                 lon_grid=None, climo=None, rotated_pole=False):
        self.name = name
        self.dim_order = dims_ordered
        self.ndim = len(dims_ordered)
        self.data = data
        self.climo = climo
        self.time = time
        self.lev = self._cnvt_to_float_64(lev)
        self.lat = self._cnvt_to_float_64(lat)
        lon_adjusted = fix_lon(lon)
        self.lon = self._cnvt_to_float_64(lon_adjusted)
        self.avg_interval = avg_interval
        self.regrid_method = regrid_method
        self.regrid_grid = regrid_grid
        self.esmpy_interp = esmpy_interp
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.rotated_pole = rotated_pole

        self._fill_val = fill_val
        self._idx_used_for_sample = sampled

        self._dim_coord_map = {_TIME: self.time,
                               _LEV: self.lev,
                               _LAT: self.lat,
                               _LON: self.lon}

        # Make sure ndimensions specified match data
        if self.ndim != len(self.data.shape):
            raise ValueError('Number of dimensions given do not match data'
                             ' dimensions.')

        # Make sure each dimension has consistent number of values as data
        for i, dim in enumerate(self.dim_order):
            if self._dim_coord_map[dim] is None:
                raise ValueError('Dimension specified but no values provided '
                                 'for initialization')

            if data.shape[i] != len(self._dim_coord_map[dim]):
                raise ValueError('Dimension values provided do not match in '
                                 'length with dimension axis of data')

        # Right now it shouldn't happen for loaders, may change in future
        if time is not None:
            self.nsamples = len(self.time)
        else:
            self.nsamples = 1
            self.dim_order.insert(_TIME, 0)
            self.data = self.data.reshape(1, *self.data.shape)

        self._space_dims = [dim for dim in dims_ordered if dim != _TIME]
        self.space_shp = [len(self._dim_coord_map[dim])
                          for dim in self._space_dims]

        if len(self._space_dims) > 2:
            raise ValueError('Class cannot handle 3D data yet!')

        if not self._space_dims:
            self.type = 'timeseries'
            self.space_shp = [1]
            self.data = self.data.reshape(self.nsamples, 1)
        elif _LAT in self._space_dims and _LON in self._space_dims:
            self.type = 'horizontal'
        elif _LAT in self._space_dims and _LEV in self._space_dims:
            self.type = 'meridional_vertical'
        else:
            raise ValueError('Unrecognized dimension combination.')

    def save(self, filename):
        avg_interval = self.avg_interval
        regrid_method = self.regrid_method
        regrid_grid = self.regrid_grid
        esmpy_interp = self.esmpy_interp

        path_pieces = [avg_interval, regrid_method, regrid_grid, esmpy_interp]
        path_pieces = [str(piece) for piece in path_pieces if piece is not None]

        data_path = join('/', *path_pieces)

        print ('Saving pre-processed data file: {}'.format(filename))
        print ('Storing data under HDF5 path: {}'.format(data_path))

        obj_node_name = self.PRE_PROCESSED_OBJ_NODENAME
        data_node_name = self.PRE_PROCESSED_DATA_NODENAME

        if os.path.exists(filename):
            mode = 'a'
        else:
            mode = 'w'

        with tb.open_file(filename, mode=mode,
                          filters=tb.Filters(complib='blosc',
                                             complevel=2)) as h5f:

            try:
                h5f.get_node(data_path, name=obj_node_name)
                h5f.remove_node(data_path, name=obj_node_name)
                h5f.remove_node(data_path, name=data_node_name)
            except tb.NoSuchNodeError:
                root, name = os.path.split(data_path)
                h5f.create_group(root, name=name, createparents=True)

            obj_out = h5f.create_vlarray(data_path, name=obj_node_name,
                                         atom=tb.ObjectAtom())

            self.nan_to_fill_val()
            var_to_hdf5_carray(h5f, data_path, data_node_name, self.data)
            self.fill_val_to_nan()

            tmp_dat = self.data
            del self.data
            obj_out.append(self)
            self.data = tmp_dat

    def print_data_stats(self):
        print ('{}: Global: mean={:1.3e}, '
               'std-dev:={:1.3e}'.format(self.name, np.nanmean(self.data),
                                         np.nanstd(self.data)))

    def regrid(self, regrid_method, regrid_grid=None, grid_def=None,
               interp_method=None):

        assert self.type == 'horizontal'
        class_obj = type(self)

        if regrid_method == 'simple':
            raise NotImplemented('Have not fixed simple regridding yet -AP')
        elif regrid_method == 'spherical_harmonics':
            [regrid_data,
             new_lat,
             new_lon,
             climo] = regrid_sphere_gridded_object(self, regrid_grid)
        elif regrid_method == 'esmpy':
            target_nlat = grid_def['target_nlat']
            target_nlon = grid_def['target_nlon']
            [regrid_data,
             new_lat,
             new_lon,
             climo] = regrid_esmpy_grid_object(target_nlat, target_nlon,
                                                 self,
                                                 interp_method=interp_method)
        else:
            raise ValueError('Unrecognized regridding method: {}'.format(regrid_method))

        # Rotated pole omitted for regridded data
        return class_obj(self.name, self.dim_order, regrid_data,
                         time=self.time,
                         lev=self.lev,
                         lat=new_lat[:, 0],
                         lon=new_lon[0],
                         fill_val=self._fill_val,
                         sampled=self._idx_used_for_sample,
                         avg_interval=self.avg_interval,
                         regrid_method=regrid_method,
                         regrid_grid=regrid_grid,
                         esmpy_interp=interp_method,
                         lat_grid=new_lat,
                         lon_grid=new_lon,
                         climo=climo)

    def fill_val_to_nan(self):
        mask = self.data == self._fill_val

        if np.any(mask):
            self.data[mask] = np.nan
            self.data = np.ma.masked_array(self.data, mask=mask)

    def nan_to_fill_val(self):
        self.data[~np.isfinite(self.data)] = self._fill_val

    def flattened_spatial(self):

        flat_data = self.data.reshape(len(self.time),
                                      np.product(self.space_shp))

        # Get dimensions of data
        coords = [self._dim_coord_map[key] for key in self._space_dims]
        grids = np.meshgrid(*coords, indexing='ij')
        flat_coords = {dim: grid.flatten()
                       for dim, grid in zip(self._space_dims, grids)}

        return flat_data, flat_coords

    def random_sample(self, nens, seed):
        sample_range = range(self.data.shape[0])
        random.seed(seed)
        sample = random.sample(sample_range, nens)
        return self.sample_from_idx(sample)

    def sample_from_idx(self, sample_idxs):
        """
        Random sample ensemble of current gridded variable
        """

        cls = type(self)
        nsamples = len(sample_idxs)

        if nsamples == self.data.shape[0]:
            print ('Size of sample and total number of available members are '
                   'equivalent.  No resampling performed...')
            return self

        print ('Random selection of {} ensemble members'.format(nsamples))

        time_sample = self.time[sample_idxs]
        data_sample = np.zeros([nsamples] + list(self.data.shape[1:]))
        for k, idx in enumerate(sample_idxs):
            data_sample[k] = self.data[idx]

        # Account for timeseries trailing singleton dimension
        data_sample = np.squeeze(data_sample)

        return cls(self.name, self.dim_order, data_sample,
                   time=time_sample,
                   lev=self.lev,
                   lat=self.lat,
                   lon=self.lon,
                   fill_val=self._fill_val,
                   avg_interval=self.avg_interval,
                   rotated_pole=self.rotated_pole,
                   lat_grid=self.lat_grid,
                   lon_grid=self.lon_grid,
                   regrid_method=self.regrid_method,
                   regrid_grid=self.regrid_grid,
                   esmpy_interp=self.esmpy_interp,
                   climo=self.climo,
                   sampled=sample_idxs)

    def is_sampled(self):

        if self._idx_used_for_sample is None:
            return False
        else:
            return True

    def convert_to_anomaly(self, climo=None):
        print ('Removing temporal mean for every gridpoint...')
        if climo is None:
            self.climo = self.data[:].mean(axis=0, keepdims=True)
        else:
            self.climo = climo

        self.data = self.data - self.climo

    def convert_to_standard(self):
        print ('Adding temporal mean to every gridpoint...')
        if self.climo is None:
            raise ValueError('Cannot convert to standard state data is not an '
                             'anomaly to start.')

        self.data = self.data + self.climo
        self.climo = None

    @abstractmethod
    def load(cls, config, *args):
        pass

    @classmethod
    def _main_load_helper(cls, file_dir, file_name, varname, file_type,
                          nens=None, seed=None, sample=None,
                          avg_interval=None, avg_interval_kwargs=None,
                          regrid_method=None, regrid_grid=None,
                          esmpy_kwargs=None,
                          data_req_frac=0.0, save=True,
                          ignore_pre_avg=False, rotated_pole=False,
                          anomaly=True, detrend=False):

        try:
            ftype_loader = cls.get_loader_for_filetype(file_type)
        except KeyError:
            raise TypeError('Specified file type not supported yet.')

        try:
            if ignore_pre_avg:
                raise IOError('Ignore pre_averaged files is set to True.')

            interp_method = esmpy_kwargs['interp_method']

            var_obj = cls._load_pre_avg_obj(file_dir, file_name, varname,
                                            avg_interval=avg_interval,
                                            regrid_method=regrid_method,
                                            regrid_grid=regrid_grid,
                                            anomaly=anomaly,
                                            nens=nens,
                                            sample=sample,
                                            seed=seed,
                                            interp_method=interp_method)
        except IOError:
            print 'No pre-averaged file found or ignore specified ... '
            var_obj = ftype_loader(file_dir, file_name, varname, save=save,
                                   data_req_frac=data_req_frac,
                                   avg_interval=avg_interval,
                                   avg_interval_kwargs=avg_interval_kwargs,
                                   rotated_pole=rotated_pole,
                                   anomaly=anomaly)
            print 'Loaded from file: {}/{}'.format(file_dir, file_name)

        var_obj.fill_val_to_nan()

        if regrid_method is not None and var_obj.regrid_method is None:
            var_obj = var_obj.regrid(regrid_method=regrid_method,
                                     regrid_grid=regrid_grid,
                                     **esmpy_kwargs)

            var_obj.print_data_stats()

            if save:
                pre_tag = cls.PRE_PROCESSED_FILETAG.format(varname)
                pre_dir = cls.PRE_PROCESSED_FILEDIR
                path = join(file_dir, pre_dir, file_name + pre_tag)
                var_obj.save(path)

        if not var_obj.is_sampled() and (nens is not None or sample is not None):

            if sample is not None:
                var_obj = var_obj.sample_from_idx(sample)
            else:
                var_obj = var_obj.random_sample(nens, seed)

        return var_obj

    @classmethod
    def get_loader_for_filetype(cls, file_type):
        ftype_map = {_ftypes['netcdf']: cls._load_from_netcdf}
        return ftype_map[file_type]

    @classmethod
    def _load_pre_avg_obj(cls, dir_name, filename, varname, avg_interval=None,
                          regrid_method=None, regrid_grid=None,
                          anomaly=False, nens=None, sample=None,
                          seed=None, interp_method=None):
        """
        General structure for load pre-averaged:
        1. Load data
            a. If regrid is desired it searches for pre_avg regridded data
               but if not found, then uses loaded data and regrids
        2. Sample if desired
        3. Return a gridded variable object.
        """

        # Check if pre-processed averages file exists
        pre_proc_tag = cls.PRE_PROCESSED_FILETAG.format(varname)
        pre_filedir = cls.PRE_PROCESSED_FILEDIR

        path = join(dir_name, pre_filedir, filename + pre_proc_tag)

        # Look for pre_averaged_file
        if not os.path.exists(path):
            raise IOError('No pre-averaged file found for given specifications')

        # Load prior object
        with tb.open_file(path, 'a') as h5f:

            obj_node_name = cls.PRE_PROCESSED_OBJ_NODENAME
            data_node_name = cls.PRE_PROCESSED_DATA_NODENAME

            # Get nodes for pre-processed grid object with correct averaging
            obj_dir = join('/', avg_interval)
            obj = h5f.get_node(obj_dir, name=obj_node_name)[0]
            obj_data = h5f.get_node(obj_dir, name=data_node_name)
            print ('Found node for avg_interval path: {}'.format(obj_dir))

            # Look for pre-regridded data if specified
            do_sample = True
            if regrid_method is not None:
                # TODO: This won't alarm user if grid_def is changing
                regrid_path = [regrid_method, regrid_grid, interp_method]
                regrid_path = [str(path_piece) for path_piece in regrid_path
                               if path_piece is not None]
                regrid_obj_dir = join(obj_dir, *regrid_path)

                try:
                    regrid_obj = h5f.get_node(regrid_obj_dir,
                                              name=obj_node_name)[0]
                    regrid_obj_data = h5f.get_node(regrid_obj_dir,
                                                   name=data_node_name)
                    print ('Found node for regridded data under path: '
                           '{}'.format(regrid_obj_dir))
                    obj = regrid_obj
                    obj_data = regrid_obj_data
                except tb.NoSuchNodeError:
                    # Do not sample, since regrid specified and might save
                    do_sample = False
                    obj_data = obj_data.read()
                    print('Regridded pre-processed grid object not found for '
                          'regridding: {}.'.format(regrid_obj_dir))

            obj.data = obj_data

            if anomaly and obj.climo is None:
                obj.convert_to_anomaly()

            if do_sample:
                if sample is not None:
                    obj = obj.sample_from_idx(sample)
                elif nens is not None:
                    obj = obj.random_sample(nens, seed)
                else:
                    obj.data = obj.data.read()

        print 'Loaded pre-averaged file: {}'.format(path)
        return obj

    @classmethod
    def _load_from_netcdf(cls, dir_name, filename, varname, avg_interval=None,
                          avg_interval_kwargs=None, save=False,
                          data_req_frac=None, rotated_pole=False,
                          anomaly=False):
        """
        General structure for load origininal:
        1. Load data
        2. Avg to base resolution
        3. Separate into subannual groups
        4. Create GriddedVar Object for each group and save pre-averaged
        5. Sample if desired
        6. Return list of GriddedVar objects
        """

        # Check if pre-processed averages file exists
        pre_proc_filetag = cls.PRE_PROCESSED_FILETAG.format(varname)
        pre_proc_filedir = cls.PRE_PROCESSED_FILEDIR

        with Dataset(join(dir_name, filename), 'r') as f:
            var = f.variables[varname]
            data_shp = var.shape
            try:
                fill_val = var._FillValue
            except AttributeError:
                fill_val = 2**15 - 1

            # Convert to key names defined in _DEFAULT_DIM_ORDER
            dims = []
            dim_keys = []
            dim_exclude = []
            for i, dim in enumerate(var.dimensions):
                if data_shp[i] == 1:
                    dim_exclude.append(dim)
                elif dim in _DEFAULT_DIM_ORDER:
                    dims.append(dim.lower())
                    dim_keys.append(dim)
                else:
                    dims.append(_ALT_DIMENSION_DEFS[dim.lower()])
                    dim_keys.append(dim)

            # Make sure it has time dimension
            if _TIME not in dims:
                raise ValueError('No time dimension for specified prior data.')

            # Check order of all dimensions
            idx_order = [_DEFAULT_DIM_ORDER.index(dim) for dim in dims]
            if idx_order != sorted(idx_order):
                raise ValueError('Input file dimensions do not match default'
                                 ' ordering specified by _DEFAULT_DIM_ORDER '
                                 'in LMR_gridded.py.')

            # Load dimension values
            dim_vals = {dim_name: f.variables[dim_key]
                        for dim_name, dim_key in zip(dims, dim_keys)}

            # Convert time to datetimes
            dim_vals[_TIME] = cls._netcdf_datetime_convert(dim_vals[_TIME])

            # Extract data for each dimension
            dim_vals = {k: val[:] for k, val in dim_vals.iteritems()}

            # Extract grid in rotated pole
            if rotated_pole:
                lat_grid = dim_vals[_LAT]
                lon_grid = dim_vals[_LON]
                dim_vals[_LAT] = dim_vals[_LAT][..., 0]
                dim_vals[_LON] = dim_vals[_LON][..., 0, :]
            else:
                lat_grid = None
                lon_grid = None

            var_dat = np.squeeze(var[:])

            # Average to correct time interval
            [dim_vals[_TIME],
             avg_data] = cls._avg_to_specified_period(dim_vals[_TIME],
                                                      var_dat,
                                                      data_req_frac=data_req_frac,
                                                      **avg_interval_kwargs)

            # Create gridded object
            grid_obj = cls(varname, dims, avg_data, fill_val=fill_val,
                           avg_interval=avg_interval, rotated_pole=rotated_pole,
                           lat_grid=lat_grid, lon_grid=lon_grid,
                           **dim_vals)

            if anomaly:
                grid_obj.convert_to_anomaly()

            grid_obj.print_data_stats()

            if save:
                new_dir = join(dir_name, pre_proc_filedir)
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)

                pre_proc_fname = join(new_dir, filename + pre_proc_filetag)
                grid_obj.save(pre_proc_fname)

            return grid_obj

    @staticmethod
    def _cnvt_to_float_64(data):
        if data is not None:
            data = data.astype(np.float64)
        return data

    @staticmethod
    def _netcdf_datetime_convert(time_var):
        """
        Converts netcdf time variable into date-times.

        Used as a static method in case necesary to overwrite with subclass
        :param time_var:
        :return:
        """
        if not hasattr(time_var, 'calendar'):
            cal = 'standard'
        else:
            cal = time_var.calendar

        try:
            time = num2date(time_var[:], units=time_var.units,
                                calendar=cal)
            return time
        except ValueError:
            # num2date needs calendar year start >= 0001 C.E. (bug submitted
            # to unidata about this
            # TODO: Add a warning about likely inaccuracy day res and smaller
            tunits = time_var.units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 0001

            new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
            time = num2date(time_var[:], new_units, calendar=cal)
            reshifted_time = [datetime(d.year + year_diff, d.month, d.day,
                                       d.hour, d.minute, d.second)
                              for d in time]
            return np.array(reshifted_time)

    @staticmethod
    def _avg_to_specified_period(time_vals, data, nelem_in_yr=12,
                                 elem_to_avg=(1,2,3), nyears=1,
                                 data_req_frac=None):

        """Resample data to specified averaging period.  Assumes contiguous
        intevals are being used to resample."""

        time_vals = np.array(time_vals)

        ntimes = data.shape[0]
        spatial_shape = data.shape[1:]
        len_of_sample = len(elem_to_avg)

        # starting index for resampling,  e.g. 0, 12, 24, equivalent for monthly
        start_idx = elem_to_avg[0] % nelem_in_yr

        # Find how many full years you can average from the data and cutoff
        total_yrs = (ntimes - start_idx) // nelem_in_yr
        end_idx = start_idx + total_yrs*nelem_in_yr

        time_vals = time_vals[start_idx:end_idx]
        data = data[start_idx:end_idx]

        # reshape time dimension to (years, sub-year)
        time_vals = time_vals.reshape(total_yrs, nelem_in_yr)
        data = data.reshape(total_yrs, nelem_in_yr, *spatial_shape)

        # Find how many multi-year averages you can get from data and cutoff
        total_avg_periods = total_yrs // nyears
        year_cutoff_idx = total_yrs * nyears

        time_vals = time_vals[0:year_cutoff_idx]
        data = data[0:year_cutoff_idx]

        # reshape time dimension to (multi-annual avg periods, nyears in avg)
        time_vals = time_vals.reshape(total_avg_periods, nyears, nelem_in_yr)
        data = data.reshape(total_avg_periods, nyears, nelem_in_yr,
                            *spatial_shape)

        # Average the data
        new_times = time_vals[:, 0, 0]
        data = data[:, :, 0:len_of_sample]
        new_data = np.nanmean(data, axis=(1, 2))

        # Mask times which did not have enough data in the average
        if data_req_frac is not None and np.any(np.isnan(data)):
            nelem_in_avg = nyears*nelem_in_yr
            num_valid = np.isfinite(data).sum(axis=(1, 2))
            valid_frac = num_valid.astype(np.float) / nelem_in_avg
            invalid = valid_frac < data_req_frac
            new_data[invalid] = np.nan

        return new_times, new_data


class PriorVariable(GriddedVariable):

    @classmethod
    def load(cls, prior_config, varname, sample=None):
        file_dir = prior_config.datadir_prior
        file_name = prior_config.datafile_prior
        file_type = prior_config.dataformat_prior
        nens = prior_config.nens
        seed = prior_config.seed
        save = prior_config.save_pre_avg_file
        ignore_pre_avg = prior_config.ignore_pre_avg_file
        detrend = prior_config.detrend
        avg_interval = prior_config.avg_interval
        avg_interval_kwargs = prior_config.avg_interval_kwargs
        regrid_method = prior_config.regrid_method
        regrid_grid = prior_config.regrid_grid
        esmpy_kwargs = {'grid_def': prior_config.esmpy_grid_def,
                        'interp_method': prior_config.esmpy_interp_method}

        anomaly = prior_config.state_variables[varname]
        if anomaly == 'anom':
            anomaly = True
        elif anomaly == 'full':
            anomaly = False
        else:
            raise ValueError('Incorrect specification of state variable '
                             'anomly or full-field.')

        datainfo = prior_config.datainfo_prior
        if 'rotated_pole' in datainfo.keys():
            rotated_pole = varname in datainfo['rotated_pole']
        else:
            rotated_pole = False

        if datainfo['template']:
            file_name = file_name.replace(datainfo['template'], varname)
            varname = varname.split('_')[0]

        return cls._main_load_helper(file_dir, file_name, varname, file_type,
                                     nens=nens,
                                     seed=seed,
                                     sample=sample,
                                     save=save,
                                     ignore_pre_avg=ignore_pre_avg,
                                     avg_interval=avg_interval,
                                     avg_interval_kwargs=avg_interval_kwargs,
                                     data_req_frac=1.0,
                                     regrid_method=regrid_method,
                                     regrid_grid=regrid_grid,
                                     esmpy_kwargs=esmpy_kwargs,
                                     rotated_pole=rotated_pole,
                                     anomaly=anomaly,
                                     detrend=detrend)

    @classmethod
    def load_allvars(cls, config):
        var_names = config.prior.state_variables

        sample = None
        prior_dict = OrderedDict()
        for vname in var_names:
            pobjs = cls.load(config, vname, sample)
            # Assure that same sample is used for all variables of a prior
            sample = pobjs[0]._idx_used_for_sample
            prior_dict[vname] = pobjs

        return prior_dict


class AnalysisVariable(GriddedVariable):

    @classmethod
    def load(cls, psm_config):

        # TODO: change this to switch based on PSM
        file_dir = psm_config.datadir_calib
        file_name = psm_config.datafile_calib
        file_type = psm_config.dataformat_calib
        base_resolution = psm_config.sub_base_res
        varname = psm_config.varname_calib
        dat_frac = psm_config.min_data_req_frac
        overwrite = psm_config.overwrite_pre_avg_file
        ignore = psm_config.ignore_pre_avg_file

        return cls._main_load_helper(file_dir, file_name, varname, file_type,
                                     base_resolution, split_varname=False,
                                     data_req_frac=dat_frac,
                                     save=overwrite,
                                     ignore_pre_avg=ignore)

    @classmethod
    def load_allvars(cls):
        pass


class BerkeleyEarthAnalysisVariable(AnalysisVariable):

    @classmethod
    def load(cls, psm_config):
        return super(BerkeleyEarthAnalysisVariable, cls)

    @staticmethod
    def _netcdf_datetime_convert(time_var):
        """
        Converts netcdf time variable into date-times.

        Used as a static method in case necesary to overwrite with subclass
        :param time_var:
        :return:
        """

        time_yrs = []
        for yrAD in time_var[:]:

            year = int(yrAD)
            rem = yrAD - year
            base = datetime(year, 1, 1)
            diff_yr = base.replace(year=base.year + 1) - base
            diff_to_yr_secs = diff_yr.total_seconds()
            tdel = timedelta(seconds=(diff_to_yr_secs * rem))
            time_yrs.append(base + tdel)

        return np.array(time_yrs)


class State(object):
    """
    Class to create state vector and information
    """

    def __init__(self, prior_vars, base_res, adaptive=False):

        self._prior_vars = prior_vars
        self.base_res = base_res
        self.resolution = base_res
        self.state_list = []
        self.var_coords = {}
        self.var_view_range = {}
        self.var_space_shp = {}
        self.augmented = False
        self._adaptive = adaptive

        # Attr for backend storage
        self.output_backend = None
        self._orig_state = None
        self._tmp_state = {}

        self.len_state = 0
        for var, pobjs in prior_vars.iteritems():

            self.var_space_shp[var] = pobjs[0].space_shp

            var_start = self.len_state
            for i, pobj in enumerate(pobjs):

                flat_data, flat_coords = pobj.flattened_spatial()

                # Store range of data in state dimension
                if i == 0:
                    var_end = flat_data.T.shape[0] + var_start
                    self.var_view_range[var] = (var_start, var_end)
                    self.len_state = var_end

                # Add prior to state vector, transposed to make state the first
                # dimension, and ensemble members along the 2nd
                try:
                    self.state_list[i] = \
                        np.concatenate((self.state_list[i], flat_data.T),
                                       axis=0)
                except IndexError:
                    self.state_list.append(flat_data.T)

            # Save variable view information
            self.var_coords[var] = flat_coords

        self.state_list = np.array(self.state_list)
        self.shape = self.state_list[0].shape
        self.old_state_info = self.get_old_state_info()

        # Information for adaptive inflation
        if adaptive:
            self.infl_factor_mean = np.ones((self.shape[0]))
            self.infl_factor_var = self.infl_factor_mean * 0.6
        else:
            self.infl_factor_mean = None
            self.infl_factor_var = None

    @classmethod
    def from_config(cls, config):
        pvars = PriorVariable.load_allvars(config)
        base_res = config.core.sub_base_res
        adaptive = config.core.adaptive_inflate

        return cls(pvars, base_res, adaptive=adaptive)

    def get_var_data(self, var_name, idx=None):
        """
        Returns a view (or a copy) of the variable in the state vector
        """
        # TODO: change idx to optional, if none does average
        # probably switch statelist to numpy array for easy averaging.
        start, end = self.var_view_range[var_name]

        if idx is not None:
            var_data = self.state_list[idx, start:end]
        else:
            var_data = self.state_list[:, start:end]

        return var_data

    def truncate_state(self):
        """
        Create a truncated copy of the current state
        """
        trunc_pvars = OrderedDict()
        for var_name, pvar in self._prior_vars.iteritems():
            trunc_pvars[var_name] = []
            for pobj in pvar:
                if pobj.type == 'horizontal':
                    trunc_pvars[var_name].append(pobj.truncate())
                else:
                    trunc_pvars[var_name].append(pobj)
        state_class = type(self)
        return state_class(trunc_pvars, self.base_res, adaptive=self._adaptive)

    def augment_state(self, ye_vals):

        # Add leading axis and repeat to match state_list length
        nproxies  = len(ye_vals)
        ye_vals = ye_vals[None]
        ye_vals = np.repeat(ye_vals, len(self.state_list), axis=0)

        self.state_list = np.concatenate((self.state_list, ye_vals), axis=1)
        self.augmented = True

        self.var_view_range['state'] = (0, self.len_state)
        self.var_view_range['ye_vals'] = (self.len_state,
                                          self.len_state + nproxies)

    def reg_inflate_xb(self, inf_factor):
        """
        Regular variance inflation

        Parameters
        ----------
        inf_factor

        Returns
        -------
        """

        xb_vals = self.state_list[0]
        xb_vals[:] = regular_cov_infl(xb_vals, inf_factor)

    def adaptive_inflate_xb(self, prox_man, t):
        """
        J.L. Anderson adaptive inflation algorithm. Only takes annual data for
        now.

        Parameters
        ----------
        self
        prox_man
        t

        Returns
        -------

        """
        assert self.resolution == 1.0

        lambda_upper_bound = 1.e6
        lambda_lower_bound = 1.
        lambda_sd_lower_bound = 0.0

        # Inflation factor mean and variance
        lambda_p_bar = self.infl_factor_mean
        lambda_p_sd = self.infl_factor_var

        sigma_lambda_p_2 = lambda_p_sd ** 2

        ye_vals = self.get_var_data('ye_vals', idx=0)
        y_vals_iter = prox_man.sites_assim_res_proxy_objs(1.0, t)

        xb_vals = self.get_var_data('state', idx=0)
        xb_mean = xb_vals.mean(axis=1)
        xb_pert = xb_vals - xb_mean[:, None]
        nens = xb_pert.shape[1]
        xb_var = (xb_pert**2).sum(axis=1) / (nens - 1)

        for y, ye in izip(y_vals_iter, ye_vals):

            try:
                y_val = y.values[t]
                y_err = y.psm_obj.R
            except KeyError:
                # No ob for current year ... skip
                continue

            ye_mean = ye.mean()
            ye_pert = ye - ye_mean
            ye_err = ye.var(ddof=1)

            D = abs(y_val - ye.mean())
            D2 = D**2

            # Correlation between state (at every location) and ye_ens

            state_ye_cov = (ye_pert * xb_pert).sum(axis=1) / (nens - 1)
            norm = np.sqrt(ye_err * xb_var)

            r = state_ye_cov / norm

            # Expected inflation for observation
            lambda_o_bar = (1 + r * (np.sqrt(lambda_p_bar) - 1)) ** 2

            theta_bar_2 = lambda_o_bar*ye_err + y_err
            theta_bar = np.sqrt(theta_bar_2)

            dtheta_dlambda = 0.5 * ye_err * r
            dtheta_dlambda *= (1 - r + (r * np.sqrt(lambda_p_bar)))
            dtheta_dlambda /= (theta_bar * np.sqrt(lambda_p_bar))

            # Update the mean
            u_bar = 1. / (np.sqrt(2*np.pi) * theta_bar)
            l_exp_bar = D2 / (-2. * theta_bar_2)
            v_bar = np.exp(l_exp_bar)
            l_bar = u_bar * v_bar

            l_prime = l_bar * (D2 / theta_bar_2 - 1) / theta_bar
            l_prime *= dtheta_dlambda

            a = np.array([1])
            b = l_bar/l_prime - 2 * lambda_p_bar
            c = lambda_p_bar**2 - sigma_lambda_p_2 - l_bar*lambda_p_bar/l_prime

            # Quadratic equation
            disc = np.sqrt(b**2 - 4*a*c)
            pos_root = (-b + disc) / (2*a)
            neg_root = (-b - disc) / (2*a)

            root_comp = np.zeros((2, len(pos_root)))
            root_comp[0] = abs(neg_root - lambda_p_bar)
            root_comp[1] = abs(pos_root - lambda_p_bar)
            chosen = root_comp.argmin(axis=0)
            lambda_u = root_comp[chosen, xrange(len(pos_root))]

            revert_idx = ((lambda_u < lambda_lower_bound) &
                          (lambda_u > lambda_upper_bound))
            lambda_u[revert_idx] = lambda_p_bar[revert_idx]

            # Calculate ratio for variance update
            num_max = self._compute_new_density(D2, ye_err, y_err, lambda_p_bar,
                                                lambda_p_sd, r, lambda_u)
            num_2 = self._compute_new_density(D2, ye_err, y_err, lambda_p_bar,
                                              lambda_p_sd, r,
                                              lambda_u + lambda_p_sd)

            # TODO: add check for finite or 0 before taking ratio

            ratio = num_2 / num_max

            lambda_u_sd = np.sqrt(-sigma_lambda_p_2 / (2 * np.log(ratio)))
            lambda_u_sd[ratio > 0.99] = lambda_p_sd[ratio > 0.99]

            revert_idx = (lambda_u_sd < lambda_sd_lower_bound)

            # If out of bounds, revert
            lambda_u_sd[revert_idx] = lambda_p_sd[revert_idx]
            # update prior

            lambda_p_bar[:] = lambda_u
            lambda_p_sd[:] = lambda_u_sd

        # Inflate the state perturbation
        xb_pert *= lambda_p_bar[:, None]
        xb_vals[:] = xb_mean[:, None] + xb_pert

    @staticmethod
    def _compute_new_density(D2, sigma_p_2, sigma_o_2, lambda_p_bar,
                             lambda_sd, r, lambda_new):

        exp_prior = (lambda_new - lambda_p_bar)**2 / (-2 * lambda_sd**2)
        theta_2 = ((1 + r * (np.sqrt(lambda_p_bar) - 1))**2 * sigma_p_2 +
                   sigma_o_2)
        theta = np.sqrt(theta_2)

        exp_likeli = D2 / (-2 * theta_2)
        new_density = (np.exp(exp_likeli + exp_prior) /
                       (2 * np.pi * lambda_sd * theta))
        return new_density

    def reset_augmented_ye(self, ye_vals):

        ye_state = self.get_var_data('ye_vals')
        for i, subann_ye in enumerate(ye_state):
            subann_ye[:] = ye_vals

    def annual_avg(self, var_name=None):

        if var_name is None:
            subannual_data = self.state_list
        else:
            # Get variable data for each subannual state vector, creat ndarray
            subannual_data = self.get_var_data(var_name)

        avg = subannual_data.mean(axis=0)
        return avg

    def stash_state_list(self, name):

        self._tmp_state[name] = (self.state_list.copy(), self.resolution)

    def stash_recall_state_list(self, name, pop=False, copy=False):

        if name in self._tmp_state:
            if pop:
                state, res = self._tmp_state.pop(name)
            else:
                state, res = self._tmp_state[name]

                if copy:
                    state = state.copy()

            self.state_list = state
            self.resolution = res
        else:
            print 'No currently stashed state with name {}....'.format(name)

    def stash_pop_state_list(self, name):
        self.stash_recall_state_list(name, pop=True)

    def avg_to_res(self, res, shift):

        """Average current state list to resolution"""

        if res == self.resolution:
            return

        if res < 1:
            chunk = int(res / self.base_res)
            shift_idx = int(shift / self.base_res)
            tmp_dat = np.roll(self.state_list, shift_idx, axis=0)
            nelem = len(tmp_dat) / chunk
            new_state = tmp_dat.reshape(nelem, chunk, *tmp_dat.shape[1:])
            new_state = new_state.mean(axis=1)
            self.state_list = new_state
        elif res == 1:
            self.state_list = self.state_list.mean(axis=0, keepdims=True)
        else:
            raise ValueError('Cannot handle resolutions larger than 1 yet.')

        self.resolution = res

    def restore_orig_state(self):
        self.state_list = self._orig_state.copy()
        self.resolution = self.base_res

    def get_old_state_info(self):

        state_info = {}
        for var in self.var_view_range.keys():
            var_info = {'pos': self.var_view_range[var]}

            space_dims = [dim for dim in _DEFAULT_DIM_ORDER
                          if dim in self.var_coords[var].keys()]

            var_info['spacecoords'] = space_dims
            if not space_dims:
                var_info['spacedims'] = None
            else:
                var_info['spacedims'] = self.var_space_shp[var]
            state_info[var] = var_info

        return state_info

    def initialize_storage_backend(self, btype, nyears, fdir):

        _types = {'NPY': _NPYStateStorage,
                  'H5': _HDF5StateStorage}

        backend_class = _types[btype]
        store_orig = True

        if self._orig_state is not None:
            store_orig = False
            self.stash_state_list('tmp')
            self.restore_orig_state()

        # Extra year, just in case we need to use shifted dates
        self.output_backend = backend_class(nyears+1, self, fdir=fdir)

        self.output_backend.insert(self.state_list, 0)

        if store_orig:
            self._orig_state = self.state_list.copy()
        else:
            self.stash_pop_state_list('tmp')

    def insert_upcoming_prior(self, curr_yr_idx, use_curr=False):

        if not use_curr:
            dat = self._orig_state
        else:
            dat = self.output_backend.get_xb(0, curr_yr_idx)

        self.output_backend.insert(dat, curr_yr_idx+1)

    def xb_from_backend(self, yr_idx, res, shift):

        self.state_list = self.output_backend.get_xb(shift, yr_idx)
        self.resolution = self.base_res
        self.avg_to_res(res, 0)

    def propagate_avg_to_backend(self, yr_idx, shift):

        self.output_backend.propagate_avg_to_storage(shift, self, yr_idx)

    def close_xb_container(self):
        self.output_backend.close()


class _BaseStateStorage(object):
    """
    Class for storing state vector data
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, nyears, state, fdir=None):
        self._yr_len = None
        self._base_res = None
        self.xb_out = None
        pass

    @abstractmethod
    def close(self):
        pass

    def insert(self, data, yr_idx):
        """
        Insert as we go to prevent huge upfront write cost
        """
        istart = yr_idx*self._yr_len
        iend = istart + self._yr_len

        self.xb_out[istart:iend] = data

    def get_xb(self, shift, yr_idx):

        ishift = int(shift / self._base_res)
        istart = yr_idx*self._yr_len + ishift
        iend = istart + self._yr_len

        return self.xb_out[istart:iend]

    def propagate_avg_to_storage(self, shift, state, yr_idx):
        nchunks = len(state.state_list)
        chk_size = self._yr_len / nchunks
        ishift = int(shift / self._base_res)

        for i in xrange(nchunks):
            avg = state.state_list[i]

            istart = yr_idx*self._yr_len + i*chk_size + ishift
            iend = istart + chk_size
            tmp_dat = self.xb_out[istart:iend]

            if len(tmp_dat) > 1:
                tmp_dat = tmp_dat - tmp_dat.mean(axis=0)
                tmp_dat += avg

                self.xb_out[istart:iend] = tmp_dat
            else:
                # same size as _base_res, just replace
                self.xb_out[istart:iend] = avg


class _HDF5StateStorage(_BaseStateStorage):

    """
    Uses Pytables to store *ALL* years in the sub_base resolution in HDF5
    format.  Slower, but saves the entire ensemble for all variables in a
    semi-compressed format.
    """

    def __init__(self, nyears, state, fdir='./'):
        res = state.base_res
        fname = 'state_output_res{:1d}pt{:2d}'.format(int(res),
                                                      int((res % 1.0) * 100))

        self.h5f_out = tb.open_file(join(fdir, fname), 'w',
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=2))
        atom = tb.Atom.from_dtype(state.state_list[0].dtype)
        num_subann = len(state.state_list)
        tdim_len = nyears * num_subann
        shape = [tdim_len] + list(state.state_list[0].shape)

        self.xb_out = empty_hdf5_carray(self.h5f_out, '/', 'output', atom,
                                        shape)
        self._yr_len = num_subann
        self._base_res = state.base_res

    def close(self):
        self.h5f_out.close()


class _NPYStateStorage(_BaseStateStorage):

    """
    Numpy based storage for the state. Only stores 2-years to allow for
    multi-resolution functionality. Faster because there's no IO.
    """

    def __init__(self, nyears, state, fdir=None):

        """
        Note: nyears and dir are dummy variables
        """
        self._nyears = 3

        num_subann = len(state.state_list)
        shape = [self._nyears*num_subann] + list(state.state_list[0].shape)
        self.xb_out = np.zeros(shape)
        self._yr_len = num_subann
        self._base_res = state.base_res
        self._restore_idx = None

    def insert(self, data, yr_idx):

        yr_idx %= self._nyears

        super(_NPYStateStorage, self).insert(data, yr_idx)

    def get_xb(self, shift, yr_idx):

        yr_idx %= self._nyears

        self._roll_idx_to_zero(yr_idx)
        val = super(_NPYStateStorage, self).get_xb(shift, 0).copy()
        self._unroll_idx_to_orig()

        return val

    def propagate_avg_to_storage(self, shift, state, yr_idx):

        yr_idx %= self._nyears

        self._roll_idx_to_zero(yr_idx)
        super(_NPYStateStorage, self).propagate_avg_to_storage(shift, state, 0)
        self._unroll_idx_to_orig()

    def close(self):
        pass

    def _roll_idx_to_zero(self, idx):
        if self._restore_idx is not None:
            raise AttributeError('Cannot perform index roll on storage that'
                                 ' has not been previously unrolled.')

        if idx != 0:
            self._restore_idx = idx * self._yr_len
            self.xb_out = np.roll(self.xb_out, -idx*self._yr_len, axis=0)

    def _unroll_idx_to_orig(self):
        if self._restore_idx is not None:
            self.xb_out = np.roll(self.xb_out, self._restore_idx, axis=0)
            self._restore_idx = None


_analysis_var_classes = {'BerkeleyEarth': BerkeleyEarthAnalysisVariable}


def get_analysis_var_class(analysis_source):
    return _analysis_var_classes.get(analysis_source, AnalysisVariable)
