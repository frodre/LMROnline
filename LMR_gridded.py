"""
A module containing classes and methods for gridded data

Author: Andre
Adapted from load_gridded_data, LMR_prior, LMR_calibrate
"""

from abc import abstractmethod, ABCMeta
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np
import warnings
import os
from os.path import join
import random
import tables as tb
import pylim.DataTools as DT

import LMR_config
from LMR_utils2 import regrid_sphere_gridded_object, var_to_hdf5_carray, \
    empty_hdf5_carray, regrid_esmpy_grid_object
from LMR_utils2 import fix_lon, regular_cov_infl
# import pylim.DataTools as DT

# Constant definitions
_LAT = 'lat'
_LON = 'lon'
_LEV = 'lev'
_TIME = 'time'

_DEFAULT_DIM_ORDER = [_TIME, _LEV, _LAT, _LON]
_ALT_DIMENSION_DEFS = {'latitude': _LAT,
                       'longitude': _LON,
                       'plev': _LEV}
_BYPASS_DIMENSION_DEFS = {'j': _LAT,
                          'i': _LON}

_ftypes = LMR_config.Constants.data['file_types']

def _cnvt_to_float64(num):
    if num is None:
        return None
    else:
        num_as_array = np.array(num)
        return num_as_array.astype(np.float64)


class GriddedVariable(object):
    """
    Object for holding and manipulating gridded data of a single variable.
    """

    PRE_PROCESSED_FILETAG = '.pre_{}.h5'
    PRE_PROCESSED_FILEDIR = 'pre_proc_files'
    PRE_PROCESSED_OBJ_NODENAME = 'grid_object'
    PRE_PROCESSED_DATA_NODENAME = 'grid_object_data'

    def __init__(self, name, dims_ordered, data, time=None,
                 lev=None, lat=None, lon=None, fill_val=None,
                 sampled=None, avg_interval=None, regrid_method=None,
                 regrid_grid=None, esmpy_interp=None, lat_grid=None,
                 lon_grid=None, climo=None, rotated_pole=False, cell_area=None):
        """

        Parameters
        ----------
        name: str
            Name of gridded variable.
        dims_ordered: list of str
            Ordered list of the gridded variable dimensions.  Dimension names
            should match the constant definitions at the beginning of this
            module.
        data: ndarray
            Gridded variable data
        time: ndarray, optional
            Array of time dimension values
        lev: ndarray, optional
            Array of level dimension values
        lat: ndarray, optional
            Array of latitude dimension values
        lon: ndarray, optional
            Array of longitude dimension values
        fill_val: float, optional
            Fill value indicating missing data
        sampled: array of int, optional
            List of indices indicating the sample from the original data
            used to create this object.
        avg_interval: str, optional
            Key indicating the averaging interval of the data in this
            object.  Should match distinction in constants.yml.
        regrid_method: str, optional
            Key indicating which regridding method was used to create
            data in this object.
        regrid_grid: str, optional
            Key indicating which grid the data is on
        esmpy_interp: str, optional
            Key indicating the interpolation method used when ESMPy was used to
            regrid the data
        lat_grid: ndarray, optional
            Latitude values for the grid.  Necessary for non-regular grids for
            certain operations.
        lon_grid: ndarray, optional
            Longitude values for the grid.  Necessary for non-regular grids for
            certain operations.
        climo: ndarray, optional
            Climatology used to center the data
        rotated_pole: bool, optional
            Indication of whether or not the data is on a rotated pole grid
        cell_area: ndarray, optional
            Grid array describing the area of each grid cell.

        Attributes
        ----------
        ndim: int
            Number of data dimensions
        nsamples: int
            Number of samples in the data
        space_shp: tuple of int
            Shape of spatial dimensions of the data
        type: str
            Variable definition based on dimensions. E.g., timeseries,
            2D:horizontal, 2D:vertical/meridional
        """
        self.name = name
        self.dim_order = dims_ordered
        self.ndim = len(dims_ordered)
        self.data = data
        self.climo = climo
        self.time = time
        self.lev = _cnvt_to_float64(lev)
        self.lat = _cnvt_to_float64(lat)
        lon_adjusted = fix_lon(lon)
        self.lon = _cnvt_to_float64(lon_adjusted)
        self.avg_interval = avg_interval
        self.regrid_method = regrid_method
        self.regrid_grid = regrid_grid
        self.esmpy_interp = esmpy_interp
        self.lat_grid = _cnvt_to_float64(lat_grid)
        self.lon_grid = _cnvt_to_float64(lon_grid)
        self.rotated_pole = rotated_pole
        self.cell_area = cell_area

        self._fill_val = fill_val
        self._idx_used_for_sample = sampled

        self._dim_coord_map = {_TIME: self.time,
                               _LEV: self.lev,
                               _LAT: self.lat,
                               _LON: self.lon}

        # TODO: Robert's code flips latitudes so it monotonically increases
        #  Is it necessary here?

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

        # Determine sampling dimension size if any
        if time is not None:
            self.nsamples = len(self.time)
        else:
            self.nsamples = 1
            self.dim_order = list(self.dim_order)
            self.dim_order.insert(0, _TIME)
            self.data = self.data.reshape(1, *self.data.shape)

        self._space_dims = [dim for dim in dims_ordered if dim != _TIME]
        # Spatial shape is left as a list for easy shape combining w/ sampling
        self.space_shp = [len(self._dim_coord_map[dim])
                          for dim in self._space_dims]

        if len(self._space_dims) > 2:
            raise NotImplementedError('Class cannot handle >2D data yet!'
                                      ' spatial shape = '
                                      '{}'.format(self.space_shp))

        # Determine the type of field for this gridded variable
        if not self._space_dims:
            self.type = '0D:time_series'
            self.space_shp = [1]
            self.data = self.data.reshape(self.nsamples, 1)
        elif len(self.space_shp) == 1 and _LAT in self._space_dims:
            self.type = '1D:meridional'
        elif _LAT in self._space_dims and _LON in self._space_dims:
            self.type = '2D:horizontal'
        elif _LAT in self._space_dims and _LEV in self._space_dims:
            self.type = '2D:meridional_vertical'
        else:
            raise NotImplementedError('Unrecognized dimension combination. '
                                      'This type of variable has not been '
                                      'implemented yet.')

    def save(self, filename):
        """
        Save gridded data object to file.  Creats a PyTables HDF5 file
        for the data and the gridded variable object.

        Parameters
        ----------
        filename: str
            Absolute path to the saving file.

        Returns
        -------
        None

        Notes
        -----
        If the file exists, it is opened in append mode.  This can probably
        result in very large files if resaving the same variable multiple
        times.
        """
        avg_interval = self.avg_interval
        regrid_method = self.regrid_method
        regrid_grid = self.regrid_grid
        esmpy_interp = self.esmpy_interp

        # create the path to save data within the HDF5 file
        path_pieces = [avg_interval, regrid_method, regrid_grid, esmpy_interp]
        path_pieces = [str(piece) for piece in path_pieces if piece is not None]

        data_path = join('/', *path_pieces)

        print(('Saving pre-processed data file: {}'.format(filename)))
        print(('Storing data under HDF5 path: {}'.format(data_path)))

        obj_node_name = self.PRE_PROCESSED_OBJ_NODENAME
        data_node_name = self.PRE_PROCESSED_DATA_NODENAME

        if os.path.exists(filename):
            mode = 'a'
        else:
            mode = 'w'

        with tb.open_file(filename, mode=mode,
                          filters=tb.Filters(complib='blosc',
                                             complevel=2)) as h5f:

            # If the node exists already at the path remove it, else create
            # the group
            try:
                h5f.get_node(data_path, name=obj_node_name)
                h5f.remove_node(data_path, name=obj_node_name)
                h5f.remove_node(data_path, name=data_node_name)
            except tb.NoSuchNodeError:
                root, name = os.path.split(data_path)
                h5f.create_group(root, name=name, createparents=True)

            obj_out = h5f.create_vlarray(data_path, name=obj_node_name,
                                         atom=tb.ObjectAtom())

            # Save the data to a CArray
            self.nan_to_fill_val()
            var_to_hdf5_carray(h5f, data_path, data_node_name, self.data)
            self.fill_val_to_nan()

            # Save self object to a VLArray (
            tmp_dat = self.data
            del self.data
            obj_out.append(self)
            self.data = tmp_dat

    def print_data_stats(self):
        """
        Print stats of the data contained in the object.
        **Don't call on large data! You may run out of memory**

        Returns
        -------
        None
        """
        print(('{}: Global: mean={:1.3e}, '
               'std-dev:={:1.3e}'.format(self.name, np.nanmean(self.data),
                                         np.nanstd(self.data))))

    def regrid(self, regrid_method, regrid_grid=None, grid_def=None,
               interp_method=None):
        """
        Regrid data in gridded object.  Only works for 2D:horizontal data

        Parameters
        ----------
        regrid_method: str
            Key indicating regridding package to use.  Allowed: 'simple',
            'sperical_harmonics', and 'esmpy'.
        regrid_grid: str, optional
            Key indicating the destination grid for spherical harmonics.
        grid_def: dict, optional
            Grid definition dictionary from grid_def.yml for ESMPy regridding
        interp_method: str, optional
            Interpolation method to use in ESMPy.  Allowed: bilinear, patch

        Returns
        -------
        GriddedVariable
            New gridded variable object with regridded data.
        """
        assert self.type == '2D:horizontal'
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
        """
        Convert fill value to NaN

        Returns
        -------
        None
        """
        convert_to_masked_array = False

        # Steps through the data in chunks to handle instances where data is
        # very large.  Slower, but doesn't go into swap ;)
        step = 10
        for i in np.arange(0, len(self.data), step=step):
            tmp_data = self.data[i:i+step]
            mask = tmp_data == self._fill_val

            # Determine if invalid data and set flag to convert to
            # np.ma.MaskedArray
            if np.any(mask):
                if not convert_to_masked_array:
                    convert_to_masked_array = True
                tmp_data[mask] = np.nan
                self.data[i:i+step] = tmp_data
        if convert_to_masked_array:
            pass
            # self.data = np.ma.masked_invalid(self.data)

    def nan_to_fill_val(self):
        """
        Convert NaN to fill value.

        Returns
        -------
        None
        """
        if np.ma.is_masked(self.data):
            self.data = self.data.filled(fill_value=self._fill_val)
        else:
            step = 10
            for i in np.arange(0, len(self.data), step=step):
                tmp_dat = self.data[i:i+step]
                tmp_dat[np.isnan(tmp_dat)] = self._fill_val
                self.data[i:i+step] = tmp_dat

        # self.data[~da.isfinite(self.data)] = self._fill_val

    def flattened_spatial(self):
        """
        Get a flattened spatial field representation of the data. Preserves
        sampling dimension.

        Returns
        -------
        flat_data: ndarray
            Flattened view of the data array
        flat_coords: dict{str: ndarray}
            Flattened full coordinate grids for each spatial dimension.
            Shape will match flat_data shape.
        """
        flat_data = self.data.reshape(len(self.time),
                                      np.product(self.space_shp))

        # Get dimensions of data
        coords = [self._dim_coord_map[key] for key in self._space_dims]
        grids = np.meshgrid(*coords, indexing='ij')
        flat_coords = {dim: grid.flatten()
                       for dim, grid in zip(self._space_dims, grids)}

        return flat_data, flat_coords

    def random_sample(self, nens, seed=None):
        """
        Take a random sample along the sampling dimension of the data.

        Parameters
        ----------
        nens: int
            Size of sample
        seed: int, optional
            Seed for the random number generator

        Returns
        -------
        GriddedVariable
            New gridded variable object with the sampled data.

        """
        sample_range = list(range(self.data.shape[0]))
        random.seed(seed)
        sample = random.sample(sample_range, nens)
        return self.sample_from_idx(sample)

    def sample_from_idx(self, sample_idxs):
        """
        Take a specified sample along the sampling dimension of the data.

        Parameters
        ----------
        sample_idxs: list[int]
            A list of indices to take along the sampling dimension of the data

        Returns
        -------
        GriddedVariable
            New gridded variable object with the sampled data
        """

        cls = type(self)
        nsamples = len(sample_idxs)

        if nsamples == self.data.shape[0]:
            print ('Size of sample and total number of available members are '
                   'equivalent.  No resampling performed...')
            return self

        print(('Random selection of {} ensemble members'.format(nsamples)))

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
        """
        Return whether data in the current object is from a sampling
        operation.

        Returns
        -------
        bool
        """

        if self._idx_used_for_sample is None:
            return False
        else:
            return True

    def convert_to_anomaly(self, climo=None):
        """
        Center data by removing climatological mean.

        Parameters
        ----------
        climo: ndarray, optional
            Climatological reference to center data to.  If not provided,
            the climatology is determined across the entire sampling
            dimension.

        Returns
        -------
        None
        """
        print('Removing temporal mean for every gridpoint...')
        if climo is None:
            self.climo = self.data[:].mean(axis=0, keepdims=True)
        else:
            self.climo = climo

        self.data = self.data - self.climo

    def convert_to_standard(self):
        """
        Add back climatology to centered data.

        Returns
        -------
        None
        """
        print('Adding temporal mean to every gridpoint...')
        if self.climo is None:
            raise ValueError('Cannot convert to standard state data is not an '
                             'anomaly to start.')

        self.data = self.data + self.climo
        self.climo = None

    def forecast_var_to_pylim_dataobj(self):
        """
        Create a pyLIM data object for use in LIM forecasting.

        Returns
        -------
        pylim.DataTools.BaseDataObject
            Data object for a LIM that has the same dimensions.
        """

        print(('Converting ForecastVariable to pylim.DataObject: '
               '{}'.format(self.name)))

        BDO = DT.BaseDataObject

        key_map = {_TIME: BDO.TIME,
                   _LEV: BDO.LEVEL,
                   _LAT: BDO.LAT,
                   _LON: BDO.LON}

        dim_coords = {key_map[dim]: (i, getattr(self, dim)[:])
                      for i, dim in enumerate(self.dim_order)}
        coord_grids = {}
        if self.lat_grid is not None:
            coord_grids[BDO.LAT] = self.lat_grid
        if self.lon_grid is not None:
            coord_grids[BDO.LON] = self.lon_grid
        if not coord_grids:
            coord_grids = None

        new_dobj = DT.BaseDataObject(self.data,
                                     dim_coords=dim_coords,
                                     coord_grids=coord_grids,
                                     force_flat=True,
                                     fill_value=self._fill_val,
                                     cell_area=self.cell_area)

        return new_dobj

    @classmethod
    def load(cls, gridded_config, varname, anomaly=False, sample=None,
             **kwargs):
        """
        Load a single variable as a GriddedVariable

        Parameters
        ----------
        gridded_config: LMR_Config.prior
            Configuration definition object for prior variable
        varname: str
            The name of the variable to load
        anomaly: bool, Optional
            Whether to convert data to an anomaly format.
        sample: list[int], Optional
            List of integer indices representing a sample to be taken over the
            time dimension
        nens: int, Optional
            The number of ensemble members to randomly sample from the data
        seed: int, Optional
            Seed for the random number generator.  Only used when nens is 
            specified.
        detrend: bool, Optional
            Flag specifying whether or not to detrend the data along the 
            sampling dimension.

        Returns
        -------
        GriddedVariable

        """
        file_dir = gridded_config.datadir
        file_name = gridded_config.datafile
        file_type = gridded_config.dataformat
        save = gridded_config.save_pre_avg_file
        ignore_pre_avg = gridded_config.ignore_pre_avg_file
        avg_interval = gridded_config.avg_interval
        avg_interval_kwargs = gridded_config.avg_interval_kwargs
        regrid_method = gridded_config.regrid_method
        regrid_grid = gridded_config.regrid_grid

        if isinstance(gridded_config.esmpy_interp_method, dict):
            interp_method = gridded_config.esmpy_interp_method[varname]
        else:
            interp_method = gridded_config.esmpy_interp_method

        esmpy_kwargs = {'grid_def': gridded_config.esmpy_grid_def,
                        'interp_method': interp_method}

        unique_cfg_kwargs = cls._load_unique_cfg_kwargs(gridded_config)
        for key, arg in kwargs.items():
            if key in unique_cfg_kwargs:
                unique_cfg_kwargs[key] = arg
            else:
                raise KeyError('Unrecognized keyword argument provided '
                               'to load function: {}'.format(key))

        datainfo = gridded_config.datainfo
        if 'rotated_pole' in list(datainfo.keys()):
            rotated_pole = varname in datainfo['rotated_pole']
        else:
            rotated_pole = False

        if datainfo['cell_area'] is not None:
            for realm_key, realm_val in datainfo['var_realm_def'].items():
                if realm_key in varname:
                    realm = realm_val
                    break
            else:
                raise ValueError('Realm specification in datasets.yml could '
                                 'not be found in variable name.')

            cella_template = datainfo['cell_area_template']
            cella_realm_def = datainfo['cell_area_realmvar_def'][realm]

            cell_area_file = datainfo['cell_area']
            cell_area_file = cell_area_file.replace(cella_template,
                                                    cella_realm_def)
        else:
            cell_area_file = None

        if datainfo['template'] is not None:
            file_name = file_name.replace(datainfo['template'], varname)
            varname = varname.split('_')[0]

        return cls._main_load_helper(file_dir, file_name, varname, file_type,
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
                                     cell_area_file=cell_area_file,
                                     **unique_cfg_kwargs)

    @staticmethod
    def _load_unique_cfg_kwargs(config):
        """
        Grab configuration keyword arguments that are specific to the gridded
        class.

        Parameters
        ----------
        config: LMR_config.prior
            Configuration object for the prior class.

        Returns
        -------
        cfg_kwargs:
            Special keyword arguments for the current gridded class.
        """
        return {}

    @classmethod
    def _main_load_helper(cls, file_dir, file_name, varname, file_type,
                          nens=None, seed=None, sample=None,
                          avg_interval=None, avg_interval_kwargs=None,
                          regrid_method=None, regrid_grid=None,
                          esmpy_kwargs=None,
                          data_req_frac=0.0, save=True,
                          ignore_pre_avg=False, rotated_pole=False,
                          anomaly=True, detrend=False,
                          cell_area_file=None):

        """
        Main helper for deciding which loading function to use based on the
        data.  Resampling and regridding operations are decided in this
        method.
        """

        # Get correct loader class for specified filetype.
        try:
            ftype_loader = cls.get_loader_for_filetype(file_type)
        except KeyError:
            raise TypeError('Specified file type not supported yet.')

        # Try to load pre-averaged data if it exists.  Otherwise, use the
        # specific loader for the filetype
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
        except (IOError, tb.exceptions.NoSuchNodeError):
            print(('No pre-averaged file found ({}) or '
                   'ignore specified ... '.format(varname)))
            var_obj = ftype_loader(file_dir, file_name, varname, save=save,
                                   data_req_frac=data_req_frac,
                                   avg_interval=avg_interval,
                                   avg_interval_kwargs=avg_interval_kwargs,
                                   rotated_pole=rotated_pole,
                                   anomaly=anomaly,
                                   detrend=detrend,
                                   cell_area_file=cell_area_file)
            print('Loaded from file: {}/{}'.format(file_dir, file_name))

        var_obj.fill_val_to_nan()

        # Do regridding and save if specified
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

        # Sample the data
        if not var_obj.is_sampled() and (nens is not None or sample is not None):

            if sample is not None:
                var_obj = var_obj.sample_from_idx(sample)
            else:
                var_obj = var_obj.random_sample(nens, seed)

        return var_obj

    @classmethod
    def get_loader_for_filetype(cls, file_type):
        """
        Retrieve the correct function for loading specific filetypes
        Parameters
        ----------
        file_type: str
            Key for the file type to get loader for.

        Returns
        -------
        Method that will load data for the given file type
        """
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
            print(('Found node for avg_interval path: {}'.format(obj_dir)))

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
                    print(('Found node for regridded data under path: '
                           '{}'.format(regrid_obj_dir)))
                    obj = regrid_obj
                    obj_data = regrid_obj_data
                except tb.NoSuchNodeError:
                    # Do not sample, since regrid specified and might save
                    do_sample = False
                    obj_data = obj_data.read()
                    print(('Regridded pre-processed grid object not found for '
                          'regridding: {}.'.format(regrid_obj_dir)))

            obj.data = obj_data

            if anomaly and obj.climo is None:
                obj.convert_to_anomaly()

            # Sampling done to pre-average data to take advantage of lazy
            # loading.  Sampling will be ignored by the main loader in this
            # case.
            if do_sample:
                if sample is not None:
                    obj = obj.sample_from_idx(sample)
                elif nens is not None:
                    obj = obj.random_sample(nens, seed)
                else:
                    obj.data = obj.data.read()

        print('Loaded pre-averaged file: {}'.format(path))
        return obj

    @classmethod
    def _load_from_netcdf(cls, dir_name, filename, varname, avg_interval=None,
                          avg_interval_kwargs=None, save=False,
                          data_req_frac=None, rotated_pole=False,
                          anomaly=False, detrend=False, cell_area_file=None):
        """
        General structure for from netCDF:
        1. Load data and information about dimensions
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

            # Convert to dimension key names defined in _DEFAULT_DIM_ORDER
            # TODO: Handle member dimensions for ensembles
            dims = []
            dim_keys = []
            dim_exclude = []
            for i, dim in enumerate(var.dimensions):
                dim = dim.lower()
                if data_shp[i] == 1:
                    dim_exclude.append(dim)
                elif dim in _DEFAULT_DIM_ORDER:
                    dims.append(dim)
                    dim_keys.append(dim)
                elif dim in _BYPASS_DIMENSION_DEFS:
                    dims.append(_BYPASS_DIMENSION_DEFS[dim])
                    dim_keys.append(_BYPASS_DIMENSION_DEFS[dim])
                elif dim in _ALT_DIMENSION_DEFS:
                    dims.append(_ALT_DIMENSION_DEFS[dim])
                    dim_keys.append(dim)
                else:
                    raise KeyError('Dimension, {}, not found in '
                                   'definitions of LMR_gridded.'.format(dim))

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
            dim_vals = {k: val[:] for k, val in dim_vals.items()}

            # Extract single dimension from irregularly spaced rotated pole
            # grids the netCDF files I've encountered have 2D dimensions
            # for this case
            if rotated_pole:
                lat_grid = dim_vals[_LAT]
                lon_grid = dim_vals[_LON]
                dim_vals[_LAT] = dim_vals[_LAT][..., 0]
                dim_vals[_LON] = dim_vals[_LON][..., 0, :]
            else:
                lat_grid = None
                lon_grid = None

            # Load the cell area for regridding purposes
            if cell_area_file is not None:
                try:
                    cell_area_path = join(dir_name, cell_area_file)
                    cell_f = Dataset(cell_area_path, 'r')
                    cell_area_varname = cell_area_file.split('_')[0]
                    cell_area = cell_f.variables[cell_area_varname][:]
                except IOError:
                    if rotated_pole:
                        raise IOError('Cell area file could not be loaded. '
                                      'Cell area is required for regridding '
                                      'procedures when using rotated pole '
                                      'grids.')

                    print('No cell area file designated for: '
                          '{}'.format(varname))
                    cell_area = None
            else:
                cell_area = None

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
                           cell_area=cell_area, **dim_vals)

            if anomaly:
                grid_obj.convert_to_anomaly()

            if detrend:
                # TODO Detrend
                pass

            if save:
                new_dir = join(dir_name, pre_proc_filedir)
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)

                pre_proc_fname = join(new_dir, filename + pre_proc_filetag)
                grid_obj.save(pre_proc_fname)

            return grid_obj

    @staticmethod
    def _netcdf_datetime_convert(time_var):
        """
        Converts netcdf time variable into date-times.

        Used as a static method in case necesary to overwrite with subclass
        :param time_var:
        :return:
        """
        if not hasattr(time_var, 'calendar'):
            cal = 'ISO8601'  # Default CDM calendar
        else:
            cal = time_var.calendar

        try:
            time = num2date(time_var[:], units=time_var.units,
                            calendar=cal)
            return time
        except ValueError:
            # num2date needs calendar year start >= 0001 C.E. but some fields
            # start at year 0000 C.E. (bug submitted to unidata about this)
            warnings.warn('Detected invalid unit specification for num2date'
                          ' when converting netCDF time dimension.'
                          ' If using time values please confirm converted'
                          ' datetimes are reasonable.')

            tunits = time_var.units

            # expecting format of '<units> since YYYY-MM-DD'
            # calculate the time delta in years for the updated units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 1

            # Shift YYYY from 0 -> 1 and account for it in date time creation
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

        """
        Resample data to specified averaging period.  Assumes contiguous
        intevals are being used to resample.

        Parameters
        ----------
        time_vals: ndarray
            Sampling dimension values that will be updated along with
            the data re-averaging
        data: ndarray
            Data to re-average
        nelem_in_yr: int, Optional
            Number of elements that comprise a single year along the sampling
            dimension. Defaults to 12
        elem_to_avg: tuple(int), Optional
            List of subannual elements included in the average.  These are
            assumed to be contiguous.
        nyears: int, Optional
            Number of years to include in the average. Defaults to 1.
        data_req_frac: float, Optional
            Fraction of valid data (between 0.0 and 1.0) required for
            average to be taken over data.  Only considered if invalid
            values are encountered in data.

        Returns
        -------
        new_times: ndarray
            Time values corresponding to the re-averaged data
        new_data: ndarray
            Re-averaged data

        Notes
        -----
        This operation will remove partial averages at the beginning and end
        of the samples reducing the number of total samples by 1.
        """

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
        new_times = time_vals[:, 0, 0]  # Definition is start of time period
        data = data[:, :, 0:len_of_sample]

        # Average over multi-annual and annual (with sub-annual specification)
        # dimensions in chunks to keep memory usage reasonable.
        data_list = []
        for i in np.arange(0, data.shape[0], 10):
            data_chk = np.nanmean(data[i:i+10], axis=(1,2))
            data_list.append(data_chk)

        new_data = np.concatenate(data_list, axis=0)

        # Mask times which did not have enough data in the average
        if data_req_frac is not None and np.any(np.isnan(data)):
            nelem_in_avg = nyears*nelem_in_yr
            num_valid = np.isfinite(data).sum(axis=(1, 2))
            valid_frac = num_valid.astype(np.float) / nelem_in_avg
            invalid = valid_frac < data_req_frac
            new_data[invalid] = np.nan

        return new_times, new_data


class PriorVariable(GriddedVariable):

    """
    Gridded variable with load functions defined for handling variables used
    for the prior.
    """

    @staticmethod
    def _load_unique_cfg_kwargs(config):
        """
        Grab configuration keyword arguments that are specific to the gridded
        class.

        Parameters
        ----------
        config: LMR_config.ConfigObject
            Configuration object for the prior class.

        Returns
        -------
        cfg_kwargs:
            Special keyword arguments for the current gridded class.
        """
        unique_kwargs = {'detrend': config.detrend,
                         'nens': config.nens,
                         'seed': config.seed}
        return unique_kwargs

    @classmethod
    def load_allvars(cls, prior_config):
        """
        Load all variables specified in the prior configuration

        Parameters
        ----------
        prior_config: LMR_Config.prior
            Configuration definition object for prior variables

        Returns
        -------
        dict{tuple of str: PriorVariable}
            A dictionary with key value pairs of the variable name/avg_interval
            and the PriorVariable instance.

        """
        var_names = prior_config.state_variables
        avg_interval = prior_config.avg_interval

        prior_dict = OrderedDict()
        for vname, anomaly in var_names.items():
            if anomaly == 'anom':
                anomaly = True
            else:
                anomaly = False
            pobj = cls.load(prior_config, vname, anomaly=anomaly)

            prior_dict[(vname, avg_interval)] = pobj

        return prior_dict

    @classmethod
    def load_psm_required_vars(cls, prior_config, varkey_avg_intervals):
        """
        Load the prior fields required for the PSM to work.

        Parameters
        ----------
        prior_config: LMR_config.ConfigObject
            Configuration instance for prior variables
        varkey_avg_intervals: dict
            Dictionary of variable name keys and associated averging interval
            names required for the PSM

        Returns
        -------
        dict{tuple of str: PriorVariable}
            A dictionary with key value pairs of the variable name/avg_interval
            and the PriorVariable instance.
        """

        orig_avg_interval = prior_config.avg_interval
        loaded_state_vars = prior_config.state_variables

        psm_req_prior_dict = OrderedDict()
        for varkey, avg_interval_names in varkey_avg_intervals.items():
            for avg_interval in avg_interval_names:
                if varkey in loaded_state_vars and avg_interval == orig_avg_interval:
                    continue

                prior_config.update_avg_interval(avg_interval)
                pobj = cls.load(prior_config, varkey, anomaly=True)

                psm_req_prior_dict[(varkey, avg_interval)] = pobj

        prior_config.update_avg_interval(orig_avg_interval)

        return psm_req_prior_dict


class ForecasterVariable(GriddedVariable):

    """
    Gridded variable with load functions defined for handling variables used
    for forecasting.
    """

    @classmethod
    def load_all(cls, forecaster_cfg, load_keys):

        fcast_dict = OrderedDict()
        for varname, avg_inteval in load_keys:
            forecaster_cfg.update_avg_interval(avg_inteval)
            fobj = cls.load(forecaster_cfg, varname, anomaly=True)
            fcast_dict[(varname, avg_inteval)] = fobj

        return fcast_dict

    @classmethod
    def load_all_cfg_vars_only(cls, forecaster_cfg, state_keys):

        var_names = forecaster_cfg.fcast_varnames
        prior_mapping = forecaster_cfg.prior_mapping

        var_to_load = []
        # find which avg_intervals to use for selected forecast vars
        for var in var_names:
            prior_var_key = prior_mapping[var]
            matches = [(state_var, state_avg_interval)
                       for state_var, state_avg_interval in state_keys
                       if state_var == prior_var_key]
            var_to_load += matches

        return cls.load_all(forecaster_cfg, var_to_load)


class AnalysisVariable(GriddedVariable):

    @staticmethod
    def _load_unique_cfg_kwargs(config):
        """
        Grab configuration keyword arguments that are specific to the gridded
        class.

        Parameters
        ----------
        config: LMR_config.ConfigObject
            Configuration object for loading gridded analysis data.

        Returns
        -------
        cfg_kwargs:
            Special keyword arguments for the current gridded class.
        """
        unique_kwargs = {'detrend': config.detrend}
        return unique_kwargs

    @classmethod
    def load_allvars(cls):
        raise NotImplementedError()


class BerkeleyEarthAnalysisVariable(AnalysisVariable):

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

    def __init__(self, prior_vars, psm_var_map=None):

        self._prior_vars = prior_vars
        state = []
        self.var_coords = {}
        self.var_view_range = {}
        self.var_space_shp = {}
        self.var_cell_area = {}
        self.var_keys = []
        self.augmented = False

        # Attr for backend storage
        self.output_backend = None
        self._orig_state = None
        self._tmp_state = {}

        self.len_state = 0
        for var, pobj in prior_vars.items():
            self.var_keys.append(var)
            self.var_space_shp[var] = pobj.space_shp

            var_start = self.len_state
            flat_data, flat_coords = pobj.flattened_spatial()
            var_end = flat_data.T.shape[0] + var_start
            self.var_view_range[var] = (var_start, var_end)
            self.len_state = var_end
            state.append(flat_data.T)
            self.var_coords[var] = flat_coords
            self.var_cell_area[var] = pobj.cell_area

        self.state = np.concatenate(state, axis=0)
        self.shape = self.state.shape
        self.old_state_info = self.get_old_state_info()
        self.psm_var_map = psm_var_map

    @classmethod
    def from_config(cls, prior_config, req_avg_intervals=None):
        pvars = PriorVariable.load_allvars(prior_config)

        if req_avg_intervals is not None:
            req_psm_pvars = PriorVariable.load_psm_required_vars(prior_config,
                                                                 req_avg_intervals)
            pvars.update(req_psm_pvars)

        psm_var_map = prior_config.psm_var_map

        return cls(pvars, psm_var_map=psm_var_map)

    def get_var_data(self, var_name):
        """
        Returns a view of the variable in the state vector
        """
        start, end = self.var_view_range[var_name]
        var_data = self.state[start:end]

        return var_data

    def get_psm_var_key(self, psm_vartype_dict):

        psm_type, psm_varkey = list(psm_vartype_dict.items())[0]
        prior_var_key = self.psm_var_map[psm_type][psm_varkey]

        return prior_var_key

    # TODO: Decide whether this should only be handled at prior level
    def truncate_state(self):
        """
        Create a truncated copy of the current state
        """
        trunc_pvars = OrderedDict()
        for var_name, pvar in self._prior_vars.items():
            trunc_pvars[var_name] = []
            for pobj in pvar:
                if pobj.type == 'horizontal':
                    trunc_pvars[var_name].append(pobj.truncate())
                else:
                    trunc_pvars[var_name].append(pobj)
        state_class = type(self)
        return state_class(trunc_pvars)

    def augment_state(self, ye_vals):

        # Add leading axis and repeat to match state_list length
        nproxies = len(ye_vals)
        self.state = np.concatenate((self.state, ye_vals), axis=0)
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

        xb_vals = self.state
        xb_vals[:] = regular_cov_infl(xb_vals, inf_factor)

    def reset_augmented_ye(self, ye_vals):

        ye_state = self.get_var_data('ye_vals')
        ye_state[:] = ye_vals

    def update_var_data(self, var_update_dict):

        for var_key, var_data in var_update_dict.items():
            data_view = self.get_var_data(var_key)
            data_view[:] = var_data

    def stash_state(self, name):

        self._tmp_state[name] = self.state.copy()

    def stash_recall_state_list(self, name, pop=False, copy=False):

        if name in self._tmp_state:
            if pop:
                state = self._tmp_state.pop(name)
            else:
                state = self._tmp_state[name]

                if copy:
                    state = state.copy()

            self.state = state
        else:
            raise KeyError('No currently stashed state with name {}....'.format(name))

    def stash_pop_state_list(self, name):
        self.stash_recall_state_list(name, pop=True)

    def restore_orig_state(self):
        self.state = self._orig_state.copy()

    def get_old_state_info(self):

        state_info = {}
        for var in list(self.var_view_range.keys()):
            var_info = {'pos': self.var_view_range[var],
                        'vartype': self._prior_vars[var].type}

            space_dims = [dim for dim in _DEFAULT_DIM_ORDER
                          if dim in list(self.var_coords[var].keys())]

            var_info['spacecoords'] = space_dims
            if not space_dims:
                var_info['spacedims'] = None
            else:
                var_info['spacedims'] = self.var_space_shp[var]
            state_info[var] = var_info

        return state_info

    def initialize_storage_backend(self, btype, nyears, fdir):
        warnings.warn("deprecated", DeprecationWarning)

        _types = {'NPY': _NPYStateStorage,
                  'H5': _HDF5StateStorage}

        backend_class = _types[btype]
        store_orig = True

        if self._orig_state is not None:
            store_orig = False
            self.stash_state('tmp')
            self.restore_orig_state()

        # Extra year, just in case we need to use shifted dates
        self.output_backend = backend_class(nyears+1, self, fdir=fdir)

        self.output_backend.insert(self.state_list, 0)

        if store_orig:
            self._orig_state = self.state_list.copy()
        else:
            self.stash_pop_state_list('tmp')

    def insert_upcoming_prior(self, curr_yr_idx, use_curr=False):
        warnings.warn("deprecated", DeprecationWarning)

        if not use_curr:
            dat = self._orig_state
        else:
            dat = self.output_backend.get_xb(0, curr_yr_idx)

        self.output_backend.insert(dat, curr_yr_idx+1)

    def xb_from_backend(self, yr_idx, res, shift):
        warnings.warn("deprecated", DeprecationWarning)

        self.state_list = self.output_backend.get_xb(shift, yr_idx)
        self.resolution = self.base_res
        self.avg_to_res(res, 0)

    def propagate_avg_to_backend(self, yr_idx, shift):
        warnings.warn("deprecated", DeprecationWarning)

        self.output_backend.propagate_avg_to_storage(shift, self, yr_idx)

    def close_xb_container(self):
        warnings.warn("deprecated", DeprecationWarning)
        self.output_backend.close()


class _BaseStateStorage(object, metaclass=ABCMeta):
    """
    Class for storing state vector data
    """

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

        for i in range(nchunks):
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
