"""
Tests for the loading of gridded variable classes in the LMR framework

Originator: Andre Perkins
            Dept. of Atmos. Sciences, University of Washington
"""

from .. import LMR_gridded
import numpy as np
import pytest

nlat = 40
nlon = 60
ntime = 10


@pytest.fixture(scope='module')
def lat(request):
    lat = np.linspace(-90, 90, nlat, endpoint=True)
    return lat


@pytest.fixture(scope='module')
def lon(request):
    lon = np.linspace(0, 360, nlon)
    return lon


@pytest.fixture(scope='module')
def time(request):
    time = np.arange(ntime)
    return time


@pytest.fixture(scope='module')
def data(request, lat, lon):
    data = np.ones((ntime, nlat, nlon))
    data = data * np.exp(-1 * np.abs(np.deg2rad(lat)))[:, None]
    data = data * np.cos(2 * np.deg2rad(lon))
    return data


@pytest.fixture(scope='function')
def smooth_wave_init(request, data, time, lat, lon):

    return {'time': time, 'lat': lat, 'lon': lon, 'data' : data}


class TestGriddedVariable(object):

    def test_gridvar_init(self, smooth_wave_init):
        """Test basic initialization with minimal data and dimensions"""
        var_obj = LMR_gridded.GriddedVariable('test_data',
                                              ('time', 'lat', 'lon'),
                                              **smooth_wave_init)
        data_shp = smooth_wave_init['data'].shape

        assert var_obj._space_dims == [LMR_gridded._LAT,
                                       LMR_gridded._LON]
        assert var_obj.space_shp == list(data_shp[1:])
        assert var_obj.type == '2D:horizontal'

    def test_wrong_ndim(self, smooth_wave_init):
        """Test number of dimensions data vs. dims_ordered requirement"""
        with pytest.raises(ValueError):
            LMR_gridded.GriddedVariable('test_data',
                                        ('time', 'lat'),
                                        **smooth_wave_init)
        with pytest.raises(ValueError):
            LMR_gridded.GriddedVariable('test_data',
                                        ('time', 'lev', 'lat', 'lon'),
                                        **smooth_wave_init)

    def test_wrong_dimension_length(self, smooth_wave_init):
        """Test for checks that dimensions must match those in the data"""
        data = smooth_wave_init.pop('data')

        for key, item in smooth_wave_init.items():
            with pytest.raises(ValueError):
                smooth_wave_init[key] = item[:-2]
                LMR_gridded.GriddedVariable('test_data',
                                            ('time', 'lat', 'lon'),
                                            data,
                                            **smooth_wave_init)
                smooth_wave_init[key] = item

    def test_dim_specified_but_not_provided(self, smooth_wave_init):
        """Test for checks on a dimension listed but no values provided"""
        smooth_wave_init.pop('lat')

        with pytest.raises(ValueError):
            LMR_gridded.GriddedVariable('test_data',
                                        ('time', 'lat', 'lon'),
                                        **smooth_wave_init)

    def test_no_time_dim(self, smooth_wave_init):
        data = smooth_wave_init.pop('data')
        smooth_wave_init.pop('time')
        var_obj = LMR_gridded.GriddedVariable('test_data',
                                              ('lat', 'lon'),
                                              data[0],
                                              **smooth_wave_init)

        assert var_obj.nsamples == 1
        assert var_obj.dim_order[0] == LMR_gridded._TIME
        assert var_obj.data.shape[0] == 1

    def test_3d_data(self, smooth_wave_init):
        """Check that 3D data gives error because it hasn't been implemented"""
        data = smooth_wave_init.pop('data')
        lev = np.array([1,])

        data = data[:, None, ...]

        with pytest.raises(NotImplementedError):
            LMR_gridded.GriddedVariable('test_data',
                                        ('time', 'lev', 'lat', 'lon'),
                                        data,
                                        lev=lev,
                                        **smooth_wave_init)

    def test_timeseries(self, data, time):
        tseries = data[:, 1, 2]
        var_obj = LMR_gridded.GriddedVariable('test_data',
                                              ('time',),
                                              tseries,
                                              time=time)
        assert var_obj.type == '0D:time_series'
        assert var_obj.space_shp == [1]
        assert var_obj.data.shape == (10, 1)

    def test_meridional(self, data, time, lat):

        var_obj = LMR_gridded.GriddedVariable('test_data',
                                              ('time', 'lat'),
                                              data[:, :, 2],
                                              time=time,
                                              lat=lat)
        assert var_obj.type == '1D:meridional'

    def test_merid_vertical(self, data, time, lat):

        lev = np.array([1000, 900])
        new_data = np.ones((ntime, 2, nlat, nlon))
        new_data = data[:, None, ...] * new_data

        var_obj = LMR_gridded.GriddedVariable('test_data',
                                              ('time', 'lev', 'lat'),
                                              new_data[:, :, :, 2],
                                              time=time,
                                              lat=lat,
                                              lev=lev)

        assert var_obj.type == '2D:meridional_vertical'





