import sys
sys.path.append('../')

import pytest
import LMR_utils2 as Utils
import netCDF4 as ncf
import numpy as np

@pytest.fixture(scope='module')
def ncf_data(request):
    f_obj = ncf.Dataset('data/gridded_dat.nc', 'r')

    def fin():
        f_obj.close()

    request.addfinalizer(fin)
    return f_obj


@pytest.mark.parametrize("doc", [
    """This is class docstring""",
    None])
def test_class_doc_inherit(doc):
    class foo:
        __doc__ = doc
        pass

    @Utils.class_docs_fixer
    class bar(foo):
        pass

    assert bar.__doc__ == doc


@pytest.mark.parametrize("doc", [
    """This is func docstring""",
    None])
def test_function_doc_inherit(doc):
    class foo:
        def lol(self):
            pass

    foo.lol.__func__.__doc__ = doc

    @Utils.class_docs_fixer
    class bar(foo):
        def lol(self):
            pass

    assert bar.lol.__func__.__doc__ == doc


def test_function_doc_augment():
    parent_doc = """This is the parents lol docstr"""
    child_doc = """%%aug%%
            The childs doc is here
            """

    class foo:
        def lol(self):
            pass

    foo.lol.__func__.__doc__ = parent_doc

    @Utils.class_docs_fixer
    class bar(foo):
        @Utils.augment_docstr
        def lol(self):
            """%%aug%%
            The childs doc is here
            """
            pass

    assert bar.lol.__func__.__doc__ == (parent_doc +
                                        child_doc.replace('%%aug%%', ''))


def test_generate_latlon_bnd_limits():
    # TODO: could be parametrized input
    # Defaults
    Utils.generate_latlon(5, 5)

    # Bad lat bounds
    with pytest.raises(ValueError):
        Utils.generate_latlon(5, 5, lat_bnd=(-100, 45))
    with pytest.raises(ValueError):
        Utils.generate_latlon(5, 5, lat_bnd=(-45, 91))

    # Bad lon bounds
    Utils.generate_latlon(5, 5, lon_bnd=(-90, 270))

    with pytest.raises(ValueError):
        Utils.generate_latlon(5, 5, lon_bnd=(-180, 181))
    with pytest.raises(ValueError):
        Utils.generate_latlon(5, 5, lon_bnd=(-181, 40))
    with pytest.raises(ValueError):
        Utils.generate_latlon(5, 5, lon_bnd=(14, 361))


def test_generate_latlon_output_shp():
    nlats = 4
    nlons = 5

    lats, lons, clats, clons = Utils.generate_latlon(nlats, nlons)
    assert lats.shape == (4, 5)
    assert lons.shape == (4, 5)
    assert clats.shape == (5,)
    assert clons.shape == (6,)


def test_generate_latlon_center_corner():
    lats, lons, clats, clons = Utils.generate_latlon(4,5,
                                                     lat_bnd=(-45, 45),
                                                     lon_bnd=(0, 180))

    np.testing.assert_equal(lats[:, 0], [-33.75, -11.25, 11.25, 33.75])
    np.testing.assert_equal(lons[0], [0, 36, 72, 108, 144])
    np.testing.assert_equal(clats, [-45, -22.5, 0, 22.5, 45])
    np.testing.assert_equal(clons, [-18, 18, 54, 90, 126, 162])


def test_generate_latlon_include_lat_endpts():
    lats, lons, clats, clons = Utils.generate_latlon(3, 5, include_endpts=True)
    np.testing.assert_equal(lats[:, 0], [-90, 0, 90])
    assert clats[0] == -90
    assert clats[-1] == 90

    lats, lons, clats, clons = Utils.generate_latlon(4, 5, include_endpts=True)
    np.testing.assert_equal(lats[:, 0], [-90, -30, 30, 90])


def test_calc_latlon_bnd_1d_input():
    test_data = np.linspace(10, 50, 5)
    with pytest.raises(ValueError):
        _ = Utils.calculate_latlon_bnds(test_data[:, None], test_data)
    with pytest.raises(ValueError):
        _ = Utils.calculate_latlon_bnds(test_data, test_data[:, None])


def test_calc_latlon_bnd_monotonic():
    test_data = np.linspace(0, 10, 11)
    with pytest.raises(ValueError):
        _ = Utils.calculate_latlon_bnds(test_data[::-1], test_data)
    with pytest.raises(ValueError):
        _ = Utils.calculate_latlon_bnds(test_data, test_data[::-1])


def test_calc_latlon_bnd_regular_grid():
    irregular_data = np.array([1, 2, 3, 5, 8, 13, 21], dtype=np.float32)
    regular_data = np.arange(10)
    irreg_bnds = [0.5, 1.5, 2.5, 4, 6.5, 10.5, 17, 25]
    reg_bnds = np.arange(11) - 0.5

    lat_bnds, lon_bnds = Utils.calculate_latlon_bnds(regular_data, irregular_data)
    np.testing.assert_equal(lat_bnds, reg_bnds)
    np.testing.assert_equal(lon_bnds, irreg_bnds)

    lat_bnds, lon_bnds = Utils.calculate_latlon_bnds(irregular_data, regular_data)
    np.testing.assert_equal(lat_bnds, irreg_bnds)
    np.testing.assert_equal(lon_bnds, reg_bnds)


def test_calc_latlon_bnd_bounds():
    lat_data = np.array([-33.75, -11.25, 11.25, 33.75])
    lon_data = np.array([18, 54, 90, 126, 162])

    lat_bnds, lon_bnds = Utils.calculate_latlon_bnds(lat_data, lon_data)

    np.testing.assert_equal(lat_bnds, [-45, -22.5, 0, 22.5, 45])
    np.testing.assert_equal(lon_bnds, [0, 36, 72, 108, 144, 180])


def test_calc_latlon_bnd_bounds_half_shift():
    lat_data = np.array([-90, -60, -30, 0, 30, 60, 90])
    lon_data = np.array([0, 90, 180, 270])

    lat_bnds, lon_bnds = Utils.calculate_latlon_bnds(lat_data, lon_data)

    np.testing.assert_equal(lat_bnds, [-90, -75, -45, -15, 15, 45, 75, 90])
    np.testing.assert_equal(lon_bnds, [-45, 45, 135, 225, 315])



def test_global_mean2(ncf_data):

    dat = ncf_data['air'][0:4]
    lat = ncf_data['lat'][:]
    lon = ncf_data['lon'][:]

    longrid, latgrid = np.meshgrid(lon, lat)

    gm_time, _, _ = Utils.global_hemispheric_means(dat, lat)
    gm0, _, _ = Utils.global_hemispheric_means(dat[0], lat)

    # with time
    gm_test = Utils.global_mean2(dat, lat)
    np.testing.assert_allclose(gm_test, gm_time)

    # flattened lat w/ time
    flat_dat = dat.reshape(4, 94*192)
    gm_test = Utils.global_mean2(flat_dat, latgrid.flatten())
    np.testing.assert_allclose(gm_test, gm_time)

    # no time
    gm_test = Utils.global_mean2(dat[0], lat)
    np.testing.assert_allclose(gm_test, gm0)

    # no time flattened spatial
    gm_test = Utils.global_mean2(dat[0].flatten(), latgrid.flatten())
    np.testing.assert_allclose(gm_test, gm0)

    # NaN values
    dat[:, 0, :] = np.nan
    gm_nan_time, _, _ = Utils.global_hemispheric_means(dat, lat)
    gm_nan_test = Utils.global_mean2(dat, lat)
    np.testing.assert_allclose(gm_nan_test, gm_nan_time)

    # Test hemispheric
    gm_time, nhm_time, shm_time = Utils.global_hemispheric_means(dat, lat)
    gm_test, nhm_test, shm_test = Utils.global_mean2(dat, lat,
                                                     output_hemispheric=True)
    np.testing.assert_allclose(gm_test, gm_time)
    np.testing.assert_allclose(nhm_test, nhm_time)
    np.testing.assert_allclose(shm_test, shm_time)


if __name__ == '__main__':

    tst_dat = ncf.Dataset('data/gridded_dat.nc', 'r')
    test_global_mean2(tst_dat)



