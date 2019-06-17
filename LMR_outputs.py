from LMR_gridded import PriorVariable
from LMR_utils import get_chunk_shape, haversine
from collections import Sequence, namedtuple
from numcodecs import Blosc
import numpy as np
import pickle
import zarr
import warnings

import pylim.Stats as ST
from os.path import join


def prepare_scalar_calculations(scalar_outdef, state, prior_cfg, ntimes, nens,
                                return_insert_func=True):

    func_by_var = {}
    scalar_containers_by_var = {}
    avg_interval = prior_cfg.avg_interval
    for var_name, scalar_measures in scalar_outdef.items():
        var_avg_key = (var_name, avg_interval)
        func_by_measure = {}
        scalar_containers_by_meas = {}

        for measure in scalar_measures:
            container = np.zeros((ntimes, nens))
            scalar_containers_by_meas[measure] = container
            curr_func = _gen_scalar_func(measure, var_name,
                                         state, prior_cfg)
            func_by_measure[measure] = curr_func

        func_by_var[var_avg_key] = func_by_measure
        scalar_containers_by_var[var_avg_key] = scalar_containers_by_meas

    def insert_scalars(state, idx):

        for _varkey, _scalar_measures in func_by_var.items():
            data = state.get_var_data(_varkey)

            for _measure, func in _scalar_measures.items():
                scalar = func(data)
                _container = scalar_containers_by_var[_varkey][_measure]
                _container[idx] = scalar

    if return_insert_func:
        return insert_scalars, scalar_containers_by_var
    else:
        return func_by_var, scalar_containers_by_var


def save_scalar_ensembles(workdir, times, containers_by_var):

    fname = '{}_{}_ens_output'
    for varkey, scalar_measures in containers_by_var.items():
        var_name = varkey[0]
        for measure, container in scalar_measures.items():
            filepath = join(workdir, fname.format(var_name, measure))
            np.savez(filepath, scalar_ens=container, recon_times=times)


def _gen_scalar_func(measure, varname, state, prior_cfg):

    coord_data = state.var_coords[varname]
    lat = coord_data['lat']
    lon = coord_data['lon']
    cell_area = state.var_cell_area[varname]

    use_general_avg = ('glob_mean', 'nino3.4', 'nino3', 'nino4', 'npi',
                       'nh', 'sh', 'natlantic', 'europe')
    use_single_point = ('tahiti', 'darwin')
    use_zonal_avg = ('65s', '45s')

    point_latlons = {'tahiti': (-17.55, 360 - 149.617), # 17.55 S 149.617 W
                     'darwin': (-12.467, 130.85)} # 12.467 S 130.85 E

    zonal_avg_lats = {'65s': -65,
                      '45s': -45}

    if measure in use_general_avg:
        if measure == 'glob_mean':
            region = None
        else:
            region = measure

        func = _get_area_avg_scalar_func(cell_area, lat, lon, region)
    elif measure in use_single_point:
        lat_lon = point_latlons[measure]
        func = _get_single_gridpoint_func(*lat_lon, lat, lon)
    elif measure in use_zonal_avg:
        func = _get_zonal_avg_func(zonal_avg_lats[measure], lat)
    elif measure == 'pdo':
        func = _get_pdo_index_func(prior_cfg, varname)
    else:
        raise KeyError('Unrecognized scalar measure {}'.format(measure))

    return func


def _masked_to_full_space(mask, data):

    full_space = np.zeros_like(mask, dtype=data.dtype)
    full_space[mask] = data

    return full_space


def get_scalar_factor(measure, varname, prior_cfg, lat, lon, cell_area=None):
    # Get scalar factors for matrix multiplication against forecast space output
    # assumes lat/lon reduced to valid data locations

    use_general_avg = ('glob_mean', 'nino3.4', 'nino3', 'nino4', 'npi',
                       'nh', 'sh', 'natlantic', 'europe')
    use_single_point = ('tahiti', 'darwin')
    use_zonal_avg = ('65s', '45s')

    point_latlons = {'tahiti': (-17.55, 360 - 149.617),  # 17.55 S 149.617 W
                     'darwin': (-12.467, 130.85)}  # 12.467 S 130.85 E

    zonal_avg_lats = {'65s': -65,
                      '45s': -45}

    if measure in use_general_avg:
        if measure == 'glob_mean':
            region = None
        else:
            region = measure

        mask, factor = get_area_avg_mask_and_weights(lat, lon, region,
                                                     cell_area=cell_area)
        if mask is not None:
            factor = _masked_to_full_space(mask, factor)
    elif measure in use_single_point:
        lat_lon = point_latlons[measure]
        factor = get_single_gridpoint_factor(*lat_lon, lat, lon)
    elif measure in use_zonal_avg:
        factor = get_zonal_avg_weights(zonal_avg_lats[measure], lat)
    elif measure == 'pdo':
        factor, valid_mask, pdo_mask = _gen_pdo_index_factor(prior_cfg, varname)
        factor = _masked_to_full_space(pdo_mask, factor)
    else:
        raise KeyError('Unrecognized scalar measure {}'.format(measure))

    return factor


def _remove_nan_from_state(state_data):

    nan_locs = np.isnan(state_data)
    if np.any(nan_locs):
        # Sum True values across ensemble members, if any NaNs, we mask it
        valid_data = nan_locs.sum(axis=1) == 0
        state_data = state_data[valid_data]
    else:
        valid_data = None

    return state_data, valid_data


def _get_area_weights(cell_area, lat, mask=None):

    if cell_area is not None:
        weights = cell_area
    else:
        # warnings.warn('No cell area grid provided for area weighting.  If '
        #               'using an irregular grid, latitude weighting will not '
        #               'be correct.')

        # otherwise there'll be cell area loaded for fields.
        weights = np.cos(np.deg2rad(lat))
        weights = weights

    if mask is not None:
        weights = weights[mask]

    weights = weights / weights.sum()

    return weights


def _gen_latlon_grid_mask(lat, lon, latmin, latmax, lonmin, lonmax):

    if lonmin > lonmax:
        # straddling the data line
        lon_mask = (lon >= lonmin) | (lon <= lonmax)
    else:
        lon_mask = (lon >= lonmin) & (lon <= lonmax)

    mask = (lon_mask & (lat >= latmin) & (lat <= latmax))

    return mask


def get_area_avg_mask_and_weights(lat, lon, region, cell_area=None):
    if region == 'nino3.4':
        # 5S - 5N, 170W - 120W
        mask = _gen_latlon_grid_mask(lat, lon, -5, 5, 190, 240)
    elif region == 'nino3':
        # 5S - 5N, 140W - 90W
        mask = _gen_latlon_grid_mask(lat, lon, -5, 5, 210, 270)
    elif region == 'nino4':
        # 5S - 5N, 160E - 140W
        mask = _gen_latlon_grid_mask(lat, lon, -5, 5, 160, 210)
    elif region == 'npi':
        # 30N - 65N, 160E - 130W
        mask = _gen_latlon_grid_mask(lat, lon, 30, 65, 160, 220)
    elif region == 'natlantic':
        # 0 N - 80 N, 75W - 0W
        mask = _gen_latlon_grid_mask(lat, lon, 0, 80, 285, 360)
    elif region == 'nh':
        # 0 - 90 N
        mask = _gen_latlon_grid_mask(lat, lon, 0, 90, 0, 360)
    elif region == 'sh':
        # 90S - 0 N
        mask = _gen_latlon_grid_mask(lat, lon, -90, 0, 0, 360)
    elif region == 'europe':
        # 40 - 80 N, 20W - 40E
        mask = _gen_latlon_grid_mask(lat, lon, 40, 80, 340, 40)
    elif region is not None:
        raise KeyError('Unrecognized region in scalar function'
                       ' generation: {}'.format(region))
    else:
        mask = None

    weights = _get_area_weights(cell_area, lat, mask=mask)

    return mask, weights


def _get_area_avg_scalar_func(cell_area, lat, lon, region=None):

    mask, weights = get_area_avg_mask_and_weights(lat, lon, region, cell_area)

    def area_avg_index(state_data):
        if mask is not None:
            state_data = state_data[mask]

        valid_region, valid_mask = _remove_nan_from_state(state_data)
        if valid_mask is not None:
            use_factor = weights[valid_mask]
        else:
            use_factor = weights

        return valid_region.T @ use_factor

    return area_avg_index


def get_zonal_avg_weights(zon_lat, lat):
    # Grabs closest strip of zonal gridpoints to average
    min_val = abs(lat - zon_lat).min()
    mask = abs(lat - zon_lat) == min_val
    npts = mask.sum()
    weights = np.zeros_like(mask, dtype=np.float)
    weights[mask] = 1 / npts

    return weights


def _get_zonal_avg_func(zon_lat, lat):

    zonal_factor = get_zonal_avg_weights(zon_lat, lat)

    def zonal_avg(state_data):

        series = state_data.T @ zonal_factor

        return series

    return zonal_avg


def get_single_gridpoint_factor(pt_lat, pt_lon, lat, lon):

    dist = haversine(pt_lon, pt_lat, lon, lat)
    idx = np.argmin(dist)
    pt_factor = np.zeros_like(lat)
    pt_factor[idx] = 1

    return pt_factor


def _get_single_gridpoint_func(pt_lat, pt_lon, lat, lon):

    pt_factor = get_single_gridpoint_factor(pt_lat, pt_lon, lat, lon)

    def single_point(state_data):

        series = state_data.T @ pt_factor

        return series

    return single_point


def _gen_pdo_index_factor(prior_cfg, varname):
    print('Generating PDO EOF for index calculation.')
    # Loads prior data and averages it.
    prior_var = PriorVariable.load(prior_cfg, varname, anomaly=True, nens=None)
    var_dobj = prior_var.forecast_var_to_pylim_dataobj()
    grids = var_dobj.get_coordinate_grids(['lat', 'lon'], compressed=True,
                                          flat=True)
    latgrid = grids['lat']
    longrid = grids['lon']

    # PDO region mask from the compressed grid
    mask = _gen_latlon_grid_mask(latgrid, longrid, 20, 70, 110, 250)

    # Valid mask from the full grid
    valid_data = var_dobj.valid_data

    # TODO: Change to removal of GMT regression signal
    var_dobj.detrend_data()
    var_dobj.area_weight_data(use_sqrt=True)
    data = var_dobj.data[:][:, mask]
    npac_eofs, npac_svals = ST.calc_eofs(data, 1)
    pdo_eof = npac_eofs[:, 0]
    # full_grid_pdo_eof = var_dobj.inflate_full_grid(data=compressed_pdo_eof)

    return pdo_eof, valid_data, mask


def _get_pdo_index_func(prior_cfg, varname):

    pdo_eof, valid_data, mask = _gen_pdo_index_factor(prior_cfg, varname)

    def pdo_index(state_data):
        # If valid_data set then there were NaN points, remove them
        if valid_data is not None:
            state_data = state_data[valid_data, :]

        # Take the N Pac. region
        state_npac = state_data[mask, :]
        return state_npac.T @ pdo_eof

    return pdo_index


def prepare_field_output(outputs, state, ntimes, nens, output_dir, recon_times):

    # create zarr output files
    compressor = Blosc(cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE)

    ens_get_func_by_var = {}
    zarr_files_by_var = {}
    for var_key in state.base_prior_keys:

        var_name, avg_interval = var_key
        sptl_shape = state.var_space_shp[var_name]

        var_filename = 'field_out_{}_{}.zarr'.format(var_name, avg_interval)
        var_filepath = join(output_dir, var_filename)
        out_shape = (ntimes, *sptl_shape)

        # TODO: Check if this is okay for chunking
        state_dtype = state.state.dtype
        chunk_shape = get_chunk_shape(out_shape, state_dtype, 5)

        store = zarr.DirectoryStore(var_filepath)
        root = zarr.group(store=store, overwrite=True)

        lat = state.var_coords[var_name]['lat'].reshape(sptl_shape)
        lon = state.var_coords[var_name]['lon'].reshape(sptl_shape)

        zarr.save_group(store, lat=lat, lon=lon, time=recon_times)

        prior_grp = root.create_group('prior', overwrite=True)
        for measure in outputs['prior']:
            prior_grp.create_dataset(measure, shape=out_shape,
                                     chunks=chunk_shape, compressor=compressor,
                                     dtype=state_dtype)

        posterior_grp = root.create_group('posterior', overwrite=True)
        for measure in outputs['posterior']:
            posterior_grp.create_dataset(measure, shape=out_shape,
                                         chunks=chunk_shape,
                                         compressor=compressor,
                                         dtype=state_dtype)

        fullfield_opt = outputs['field_ens_output']
        if fullfield_opt is not None:
            [out_shp,
             ens_get_func] = _get_ensout_shp_and_func(fullfield_opt,
                                                      sptl_shape,
                                                      nens,
                                                      ntimes)

            ens_chunk_shp = get_chunk_shape(out_shape, state_dtype, 5)
            root.create_dataset('field_ens_output', shape=out_shp,
                                chunks=ens_chunk_shp, compressor=compressor,
                                dtype=state_dtype)
            root.attrs['options'] = fullfield_opt

            ens_get_func_by_var[var_key] = ens_get_func

        zarr_files_by_var[var_key] = root

    return zarr_files_by_var, ens_get_func_by_var


def _get_ensout_shp_and_func(option, sptl_shape, nens, ntimes):

    if isinstance(option, Sequence) and not isinstance(option, str):
        stored_nens = len(option)

        def grab_ens_members(state_data):
            ens = state_data[:, option]
            ens = ens.T.reshape(stored_nens, *sptl_shape)
            return ens
    elif option == 'all':
        stored_nens = nens

        def grab_ens_members(state_data):
            ens = state_data.T.reshape(stored_nens, *sptl_shape)
            return ens
    elif isinstance(option, int):
        # first element + number of time step fits into total nens
        stored_nens = (nens // option)

        def grab_ens_members(state_data):
            step = option
            ens = state_data[:, ::step]
            ens = ens.T.reshape(stored_nens, *sptl_shape)
            return ens

    else:
        raise ValueError('Unrecognized option for determining fullfield '
                         'ensemble output size.')

    full_out_shp = (ntimes, stored_nens, *sptl_shape)

    return full_out_shp, grab_ens_members


def save_field_output(idx, field_type, state, zarr_var_files,
                      output_def=None, ens_out_funcs=None):

    for var_key in state.base_prior_keys:

        data = state.get_var_data(var_key)
        var_name, avg_interval = var_key
        sptl_shape = state.var_space_shp[var_name]
        file_out = zarr_var_files[var_key]

        if field_type == 'prior' or field_type == 'posterior':

            for measure in output_def:
                path = join(field_type, measure)
                out_zarr_node = file_out[path]

                output = field_output_reduction(measure, data, sptl_shape)

                out_zarr_node[idx] = output

        elif field_type == 'field_ens_output':

            ens_out_func = ens_out_funcs[var_key]
            if ens_out_func is None:
                raise ValueError('Ensemble reduction function is required '
                                 'when to this function when "field_ens_out" '
                                 'is specified...')

            ens_out = ens_out_func(data)
            ens_out_zarr_node = file_out[field_type]

            ens_out_zarr_node[idx] = ens_out


def field_output_reduction(measure, state_data, sptl_shp):

    if measure == 'ens_var':
        out = state_data.var(ddof=1, axis=1)
    elif measure == 'ens_mean':
        out = state_data.mean(axis=1)
    else:
        raise KeyError('Unrecognized ensemble field reduction key: '
                       '{}'.format(measure))

    out = out.reshape(sptl_shp)
    return out


def save_recon_proxy_information(proxy_manager, out_dir):
    # save proxy assim ids and info
    out_fname_assim = 'assimilated_proxies.pkl'
    pobjs_assim = proxy_manager.sites_assim_proxy_objs()
    assim_proxy_info = _gather_proxy_object_info(pobjs_assim, 'assimilated')
    out_path = join(out_dir, out_fname_assim)
    with open(out_path, 'wb') as f:
        pickle.dump(assim_proxy_info, f)

    # save proxy eval ids and info
    out_fname_omit = 'witheld_proxies.pkl'
    pobjs_omit = proxy_manager.sites_eval_proxy_objs()
    if pobjs_omit:
        omit_proxy_info = _gather_proxy_object_info(pobjs_omit, 'witheld')
        out_path = join(out_dir, out_fname_omit)
        with open(out_path, 'wb') as f:
            pickle.dump(omit_proxy_info, f)


ProxyInfo = namedtuple('ProxyInfo', ['type', 'key', 'lat', 'lon', 'status',
                                     'available_years', 'psm_type',
                                     'psm_R', 'ye_data_idx'])


def _gather_proxy_object_info(pobj_list, status):

    proxy_info = []
    for i, pobj in enumerate(pobj_list):
        info = ProxyInfo(pobj.type, pobj.id, pobj.lat, pobj.lon, status,
                          pobj.time, pobj.psm_obj.psm_key, pobj.psm_obj.R, i)
        proxy_info.append(info)

    return proxy_info


def create_Ye_output(out_path, nproxies, nens, nyears, recon_yr_range):
    nbytes_in_nens = nens * 8

    # number of nens chunks in 5mb
    n_units_in_mb = 5 / (nbytes_in_nens / 1024**2)

    base_chunk_size = np.sqrt(n_units_in_mb / 2)
    nyr_chunk = int(base_chunk_size)
    nproxy_chunk = 2 * nyr_chunk

    if nyr_chunk > nyears:
        nyr_chunk = nyears
    if nproxy_chunk > nproxies:
        nproxy_chunk = nproxies

    chunk_shape = (nproxy_chunk, nyr_chunk, nens)

    compressor = Blosc(cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE)

    ye_arr = zarr.open(out_path, mode='w',
                       shape=(nproxies, nyears, nens), chunks=chunk_shape,
                       compressor=compressor, dtype=np.float64)
    ye_arr.attrs['recon_time_range'] = recon_yr_range

    return ye_arr









