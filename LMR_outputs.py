from LMR_gridded import PriorVariable
from LMR_utils import var_to_hdf5_carray, empty_hdf5_carray
from collections import Sequence, namedtuple
from numcodecs import Blosc
import numpy as np
import tables as tb
import pickle
import zarr

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

    if measure == 'glob_mean':
        cell_area = state.var_cell_area[varname]
        func = _gen_global_mean_func(cell_area, lat)
    elif measure == 'enso34':
        func = _gen_enso_index(lat, lon, region='34')
    elif measure == 'pdo':
        func = _gen_pdo_index(prior_cfg, varname)
    else:
        raise KeyError('Unrecognized scalar measure {}'.format(measure))

    return func


def _remove_nan_from_state(state_data):

    nan_locs = np.isnan(state_data)
    if np.any(nan_locs):
        # Sum True values across ensemble members, if any NaNs, we mask it
        valid_data = nan_locs.sum(axis=1) == 0
        state_data = state_data[valid_data]
    else:
        valid_data = None

    return state_data, valid_data


def _gen_global_mean_func(cell_area, lat):

    if cell_area is not None:
        weights = cell_area / cell_area.sum()
    else:
        # TODO: This only works for regular grids, but that's what we regrid to
        # otherwise there'll be cell area loaded for fields.
        weights = np.cos(np.deg2rad(lat))
        weights = weights / weights.sum()

    def global_average(state_data):
        valid_state, valid_locs = _remove_nan_from_state(state_data)

        if valid_locs is not None:
            use_weights = weights[valid_locs]
        else:
            use_weights = weights

        return valid_state.T @ use_weights

    return global_average


def _gen_enso_index(lat, lon, region='34'):

    if region == '34':
        mask = ((lon >= 190) & (lon <= 240) &
                (lat >= -5) & (lat <= 5))
    else:
        raise KeyError('Unrecognized enso region in scalar function'
                       ' generation: {}'.format(region))

    num_pts_enso = mask.sum()
    enso_avg_factor = np.zeros(num_pts_enso)
    enso_avg_factor[:] = 1/num_pts_enso

    def enso_index(state_data):
        state_enso_region = state_data[mask]

        valid_enso_region, valid_data = _remove_nan_from_state(state_enso_region)
        if valid_data is not None:
            use_enso_factor = enso_avg_factor[valid_data]
        else:
            use_enso_factor = enso_avg_factor

        return valid_enso_region.T @ use_enso_factor

    return enso_index


def _gen_pdo_index(prior_cfg, varname):

    print('Generating PDO EOF for index calculation.')
    # Loads prior data and averages it.
    prior_var = PriorVariable.load(prior_cfg, varname, anomaly=True, nens=None)
    var_dobj = prior_var.forecast_var_to_pylim_dataobj()
    grids = var_dobj.get_coordinate_grids(['lat', 'lon'], compressed=True,
                                          flat=True)
    latgrid = grids['lat']
    longrid = grids['lon']

    mask = ((latgrid >= 20) & (latgrid <= 70) &
            (longrid >= 110) & (longrid <= 250))

    # Valid mask from the full grid
    valid_data = var_dobj.valid_data

    # TODO: Is detrending generally right?
    var_dobj.detrend_data()
    var_dobj.area_weight_data(use_sqrt=True)
    data = var_dobj.data[:][:, mask]
    npac_eofs, npac_svals = ST.calc_eofs(data, 1)
    pdo_eof = npac_eofs[:, 0]
    # full_grid_pdo_eof = var_dobj.inflate_full_grid(data=compressed_pdo_eof)

    def pdo_index(state_data):
        # If valid_data set then there were NaN points, remove them
        if valid_data is not None:
            state_data = state_data[valid_data, :]

        # Take the N Pac. region
        state_npac = state_data[mask, :]
        return state_npac.T @ pdo_eof

    return pdo_index


def prepare_field_output(outputs, state, ntimes, nens, output_dir):

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
        element_nbytes = state_dtype.itemsize
        nelem_in_map = np.prod(np.array(sptl_shape))
        ntimes_in_5mb = int(5 / (element_nbytes * nelem_in_map / 1024**2))
        chunk_shape = (ntimes_in_5mb, *[None]*len(sptl_shape))

        store = zarr.DirectoryStore(var_filepath)
        root = zarr.group(store=store, overwrite=True)

        lat = state.var_coords[var_name]['lat'].reshape(sptl_shape)
        lon = state.var_coords[var_name]['lon'].reshape(sptl_shape)

        zarr.save_group(store, lat=lat, lon=lon)

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

            nens_out = out_shp[1]
            nens_in_ntimes = ntimes_in_5mb / nens_out
            if nens_in_ntimes > 1:

                ens_chunk_shp = (np.round(nens_in_ntimes),
                                 None,
                                 *[None]*len(sptl_shape))
            else:
                ens_chunk_shp = (1, int(nens_in_ntimes * nens_out),
                                 *[None]*len(sptl_shape))
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









