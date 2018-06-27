from LMR_gridded import PriorVariable
from LMR_utils2 import var_to_hdf5_carray, empty_hdf5_carray
from collections import Sequence
import numpy as np
import tables as tb

def prepare_scalar_calculations(scalar_outdef, state, prior_cfg):

    func_by_var = {}
    for varname, scalar_measures in scalar_outdef.items:
        func_by_measure = {}
        for measure in scalar_measures:
            curr_func = _gen_scalar_func(measure, varname,
                                         state, prior_cfg)
            func_by_measure[measure] = curr_func

        func_by_var[varname] = func_by_measure

    return func_by_var


def prepare_output_containers(outputs, state, ntimes, nens, h5f_path):

    # Scalar containers
    scalar_outputs = outputs['scalar_ens']
    scalar_containers_by_var = {}
    for varkey, output_measures in scalar_outputs.items():
        scalar_containers_by_meas = {}
        for measure in output_measures:
            container = np.zeros((ntimes, nens))
            scalar_containers_by_meas[measure] = container

        scalar_containers_by_var[varkey] = scalar_containers_by_meas

    # create h5 file for field outputs

    filters = tb.Filters(complevel=4, complib='blosc')
    h5f = tb.open_file(h5f_path, mode='w', filters=filters)
    dtype = state.state.dtype
    atom = tb.Atom.from_dtype(dtype)
    for varkey, sptl_shape in state.var_space_shp.items():
        var_grp = tb.Group(h5f.root, name=varkey)
        out_shape = (ntimes, *sptl_shape)

        prior_grp = tb.Group(var_grp, 'prior')
        for measure in outputs['prior']:
            empty_hdf5_carray(h5f, prior_grp, measure, atom, out_shape)

        posterior_grp = tb.Group(var_grp, 'posterior')
        for measure in outputs['posterior']:
            empty_hdf5_carray(h5f, posterior_grp, measure, atom, out_shape)

        fullfield_opt = outputs['fullfield_ens_output']
        if fullfield_opt is not None:
            [out_shp,
             ens_get_func] = _get_ensout_shp_and_func(fullfield_opt,
                                                      sptl_shape,
                                                      nens,
                                                      ntimes)
            node = empty_hdf5_carray(h5f, var_grp, 'fullfield_ens', atom,
                                     out_shp)
            node._v_attrs.opt = fullfield_opt

    return scalar_containers_by_var, h5f, ens_get_func


def _get_ensout_shp_and_func(option, sptl_shape, nens, ntimes):

    if isinstance(option, Sequence) and not isinstance(option, str):
        stored_nens = len(option)

        def grab_ens_members(state_data):
            ens = state_data[:, stored_nens]
            ens = ens.reshape(stored_nens, *sptl_shape)
            return ens
    elif option == 'all':
        stored_nens = nens

        def grab_ens_members(state_data):
            ens = state_data.reshape(stored_nens, *sptl_shape)
            return ens
    elif isinstance(option, int):
        stored_nens =  option

        def grab_ens_members(state_data):
            step = nens // stored_nens
            ens = state_data[:, ::step]
            ens = ens.reshape(stored_nens, *sptl_shape)
            return ens

    else:
        raise ValueError('Unrecognized option for determining fullfield '
                         'ensemble output size.')

    full_out_shp = (ntimes, stored_nens, *sptl_shape)

    return full_out_shp, grab_ens_members



def _gen_scalar_func(measure, varname, state, prior_cfg):

    if measure == 'glob_mean':
        cell_area = state.var_cell_area[varname]
        func = _gen_global_mean_func(cell_area)
    elif measure == 'enso34':
        coord_data = state.var_coords[varname]
        lat = coord_data['lat']
        lon = coord_data['lon']
        func = _gen_enso_index(lat, lon, region='34')
    elif measure == 'pdo':
        func = _gen_pdo_index(prior_cfg, varname)
    else:
        raise KeyError('Unrecognized scalar measure {}'.format(measure))

    return func


def _gen_global_mean_func(cell_area):

    weights = cell_area / cell_area.sum()

    def global_average(state_data):
        return state_data.T @ weights

    return global_average


def _gen_enso_index(lat, lon, region='34'):

    if region == '34':
        mask = ((lon >= 240) & (lon <= 290) &
                (lat >= -5) & (lat <= 5))
    else:
        raise KeyError('Unrecognized enso region in scalar function'
                       ' generation: {}'.format(region))

    num_pts_enso = mask.sum()
    enso_avg_factor = np.zeros_like(lat)
    enso_avg_factor[mask] = 1/num_pts_enso

    def enso_index(state_data):
        return state_data.T @ enso_avg_factor

    return enso_index


def _gen_pdo_index(prior_cfg, varname):

    print('Generating PDO EOF for index calculation.')
    var_obj = PriorVariable.load(prior_cfg, varname, anomaly=True, nens=None)
    var_dobj = var_obj.forecast_var_to_pylim_dataobj()
    grids = var_dobj.get_coordinate_grids(['lat', 'lon'], compressed=True,
                                          flat=True)
    latgrid = grids['lat']
    longrid = grids['lon']

    mask = ((latgrid >= 20) & (latgrid <= 70) &
            (longrid >= 110) & (longrid <= 250))

    # TODO: Is detrending generally right?
    var_dobj.detrend_data()
    data = var_obj.data[:][:, mask]
    npac_eofs, npac_svals = ST.calc_eofs(data, 1)
    compressed_pdo_eof = np.zeros_like(latgrid)
    compressed_pdo_eof[mask] = npac_eofs[:, 0]
    full_grid_pdo_eof = var_dobj.inflate_full_grid(data=compressed_pdo_eof)

    def pdo_index(state_data):
        return state_data.T @ full_grid_pdo_eof

    return pdo_index
