import os
import numpy as np
from multiprocessing import Pool
from itertools import product
from collections import defaultdict

import LMR_outputs as lmout
from LMR_utils import crps

import lim_diagnostics.lim_utils as lutils
import lim_diagnostics.plot_tools as ptools
import lim_diagnostics.misc_utils as mutils


def get_scalar_factors(scalar_measures, avg_interval, var_valid_data, prior_cfg,
                       latgrid, longrid, cell_area=None, psm_req_var_keys=None):

    # divvy out var_keys loaded by psm by var_name
    if psm_req_var_keys is not None:
        psm_varname_varkey_map = defaultdict(list)
        for psm_var_key in psm_req_var_keys:
            psm_var_name, psm_avg_int = psm_var_key
            psm_varname_varkey_map[psm_var_name].append(psm_var_key)
    else:
        psm_varname_varkey_map = None

    scalar_factors = {}
    for varname, measure_list in scalar_measures.items():
        for measure in measure_list:
            var_key = (varname, avg_interval)

            if var_key in var_valid_data:
                valid_data = var_valid_data[var_key]
                lat = latgrid[valid_data]
                lon = longrid[valid_data]
            else:
                lat = latgrid
                lon = longrid

            factor_key = (varname, avg_interval, measure)
            scalar_factors[factor_key] = \
                lmout.get_scalar_factor(measure, varname, prior_cfg, lat, lon,
                                        cell_area=cell_area)

            # scalar factor is same for equivalent variable names
            if psm_varname_varkey_map and varname in psm_varname_varkey_map:
                for psm_var_key in psm_varname_varkey_map[varname]:
                    psm_factor_key = (*psm_var_key, measure)
                    scalar_factors[psm_factor_key] = scalar_factors[factor_key]


    return scalar_factors


def add_eofs_to_scalar_factors(base_factors, lim_fcaster, base_state_keys):
    full_scalar_factors = {}
    full_field_factors = {}
    for measure_key, factor in base_factors.items():

        # last value in tuple is measure, var_key is (varname, avg_interval)
        var_key = measure_key[:-1]

        full_field = mutils.get_field_factor(var_key, lim_fcaster.var_eofs,
                                             lim_fcaster.var_eof_std_factor,
                                             lim_fcaster.var_span,
                                             lim_fcaster.calib_eofs)

        if var_key not in full_field_factors:
            full_field_factors[var_key] = full_field
        full_scalar = full_field @ factor
        full_scalar_factors[measure_key] = full_scalar

    for var_key in base_state_keys:
        if var_key not in full_field_factors.keys():
            full_field = mutils.get_field_factor(var_key, lim_fcaster.var_eofs,
                                                 lim_fcaster.var_eof_std_factor,
                                                 lim_fcaster.var_span,
                                                 lim_fcaster.calib_eofs)
            full_field_factors[var_key] = full_field

    return full_scalar_factors, full_field_factors


def calc_scalar_ce_r(fcast, reference, is_ar1=False):
    [r, r_conf95] = lutils.conf_bound95(fcast, reference, metric='r')
    [ce, ce_conf95] = lutils.conf_bound95(fcast, reference, metric='ce')

    r_key = 'r_conf95'
    ce_key = 'ce_conf95'

    if is_ar1:
        r_key = 'auto1_' + r_key
        ce_key = 'auto1_' + ce_key

    return (r, ce), {r_key: r_conf95, ce_key: ce_conf95}


def plot_spatial_verif(field, valid_data, sptl_shp, lat, lon,
                       metric, experiment_name,
                       avg_key, var_key, fig_dir=None):

    if valid_data is not None:
        reinfl_field = np.empty_like(valid_data, dtype=field.dtype)
        reinfl_field *= np.nan
        reinfl_field[valid_data] = field
        field = reinfl_field

    field = field.reshape(sptl_shp)
    lat = lat.reshape(sptl_shp)
    lon = lon.reshape(sptl_shp)

    fname_template = 'spatial_{}_{}_{}.png'
    title_template = 'Exp: {}, {} Field: {} Metric: {}'

    save_metric = metric.lower().replace(' ', '-')
    fname = fname_template.format(var_key, avg_key, save_metric)
    title = title_template.format(experiment_name, avg_key, var_key, metric)
    if fig_dir is not None:
        fpath = os.path.join(fig_dir, fname)
    else:
        fpath = None

    if 'ce' in metric.lower():
        extend = 'min'
    else:
        extend = 'neither'

    if 'ens_calib' in metric.lower():
        cmap = 'inferno'
        bnds = [0, 10]
        midpoint = 1
    else:
        midpoint = None
        bnds = [-1, 1]
        cmap = 'RdBu_r'

    ptools.plot_single_spatial_field(lat, lon, field, title, data_bnds=bnds,
                                     savefile=fpath, gridlines=False,
                                     extend=extend, cmap=cmap,
                                     midpoint=midpoint)


def calc_ens_calib_ratio(fcast, ref):

    ens_mean_sq_err = (fcast.mean(axis=0) - ref)**2
    mse_avg = ens_mean_sq_err.mean(axis=0)

    mean_ens_var = fcast.var(ddof=1, axis=0).mean(axis=0)

    ens_calib_ratio = mse_avg / mean_ens_var

    return ens_calib_ratio


def calc_crps(fcast, ref):

    # Proper scoring for probabilistic forecast
    # Based on Tipton et al. 2016 & ClimateCorp proper scoring package

    return crps(fcast, ref)


#  This is for spatial ensemble calibration
def _stl_ens_calib_func(args):

    i, var_fcast, ref_data, ref_data_attr, eofs = args
    curr_fcast_t = var_fcast[:, i:i+1] @ eofs.T
    curr_ref_t = ref_data

    if ref_data_attr == 'eof_proj':
        curr_ref_t = curr_ref_t @ eofs.T

    curr_ens_calib = calc_ens_calib_ratio(curr_fcast_t, curr_ref_t)

    return curr_ens_calib


def _stl_ens_calib_arg_generator(ntimes, var_fcast,
                                 ref_data, ref_data_attr, eofs):

    for i in range(ntimes):
        yield (i, var_fcast, ref_data[i+1:i+2], ref_data_attr, eofs)


def calc_ens_reliability(fcast_probs, occurences):

    # Get the number of counts for each bin
    bin_counts, bin_edges = np.histogram(fcast_probs, bins=10, range=(0, 1))

    # Map each forecast to a bin
    fcast_bin_map = np.digitize(fcast_probs, bin_edges[:-1])

    bin_num_hits = []
    bin_fcast_mean = []
    for i, bin_count in enumerate(bin_counts, start=1):
        idxs, = np.where(fcast_bin_map == i)
        hits = occurences[idxs].sum()
        mean = fcast_probs[idxs].mean()
        bin_num_hits.append(hits)
        bin_fcast_mean.append(mean)

    bin_num_hits = np.array(bin_num_hits)
    bin_counts = np.array(bin_counts)
    bin_fcast_mean = np.array(bin_fcast_mean)

    obs_rel_freq = bin_num_hits / bin_counts

    return obs_rel_freq, bin_fcast_mean


def _get_event_func(ref, event_type):

    if event_type == 'upper':
        def event(x):
            return x >= 0.5
    elif event_type == 'lower':
        def event(x):
            return x <= -0.5
    else:
        raise ValueError('Unrecognized event designation key: '
                         '{}'.format(event_type))

    return event


def _mc_obs_rel_freq(args):

    i, fcast_probs = args

    np.random.seed(i)

    x_hat = np.random.choice(fcast_probs, size=len(fcast_probs),
                             replace=True)
    y_hat = np.random.random(size=len(x_hat))
    y_hat = np.less(y_hat, x_hat)
    obs_rel_freq, _ = calc_ens_reliability(x_hat, y_hat)

    return obs_rel_freq


def calc_reliability_with_bounds(fcast, ref, event_type='upper'):

    # Resampling operation to bound reliable forecast region
    # Brocker and Smith 2007

    event = _get_event_func(ref, event_type)

    nens = fcast.shape[0]
    fcast_probs = event(fcast).sum(axis=0) / nens
    occurrences = event(ref)

    num_mc_iter = 1000
    seeds = np.random.choice(num_mc_iter*10, size=num_mc_iter, replace=False)
    args = product(seeds, (fcast_probs, ))

    with Pool(processes=8) as reliable_pool:
        res_obs_freq = reliable_pool.map(_mc_obs_rel_freq, args)

    res_obs_freq = np.array(res_obs_freq)
    upper_bnd = np.percentile(res_obs_freq, 97.5, axis=0)
    lower_bnd = np.percentile(res_obs_freq, 2.5, axis=0)

    [obs_rel_freq,
     bin_fcast_mean] = calc_ens_reliability(fcast_probs, occurrences)

    mean_rof = res_obs_freq.mean(axis=0)
    upper_bnd = upper_bnd - mean_rof
    lower_bnd = abs(lower_bnd - mean_rof)

    errors = np.vstack((lower_bnd, upper_bnd))

    return obs_rel_freq, bin_fcast_mean, errors


def calc_soi_from_gridpoints(tahiti, darwin):

    """
    Calculate SOI from gridpoint pressure timeseries from tahiti and darwin

    Parameters
    ----------
    tahiti - Tahiti pressure time series
    darwin - Darwin pressure time series

    Returns
    -------

    """
    tahiti_anom = tahiti - tahiti.mean(axis=-1, keepdims=True)
    tahiti_std, tahiti_std_factor = _standardize_series(tahiti_anom,
                                                        preserve_ens_var=True)

    darwin_anom = darwin - darwin.mean(axis=-1, keepdims=True)
    darwin_std, darwin_std_factor = _standardize_series(darwin_anom,
                                                        preserve_ens_var=True)

    soi = tahiti_std - darwin_std
    soi, soi_std_factor = _standardize_series(soi, preserve_ens_var=True)

    return soi


def _standardize_series(data, std_dev=None, preserve_ens_var=False):

    data = data - data.mean(axis=-1, keepdims=True)

    if std_dev is None:
        if data.ndim > 1 and preserve_ens_var:
            ens_avg = data.reshape(-1, data.shape[-1]).mean(axis=0)
            std_dev = ens_avg.std(ddof=1)
        else:
            std_dev = np.std(data, axis=-1, ddof=1, keepdims=True)

    data = data / std_dev

    return data, std_dev


def handle_soi_factors(scalar_factors, base_factors, measure_out,
                       state_obj, fcast_1yr):
    """
    Take scalar factors and look for required data.  If it's there calculate it
    remove the factors used and add in soi index to the measure_out dictionary.

    Parameters
    ----------
    scalar_factors - dict to look for tahiti and darwin scalar factors
    base_factors - dict contining full_space -> scalar matrix factors
    measure_out - scalar output dictionary to add soi result to
    state_obj - reference state in full space
    fcast_1yr - 1-year forecast in LIM space

    Returns
    -------

    """
    # Calculate soi if it's there
    for measure_key in scalar_factors.keys():
        if 'tahiti' == measure_key[-1]:
            tahiti_key = measure_key
            tahiti_exist = True
            break
    else:
        tahiti_key = None
        tahiti_exist = False

    for measure_key in scalar_factors.keys():

        if 'darwin' == measure_key[-1]:
            darwin_key = measure_key
            darwin_exist = True
            break
    else:
        darwin_key = None
        darwin_exist = False

    if tahiti_exist and darwin_exist:
        tahiti_factor = scalar_factors.pop(tahiti_key)
        darwin_factor = scalar_factors.pop(darwin_key)

        # tahiti_key is (var_name, avg_interval, 'tahiti')
        ref_psl = state_obj.get_var_data(tahiti_key[:-1]).T

        # convert reference target, delete from factors
        tahiti_psl_ref = ref_psl @ base_factors[tahiti_key]
        del base_factors[tahiti_key]
        darwin_psl_ref = ref_psl @ base_factors[darwin_key]
        del base_factors[darwin_key]

        tahiti_psl_fcast = fcast_1yr @ tahiti_factor
        darwin_psl_fcast = fcast_1yr @ darwin_factor

        soi_fcast = calc_soi_from_gridpoints(tahiti_psl_fcast, darwin_psl_fcast)
        soi_ref = calc_soi_from_gridpoints(tahiti_psl_ref, darwin_psl_ref)

        soi_key = list(tahiti_key)
        soi_key[-1] = 'soi'
        soi_key = tuple(soi_key)
        measure_out[soi_key] = (soi_ref, soi_fcast)

    return scalar_factors, measure_out

