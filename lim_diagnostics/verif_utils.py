import os
import pandas as pd
import numpy as np
import dask.array as da
from multiprocessing import Pool
from itertools import product

import LMR_outputs as lmout

import lim_utils as lutils
import plot_tools as ptools
import misc_utils as mutils
import data_utils as dutils

import pylim.Stats as ST


def get_scalar_factors(scalar_measures, avg_interval, var_valid_data, prior_cfg,
                       latgrid, longrid, cell_area=None):

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

    return scalar_factors


def get_scalar_outputs(dobj, nelem_in_yr, var_fcast, verif_data_attr,
                       out_types, use_dask=False, ):
    
    if use_dask:
        truth_data = dobj.reset_data(verif_data_attr)
    else:
        truth_data = getattr(dobj, verif_data_attr)
        
    truth_1yr = truth_data[nelem_in_yr:]
    truth_init = truth_data[:-nelem_in_yr]

    curr_var_output = {}

    for out_type in out_types:

        fcast_factor, verif_factor = get_scalar_factor(dobj, out_type,
                                                       verif_data_attr)

        var_out = var_fcast @ fcast_factor
        truth_init_out = truth_init @ verif_factor
        truth_1yr_out = truth_1yr @ verif_factor

        # Standardize PDO Index relative to truth output
        if out_type == 'pdo':
            truth_1yr_out, std_dev = _standardize_series(truth_1yr_out)
            var_out, _ = _standardize_series(var_out, std_dev=std_dev)
            truth_init_out, _ = _standardize_series(truth_init_out,
                                                    std_dev=std_dev)
            
        if use_dask:
            t_truth_1yr_out = np.empty(truth_1yr_out.shape)
            t_truth_init_out = np.empty(truth_init_out.shape)

            dask_vars = [truth_1yr_out, truth_init_out]
            dask_outs = [t_truth_1yr_out, t_truth_init_out]

            if ST.is_dask_array(var_out):
                t_var_out = np.empty(var_out.shape)
                dask_vars.append(var_out)
                dask_outs.append(t_var_out)

            da.store(dask_vars, dask_outs)

            truth_1yr_out = t_truth_1yr_out
            truth_init_out = t_truth_init_out

            if ST.is_dask_array(var_out):
                var_out = t_var_out

        curr_var_output[out_type] = {'fcast': var_out,
                                     't0': truth_init_out,
                                     '1yr': truth_1yr_out}
    return curr_var_output


def calc_scalar_ce_r(fcast, reference):
    [r, r_conf95] = lutils.conf_bound95(fcast, reference, metric='r')
    [ce, ce_conf95] = lutils.conf_bound95(fcast, reference, metric='ce')

    return (r, r_conf95, ce, ce_conf95)


def ens_fcast_verification(ens_fcast, fcast_outputs, dobjs, state,
                           verif_spec, nelem_in_yr, experiment_name, avg_key,
                           var_name_map, out_name_map, fig_dir,
                           do_hist=True, do_reliability=True):

    ens_metr_by_var = {}
    ens_scalar_out = {}
    for var_key, dobj in dobjs.items():

        var_fcast = state.get_var_from_state(var_key, data=ens_fcast)
        verif_data_attr = verif_spec.get(var_key, 'detrended')
        out_types = fcast_outputs[var_key]

        curr_var_output = get_scalar_outputs(dobj, nelem_in_yr, var_fcast,
                                             verif_data_attr, out_types,
                                             use_dask=True)

        ens_scalar_out[var_key] = curr_var_output

        curr_var_ens_scalar = {}
        for out_type, scalar_output in curr_var_output.items():
            print('Ens. Scalar Verification: {}, {}'.format(var_key, out_type))

            fcast_ens = scalar_output['fcast']
            ref = scalar_output['1yr']
            title = ('Exp: {}, {}  '
                     'Field: {} Measure: {}'.format(experiment_name,
                                                    avg_key,
                                                    var_key,
                                                    out_type))
            fig_fname = 'rank_hist_{}_{}_{}_{}.png'.format(experiment_name,
                                                           avg_key,
                                                           var_key,
                                                           out_type)
            fig_fpath = os.path.join(fig_dir, fig_fname)

            ens_calib = calc_ens_calib_ratio(fcast_ens, ref)
            curr_var_ens_scalar[out_type] = {'calib': ens_calib}

            if do_hist:
                rank_data = ptools.plot_rank_histogram(fcast_ens, ref, title,
                                                       savefile=fig_fpath)
                curr_var_ens_scalar[out_type]['rank'] = rank_data

            if (out_type == 'enso' or out_type == 'pdo') and do_reliability:

                title_temp = ('Exp: {}, {}  Metr: {} Reliability '
                              '(index {})')
                savefile_temp = 'reliability_{}_{}_{}_{}_{}.png'

                pct_map = {'upper': '>0.5',
                           'lower': '<-0.5'}

                reliab_dict = {}
                for event_type in ['upper', 'lower']:
                    obs_freq, bin_fcast_avg, errors = \
                        calc_reliability_with_bounds(fcast_ens, ref,
                                                     event_type=event_type)

                    title = title_temp.format(experiment_name, avg_key,
                                              out_type, pct_map[event_type])
                    fname = savefile_temp.format(experiment_name, avg_key,
                                                 var_key, out_type,
                                                 event_type)
                    savefile = os.path.join(fig_dir, fname)

                    ptools.plot_reliability(obs_freq, bin_fcast_avg, errors,
                                            title, savefile=savefile)
                    reliab_dict[event_type] = (obs_freq, bin_fcast_avg,
                                               errors)

                curr_var_ens_scalar[out_type]['reliability'] = reliab_dict

        # Run field ens_calibration measure

        # eofs = dobj._eofs
        #
        # wgts = dobj.cell_area / dobj.cell_area.sum()
        #
        # ref_data_attr = verif_spec.get(var_key, 'detrended')
        # ref_data = getattr(dobj, ref_data_attr)
        #
        # ntimes = var_fcast.shape[1]
        #
        # args = _stl_ens_calib_arg_generator(ntimes, var_fcast,
        #                                     ref_data, ref_data_attr, eofs)
        #
        # with Pool(processes=8) as proc_pool:
        #     calib_res = proc_pool.map(_stl_ens_calib_func, args)
        #
        # ens_calib_avg = np.array(calib_res).mean(axis=0)
        # spatl_avg_ens_calib = ens_calib_avg @ wgts
        # curr_var_ens_calib['spatial_avg'] = spatl_avg_ens_calib
        #
        #
        # if do_spatial_plot:
        #     _plot_spatial(ens_calib_avg, 'ens_calib', experiment_name,
        #                   avg_key, var_key, dobj, fig_dir)

        ens_metr_by_var[var_key] = curr_var_ens_scalar

    return ens_metr_by_var, ens_scalar_out


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

    sq_err = (fcast.mean(axis=0) - ref)**2
    mse = sq_err.mean(axis=0)

    mean_ens_var = fcast.var(ddof=1, axis=0).mean(axis=0)

    ens_calib_ratio = mse / mean_ens_var

    return ens_calib_ratio


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


def _standardize_series(data, std_dev=None):

    data = data - data.mean(axis=-1, keepdims=True)

    if std_dev is None:
        std_dev = np.std(data, axis=-1, ddof=1)

    data = data / std_dev

    return data, std_dev

