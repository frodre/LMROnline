import LMR_proxy
import LMR_gridded
import LMR_forecaster
import LMR_config
import LMR_outputs

import logging
import sys
import os
import pandas as pd
import numpy as np
import warnings

import lim_diagnostics.plot_tools as ptools
import lim_diagnostics.lim_utils as lutils
import lim_diagnostics.verif_utils as vutils
import lim_diagnostics.misc_utils as mutils

import pylim.Stats as ST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# defaults to config.core.nexp in working directory
fig_dir = '/home/disk/p/wperkins/ipynb/lim_diagnostics'

# Fig Output
plot_neofs = 10
plot_eofs = False
plot_state_eofs = False

plot_lim_modes = False
plot_num_lim_modes = 20

plot_lim_noise_eofs = False
plot_num_noise_modes = 10

fcast_against = 'mpi-esm-p_last_millenium'
is_diff_model = True
fcast_start_yr = 851

# Only use fields specified in prior state dimension. False emulates
# reconstruction state, including PSM required averages of fields
base_only = True

# Perfect Forecast Experiments
detrend_fcast_ref_data = True

do_perfect_fcast = True
do_scalar_verif = True
plot_scalar_verif = True
do_spatial_verif = True
plot_spatial_verif = True

# Ensemble noise integration forecast experiments
do_ens_fcast = False
nens = 100
do_hist = False
do_reliability = False

# Long integration forecast experiments
do_long_integration = False
integration_len_yr = 1000
integration_iters = 500

# ========================
# END USER PARAMS
# ========================


var_long_names = {'tas_sfc_Amon': '2m Air T',
                  'tos_sfc_Omon': 'SST',
                  'ohc_0-700m_Omon': 'OHC 0-700m',
                  'psl_sfc_Amon': 'SLP',
                  'zg_500hPa_Amon': '500 hPa Hght',
                  'pr_sfc_Amon': 'Sfc Precip'}

var_units = {'tas_sfc_Amon': 'K',
             'tos_sfc_Omon': 'K',
             'ohc_0-700m_Omon': 'W/m$^2$',
             'psl_sfc_Amon': 'Pa',
             'zg_500hPa_Amon': 'm',
             'pr_sfc_Amon': 'mm'}


def get_scalar_factors(latgrid, longrid, cfg_obj, lim_fcast_obj, base_keys):
    base_scalar_factors = \
        vutils.get_scalar_factors(cfg_obj.prior.outputs['scalar_ens'],
                                  cfg_obj.prior.avg_interval,
                                  lim_fcast_obj.valid_data_mask,
                                  cfg_obj.prior,
                                  latgrid, longrid)

    # include EOFs to go straight from LIM space -> scalar measure
    scalar_factors, field_factors = \
        vutils.add_eofs_to_scalar_factors(base_scalar_factors, lim_fcast_obj,
                                          base_keys)

    # scalar_factors: lim_space -> scalar
    # field_factors: lim_space -> full_space
    # base_scalar_factors: full_space -> scalar
    return scalar_factors, field_factors, base_scalar_factors

# add eofs and standardization into factor matrices


def perfect_fcast_verification(state_obj, cfg_obj, lim_fcast_obj,
                               state_lim_space, times, base_keys,
                               fcast_against_src,
                               fig_out_dir='.'):

    """

    Parameters
    ----------
    state_obj
        initialized state object used to grab lat lons
    cfg_obj
        initialized config object
    lim_fcast_obj
        initialized LIMForecaster object
    state_lim_space
        state reduced to LIM space by the fcast object
    times
        array of years corresponding to 1-year forecast times
    base_keys
        list of state keys to output for spatial field verification
    fcast_against_src
        Name of prior source used to forecast against
    fig_out_dir
        Directory location to store perfect forecast verification figures

    Returns
    -------

    """
    perf_figdir = os.path.join(fig_out_dir, 'perfect_fcast',
                               fcast_against_src)
    os.makedirs(perf_figdir, exist_ok=True)

    fcast_1yr = lim_fcast_obj.lim.forecast(state_lim_space[:-1], [1])
    fcast_1yr = np.squeeze(fcast_1yr)

    # load scalar factors for forecasting experiments
    grid_coords = next(iter(state_obj.var_coords.values()))
    latgrid = grid_coords['lat']
    longrid = grid_coords['lon']

    [scalar_factors,
     field_factors,
     base_scalar_factors] = get_scalar_factors(latgrid, longrid,
                                               cfg_obj,
                                               lim_fcast_obj,
                                               base_keys)

    output_dfs = []
    if do_scalar_verif:
        output_dfs += scalar_perf_fcast_verification(scalar_factors,
                                                     base_scalar_factors,
                                                     times,
                                                     fcast_1yr,
                                                     state_obj,
                                                     lim_fcast_obj.valid_data_mask,
                                                     fcast_against_src,
                                                     perf_figdir)
    if do_spatial_verif:
        output_dfs += spatial_perf_fcast_verification(base_keys, field_factors,
                                                      times,
                                                      fcast_1yr, state_obj,
                                                      latgrid, longrid,
                                                      lim_fcast_obj.valid_data_mask,
                                                      lim_fcast_obj.var_std_factor,
                                                      fcast_against_src,
                                                      perf_figdir)

    if output_dfs:
        perf_fcast_dfs = pd.concat(output_dfs)
        df_savefile = os.path.join(perf_figdir, 'perf_fcast_verif_out_df.h5')
        perf_fcast_dfs.to_hdf(df_savefile, fcast_against_src)


def scalar_perf_fcast_verification(scalar_factors, base_factors, times,
                                   fcast_1yr, state_obj, valid_data_masks,
                                   fcast_against_src, perf_figdir):
    """

    Parameters
    ----------
    scalar_factors
        dict of factors by (var, avg_interval, measure) to
        matrix multiply the lim space output by to get the scalar measure
    base_factors
        dict of factors by (var, avg_interval, measure) to matrix multiply
        spatial fields by to get the scalar measure
    times
        array of years corresponding to 1-year forecast times
    fcast_1yr
        lim forecast in lim space
    state_obj:
        state in full space, used to calculate the target scalar info
    valid_data_masks
        dict of masks by (var, avg_interval) to be applied to fields to omit
        NaN information
    fcast_against_src
        Name of prior source used to forecast against
    perf_figdir
        figure output path

    Returns
    -------

    """
    perf_fcast_dfs = []

    measure_out = {}
    scalar_factors, measure_out = vutils.handle_soi_factors(scalar_factors,
                                                            base_factors,
                                                            measure_out,
                                                            state_obj,
                                                            fcast_1yr)

    for measure_key, factor in scalar_factors.items():
        var_key = measure_key[:-1]
        valid_data = valid_data_masks.get(var_key, None)
        ref_dat = mutils.get_field_from_state(state_obj, var_key,
                                              valid_data=valid_data)
        ref_measure = ref_dat @ base_factors[measure_key]
        fcast_measure = fcast_1yr @ factor

        measure_out[measure_key] = (ref_measure, fcast_measure)

    for measure_key, (ref, fcast) in measure_out.items():
        var_name, avg_interval, measure = measure_key

        ar1_fcast = mutils.red_noise_forecast_ar1(ref)

        target_scalar = ref[1:]

        r_ce_args, r_ce_kwargs = vutils.calc_scalar_ce_r(fcast, target_scalar)
        ar1_r_ce_args, ar1_r_ce_kwargs = vutils.calc_scalar_ce_r(ar1_fcast,
                                                                 target_scalar,
                                                                 is_ar1=True)

        verif_df = mutils.ce_r_results_to_dataframe(var_name,
                                                    avg_interval,
                                                    measure,
                                                    *r_ce_args,
                                                    *ar1_r_ce_args,
                                                    **r_ce_kwargs,
                                                    **ar1_r_ce_kwargs)

        perf_fcast_dfs.append(verif_df)

        if plot_scalar_verif:
            title = '{}, {}'.format(var_long_names[var_name],
                                    measure)
            plt_savefile = 'scalar_{}_{}_{}.png'.format(*measure_key)
            plt_savepath = os.path.join(perf_figdir, plt_savefile)
            units = var_units[var_name]
            ptools.plot_scalar_verification(times, fcast,
                                            target_scalar,
                                            *r_ce_args, *ar1_r_ce_args,
                                            title, fcast_against_src, units,
                                            **r_ce_kwargs, **ar1_r_ce_kwargs,
                                            savefile=plt_savepath)

    return perf_fcast_dfs


def spatial_perf_fcast_verification(base_keys, field_factors, times, fcast_1yr,
                                    state_obj, latgrid, longrid,
                                    valid_data_masks, var_std_factors,
                                    fcast_against_src, perf_figdir):

    """

    Parameters
    ----------
    base_keys
        State keys for the basic output variables specifified in the
        configuration. Keys are of the form (var_name, avg_interval)
    field_factors
        dict of factors by (var, avg_interval) to
        matrix multiply the lim space output by to get the full field
    times
        array of years corresponding to 1-year forecast times
    fcast_1yr
        lim forecast in lim space
    state_obj
        state used as initial conditions for the forecast
    latgrid
        flattened grid of latitude coordinates
    longrid
        flattened grid of longitude coordinates
    valid_data_masks
        dict of masks by (var, avg_interval) to be applied to fields to omit
        NaN information
    var_std_factors
        dict of standardization factors for the fields by variable key (
        var_name, avg_interval)
    fcast_against_src
        Name of prior source used to forecast against
    perf_figdir
        figure output path

    Returns
    -------

    """
    perf_fcast_dfs = []
    for var_key in base_keys:
        var_name, avg_interval = var_key
        field_factor = field_factors[var_key]

        valid_data = valid_data_masks.get(var_key, None)
        var_std = var_std_factors.get(var_key, None)
        init_field = mutils.get_field_from_state(state_obj, var_key,
                                                 valid_data=valid_data,
                                                 var_std_factor=var_std)

        ar1_field_fcast = mutils.red_noise_forecast_ar1(init_field)
        target_field = init_field[1:]

        fcast_1yr_field = fcast_1yr @ field_factor

        lac = ST.calc_lac(fcast_1yr_field, target_field)

        # check for invalid LAC as a check of valid inputs
        invalid_data = np.isnan(lac)
        nonzero_data = np.logical_not(invalid_data)

        ce = ST.calc_ce(fcast_1yr_field, target_field)
        anom_corr = ST.calc_lac(fcast_1yr_field.T, target_field.T)

        ar1_lac = ST.calc_lac(ar1_field_fcast, target_field)
        ar1_ce = ST.calc_ce(ar1_field_fcast, target_field)
        ar1_anom_corr = ST.calc_lac(ar1_field_fcast.T[nonzero_data],
                                    target_field.T[nonzero_data])

        if var_key in valid_data_masks:
            valid_data = valid_data_masks[var_key]
            lat = latgrid[valid_data]
        else:
            lat = latgrid

        if np.any(invalid_data):
            warnings.warn('Grid data resulted in invalid skill metric for '
                          'field: {}, removing for average...'.format(var_name))
            lat = lat[nonzero_data]

        # Get global average weights for field
        _, gm_weights = \
            LMR_outputs.get_area_avg_mask_and_weights(lat, None, None)

        lac_gm = lac[nonzero_data] @ gm_weights
        ce_gm = ce[nonzero_data] @ gm_weights
        avg_anom_corr = anom_corr.mean()

        ar1_lac_gm = ar1_lac[nonzero_data] @ gm_weights
        ar1_ce_gm = ar1_ce[nonzero_data] @ gm_weights
        ar1_avg_anom_corr = ar1_anom_corr.mean()

        spatial_gm_df = \
            mutils.ce_r_results_to_dataframe(var_name, avg_interval,
                                             'spatial_verif_gm', lac_gm, ce_gm,
                                             ar1_lac_gm, ar1_ce_gm,
                                             anom_corr=avg_anom_corr,
                                             auto1_anom_corr=ar1_avg_anom_corr)

        perf_fcast_dfs.append(spatial_gm_df)

        if plot_spatial_verif:

            plot_maps = [lac, ce, ar1_lac, ar1_ce]
            plot_metrs = ['LIM LAC', 'LIM CE', 'AR(1) LAC', 'AR(1) CE']

            for field, metric in zip(plot_maps, plot_metrs):
                valid_mask = valid_data_masks.get(var_key, None)
                sptl_shp = state_obj.var_space_shp[var_name]
                vutils.plot_spatial_verif(field, valid_mask, sptl_shp,
                                          latgrid, longrid, metric,
                                          fcast_against_src, avg_interval,
                                          var_name, fig_dir=perf_figdir)

            acorr_file = 'spatial_anomoly_corr_{}_{}.png'.format(var_name,
                                                                 var_key)
            acorr_path = os.path.join(perf_figdir, acorr_file)

            ptools.plot_anomaly_correlation(times, anom_corr,
                                            ar1_anom_corr, var_name,
                                            avg_interval, savefile=acorr_path)

    return perf_fcast_dfs


def ens_fcast_verification(state_lim_space, num_ens_members, lim, state_obj,
                           cfg_obj, lim_fcast_obj, base_keys, fcast_against_src,
                           fig_out_dir='.'):

    t0 = state_lim_space[:-1]
    ens_1yr_fcast = lutils.ens_1yr_fcast(num_ens_members, lim, t0)

    # load scalar factors for forecasting experiments
    grid_coords = next(iter(state_obj.var_coords.values()))
    latgrid = grid_coords['lat']
    longrid = grid_coords['lon']

    [scalar_factors,
     field_factors,
     base_scalar_factors] = get_scalar_factors(latgrid, longrid, cfg_obj,
                                               lim_fcast_obj, base_keys)
    ens_figdir = os.path.join(fig_out_dir, 'ens_fcast',
                              fcast_against_src)
    os.makedirs(ens_figdir, exist_ok=True)

    ens_scalar_output = {}
    [scalar_factors,
     ens_scalar_output] = vutils.handle_soi_factors(scalar_factors,
                                                    base_scalar_factors,
                                                    ens_scalar_output,
                                                    state_obj,
                                                    ens_1yr_fcast)

    for measure_key, factor in scalar_factors.items():
        var_key = measure_key[:-1]
        valid_data = lim_fcast_obj.valid_data_mask.get(var_key, None)
        ref_dat = mutils.get_field_from_state(state_obj, var_key,
                                              valid_data=valid_data)
        ref_measure = ref_dat @ base_scalar_factors[measure_key]
        fcast_measure = ens_1yr_fcast @ factor
        ens_scalar_output[measure_key] = (ref_measure, fcast_measure)

    ens_calib_dfs = []
    for measure_key, (ref_scalar, ens_scalar) in ens_scalar_output.items():

        ref_scalar = ref_scalar[1:]

        ens_calib = vutils.calc_ens_calib_ratio(ens_scalar,
                                                ref_scalar)
        crps = vutils.calc_crps(ens_scalar, ref_scalar)

        # create a 1x2 array
        frame_data = np.array([ens_calib, crps])[None, :]

        columns = ['ens_calib', 'crps']
        index = pd.MultiIndex.from_tuples((measure_key,),
                                          names=['Variable', 'Average',
                                                 'ScalarType'])
        df = pd.DataFrame(index=index,
                          columns=columns,
                          data=frame_data)
        ens_calib_dfs.append(df)

        if do_hist:
            title = ('Field: {} Avg Int: {} Measure: {}'.format(*measure_key))
            fig_fname = 'rank_hist_{}_{}_{}.png'.format(*measure_key)

            fig_path = os.path.join(ens_figdir, fig_fname)
            ptools.plot_rank_histogram(ens_scalar, ref_scalar, title,
                                       savefile=fig_path)

        if do_reliability:
            measure = measure_key[-1]

            if ('enso' in measure or 'nino' in measure or
               'pdo' in measure or 'npi' in measure or 'soi' in measure):

                std_factor = ref_scalar.std()
                ens_scalar_std = ens_scalar / std_factor
                ref_data_std = ref_scalar / std_factor

                title_temp = ('Field: {}  Avg Int: {} Metr: {} Reliability '
                              '(index {})')
                savefile_temp = 'reliability_{}_{}_{}_{}.png'

                pct_map = {'upper': '>0.5',
                           'lower': '<-0.5'}

                for event_type in ['upper', 'lower']:
                    obs_freq, bin_fcast_avg, errors =\
                        vutils.calc_reliability_with_bounds(
                            ens_scalar_std, ref_data_std, event_type=event_type
                        )
                    title = title_temp.format(*measure_key, pct_map[event_type])

                    fname = savefile_temp.format(*measure_key, event_type)
                    savefile = os.path.join(ens_figdir, fname)

                    ptools.plot_reliability(obs_freq, bin_fcast_avg, errors,
                                            title, savefile=savefile)

    ens_calib_fname = 'scalar_ens_calib_df.h5'
    ens_calib_fpath = os.path.join(ens_figdir, ens_calib_fname)
    ens_calib_out = pd.concat(ens_calib_dfs)
    ens_calib_out.to_hdf(ens_calib_fpath, fcast_against_src)


def run(cfg_class=None, fcast_against=None, figure_dir=None):

    if cfg_class is None:
        if not LMR_config.LEGACY_CONFIG:
            if len(sys.argv) > 1:
                yaml_file = sys.argv[1]
            else:
                yaml_file = os.path.join(LMR_config.SRC_DIR, 'config.yml')

            LMR_config.initialize_config_yaml(LMR_config, yaml_file)

        LMR_config.proxies.proxy_frac = 1.0
        cfg_class = LMR_config

    cfg = cfg_class.Config()

    # Create figure directory
    if figure_dir is None:
        figure_dir = os.path.join('.', cfg.core.nexp + '_lim_figs')
    else:
        figure_dir = os.path.join(figure_dir, cfg.core.nexp)

    os.makedirs(figure_dir, exist_ok=True)

    recon_period = cfg.core.recon_period
    save_analysis_ye = cfg.prior.outputs['analysis_Ye']

    if not base_only:
        # Get the necessary averaging intervals for the gridded data
        prox_manager = LMR_proxy.ProxyManager(cfg.proxies, cfg.psm,
                                              recon_period,
                                              include_eval=save_analysis_ye)
        req_avg_intervals = prox_manager.avg_interval_by_psm_type
    else:
        req_avg_intervals = {}

    # Load the state
    state = LMR_gridded.State.from_config(cfg.prior, req_avg_intervals)

    base_keys, psm_req_keys = \
        LMR_gridded.PriorVariable.get_base_and_psm_req_vars(cfg.prior,
                                                            req_avg_intervals)

    load_keys = base_keys + psm_req_keys
    lim_fcaster = LMR_forecaster.LIMForecaster.from_config(cfg.forecaster,
                                                           load_keys)

    regrid_grid = cfg.prior.regrid_cfg.esmpy_regrid_to

    # load scalar factors for forecasting experiments
    var_key, grid_coords = next(iter(state.var_coords.items()))
    lat = grid_coords['lat']
    lon = grid_coords['lon']
    space_shp = state.var_space_shp[var_key]

    if plot_eofs:
        print('Plotting variable EOFs.')
        fig_fname = os.path.join(figure_dir,
                                 '{}_basis_eofs.png'.format(regrid_grid))
        dobj_eofs = {var_key: eofs[:, :plot_neofs]
                     for var_key, eofs in lim_fcaster.var_eofs.items()}
        ptools.plot_exp_eofs(dobj_eofs, state, lim_fcaster.valid_data_mask,
                             var_eof_stats=lim_fcaster.var_eof_stats,
                             filename=fig_fname)

    if plot_state_eofs:
        print('Plotting multi-variable EOFs.')
        fig_fname = os.path.join(figure_dir,
                                 '{}_multivar_eofs.png'.format(regrid_grid))

        multivar_eofs = {}
        for var_key, var_eofs in lim_fcaster.var_eofs.items():
            multi_eof_var_span = lim_fcaster.var_span[var_key]
            vstart, vend = multi_eof_var_span
            state_eofs = lim_fcaster.calib_eofs[vstart:vend, :plot_neofs]

            multivar_eofs[var_key] = var_eofs @ state_eofs

        title = 'Multivar EOF_{:d}  Field: {}'

        ptools.plot_exp_eofs(multivar_eofs, state, lim_fcaster.valid_data_mask,
                             multi_var_eof_stats=lim_fcaster.multi_var_eof_stats,
                             filename=fig_fname, title=title)

    lim = lim_fcaster.lim

    if plot_lim_noise_eofs:
        fig_fname = os.path.join(figure_dir,
                                 '{}_noise_eofs.png'.format(regrid_grid))
        Q_evect = lim.Q_evects[:, :plot_num_noise_modes].real
        noise_eofs = {}
        for var_key, var_eofs in lim_fcaster.var_eofs.items():
            multi_eof_var_span = lim_fcaster.var_span[var_key]
            vstart, vend = multi_eof_var_span
            state_eofs = lim_fcaster.calib_eofs[vstart:vend, :]

            noise_eofs[var_key] = var_eofs @ state_eofs @ Q_evect

        title = 'Noise EOF_{:d}  Field: {}'

        ptools.plot_exp_eofs(noise_eofs, state, lim_fcaster.valid_data_mask,
                             filename=fig_fname, title=title)

    if plot_lim_modes:
        print('Plotting LIM modes!')
        fig_fname = os.path.join(figure_dir,
                                 '{}_lim_fcast_modes.png'.format(regrid_grid))
        ptools.plot_multi_lim_modes(lim, lat, lon, space_shp, lim_fcaster,
                                    row_limit=plot_num_lim_modes,
                                    save_file=fig_fname)

    if do_perfect_fcast or do_ens_fcast:

        if fcast_against is not None:
            cfg_class.prior.prior_source = fcast_against
        else:
            fcast_against = cfg_class.prior.prior_source

        cfg_class.core.nens = None
        cfg_class.prior.detrend = detrend_fcast_ref_data

        full_time_cfg = cfg_class.Config()

        state = LMR_gridded.State.from_config(full_time_cfg.prior,
                                              req_avg_intervals=req_avg_intervals)

        reduced_state, compressed = \
        lim_fcaster.phys_space_data_to_fcast_space(state,
                                                   is_diff_model)

        start = fcast_start_yr
        end = start + reduced_state.shape[0]
        times = list(range(start, end))[1:]

        if do_perfect_fcast:
            perfect_fcast_verification(state, cfg, lim_fcaster, reduced_state,
                                       times, base_keys, fcast_against,
                                       fig_out_dir=figure_dir)

        if do_ens_fcast:
            ens_fcast_verification(reduced_state, nens, lim, state, cfg,
                                   lim_fcaster, base_keys, fcast_against,
                                   fig_out_dir=figure_dir)
    else:
        reduced_state, _ = lim_fcaster.phys_space_data_to_fcast_space(state)

    if do_long_integration:

        t0 = reduced_state[0:1, :]

        # long integration with buffer of 50 years to forget initial state
        last = lutils.ens_long_integration(integration_iters,
                                           integration_len_yr + 50,
                                           lim, t0)

        last = last[50:]

        fname = 'long_integration_output_{}.npy'.format(regrid_grid)
        path = os.path.join(figure_dir, fname)
        np.save(path, last)

        # load scalar factors for forecasting experiments
        grid_coords = next(iter(state.var_coords.values()))
        latgrid = grid_coords['lat']
        longrid = grid_coords['lon']

        [scalar_factors,
         field_factors,
         base_scalar_factors] = get_scalar_factors(latgrid, longrid, cfg,
                                                   lim_fcaster, base_keys)

        scalar_outdef = cfg.prior.outputs['scalar_ens']
        [func_by_var, scalar_output_containers] = \
            LMR_outputs.prepare_scalar_calculations(scalar_outdef,
                                                    state, cfg.prior,
                                                    integration_len_yr,
                                                    integration_iters,
                                                    return_insert_func=False)

        # Calculate scalar output values defined in config
        for _var_key, scalar_funcs in func_by_var.items():

            curr_containers = scalar_output_containers[_var_key]

            eof_var_span = lim_fcaster.var_span[_var_key]
            var_slice = slice(*eof_var_span)
            multi_eof_var = lim_fcaster.calib_eofs[var_slice, :]
            var_eof = lim_fcaster.var_eofs[_var_key]

            _varname, _avg_interval = _var_key

            std_factor = lim_fcaster.var_eof_std_factor[_var_key]
            last_var_eofspace = last @ multi_eof_var.T
            last_var_eofspace = last_var_eofspace / std_factor

            decompress = _var_key in lim_fcaster.valid_data_mask

            if _var_key in lim_fcaster.var_std_factor:
                var_std_factor = lim_fcaster.var_std_factor[_var_key]
            else:
                var_std_factor = None

            time_chk = 100

            for start in np.arange(0, integration_len_yr, time_chk):

                end = start + time_chk
                if end >= integration_len_yr:
                    end = None

                time_slice = slice(start, end)
                curr_dat = last_var_eofspace[time_slice]

                phys_curr_dat = curr_dat @ var_eof.T
                ens_yr_shp = phys_curr_dat.shape[:-1]
                sptl_shp = phys_curr_dat.shape[-1]
                phys_curr_dat = phys_curr_dat.reshape(-1, sptl_shp)

                if var_std_factor is not None:
                    phys_curr_dat = phys_curr_dat / var_std_factor

                if decompress:
                    phys_curr_dat = lim_fcaster._decompress_field(_var_key,
                                                                  phys_curr_dat)
                for _measure, _measure_func in scalar_funcs.items():
                    scalar_data = _measure_func(phys_curr_dat.T)
                    scalar_data = scalar_data.reshape(*ens_yr_shp)
                    _container = curr_containers[_measure]
                    _container[time_slice] = scalar_data

        LMR_outputs.save_scalar_ensembles(figure_dir,
                                          np.arange(integration_len_yr),
                                          scalar_output_containers)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = os.path.join(LMR_config.SRC_DIR, 'config.yml')

    ### Single Run
    LMR_config.initialize_config_yaml(LMR_config, yaml_file)
    LMR_config.proxies.proxy_frac = 1.0
    LMR_config.core.nexp = 'testdev_ccsm_fcast_on_mpi'
    run(LMR_config, fcast_against=fcast_against,
        figure_dir=fig_dir)

    ### Sensitivity Experiments
    # levels
    # params = [
    #           2, 5, 10, 15, 20, 25,
    #           30, 31, 32, 33, 34, 35, 36,
    #           37, 38, 39,
    #           40,
    #           # 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
    # ]
    # # params = [20, 25,
    # #           30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    # #           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # # params = [43]
    # pname = 'nmodes'
    # nexp = 'testdev_mpi_atmocn_coupled_retmodes{:d}'
    # proxy_frac = 1.0
    #
    # # nens
    # # params = [5, 10, 25, 50, 100, 200]
    # # pname = 'nens'
    # # nexp = 'testdev_{:d}ens_seasbil_ccsm4_past1000_43modes'
    # # proxy_frac = 1.0
    #
    # # mc iters
    # # params = np.arange(5)
    # # pname = 'mc_iter'
    # # nexp = 'testdev_{:d}iter_100ens_seasbil_ccsm4_past1000_34modes'
    # # proxy_frac = 0.75
    #
    # for i, param in enumerate(params):
    #
    #     LMR_config.initialize_config_yaml(LMR_config, yaml_file)
    #     print('RUN SENSITIVITY EXP ({}={:d})'.format(pname, param))
    #     LMR_config.proxies.proxy_frac = proxy_frac
    #     LMR_config.core.nexp = nexp.format(param)
    #
    #     # for mc iters
    #     # LMR_config.core.seed = param
    #
    #     # for modes
    #     LMR_config.forecaster.lim.fcast_num_pcs = param
    #
    #     # for nens
    #     # nens = param
    #
    #     run(cfg_class=LMR_config, fcast_against=fcast_against,
    #         figure_dir=fig_dir)
