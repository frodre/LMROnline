import LMR_proxy
import LMR_gridded
import LMR_forecaster
import LMR_config
import LMR_outputs

import logging
import sys
import os
import pickle
import numpy as np

import lim_diagnostics.plot_tools as ptools
import lim_diagnostics.lim_utils as lutils
import lim_diagnostics.verif_utils as vutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# defaults to config.core.nexp in working directory
fig_dir = None

# Fig Output
plot_neofs = 5
plot_eofs = False
plot_state_eofs = False

plot_lim_modes = False
plot_num_lim_modes = 8

plot_lim_noise_eofs = False
plot_num_noise_modes = 8

# Perfect Forecast Experiments
do_perfect_fcast = True
fcast_outputs = {'tas': ['glob_mean'],
                 'tos': ['glob_mean',
                         'enso',
                         'pdo'],
                 'zos': ['glob_mean']}
verif_spec = {'zos': 'eof_proj'}
plot_scalar_verif = True
plot_spatial_verif = True

# Ensemble noise integration forecast experiments
do_ens_fcast = True
do_hist = True
do_reliability = True

# Long integration forecast experiments
do_long_integration = True
integration_len_yr = 1000
integration_iters = 500

# ========================
# END USER PARAMS
# ========================

if not LMR_config.LEGACY_CONFIG:
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = os.path.join(LMR_config.SRC_DIR, 'config.yml')

    LMR_config.initialize_config_yaml(LMR_config, yaml_file)

cfg = LMR_config.Config()

# Create figure directory
if fig_dir is None:
    fig_dir = os.path.join('.', cfg.core.nexp + '_lim_figs')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

recon_period = cfg.core.recon_period
save_analysis_ye = cfg.prior.outputs['analysis_Ye']

prox_manager = LMR_proxy.ProxyManager(cfg.proxies, cfg.psm,
                                      recon_period,
                                      include_eval=save_analysis_ye)
req_avg_intervals = prox_manager.avg_interval_by_psm_type

state = LMR_gridded.State.from_config(cfg.prior, req_avg_intervals)

base_keys, psm_req_keys = \
    LMR_gridded.PriorVariable.get_base_and_psm_req_vars(cfg.prior,
                                                        req_avg_intervals)

load_keys = base_keys + psm_req_keys
lim_fcaster = LMR_forecaster.LIMForecaster.from_config(cfg.forecaster,
                                                       load_keys)

regrid_grid = cfg.prior.regrid_cfg.esmpy_regrid_to


if plot_eofs:
    print('Plotting variable EOFs.')
    fig_fname = os.path.join(fig_dir,
                             '{}_basis_eofs.png'.format(regrid_grid))
    dobj_eofs = {var_key: eofs[:, :plot_neofs]
                 for var_key, eofs in lim_fcaster.var_eofs.items()}
    ptools.plot_exp_eofs(dobj_eofs, state, lim_fcaster.valid_data_mask,
                         var_eof_stats=lim_fcaster.var_eof_stats,
                         filename=fig_fname)

if plot_state_eofs:
    print('Plotting multi-variable EOFs.')
    fig_fname = os.path.join(fig_dir,
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
    fig_fname = os.path.join(fig_dir,
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
    fig_fname = os.path.join(fig_dir,
                             '{}_lim_fcast_modes.png'.format(regrid_grid))
    ptools.plot_multi_lim_modes(lim, state, lim_fcaster,
                                row_limit=plot_num_lim_modes,
                                save_file=fig_fname)


if do_long_integration:

    fcast_state, is_compressed = lim_fcaster.phys_space_data_to_fcast_space(state)
    t0 = fcast_state[0:1, :]

    # long integration with buffer of 50 years to forget initial state
    last = lutils.ens_long_integration(integration_iters,
                                       integration_len_yr+50,
                                       lim, t0)

    last = last[50:]

    fname = 'long_integration_output_{}.npy'.format(regrid_grid)
    path = os.path.join(fig_dir, fname)
    np.save(path, last)

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

    LMR_outputs.save_scalar_ensembles(fig_dir, np.arange(integration_len_yr),
                                      scalar_output_containers)
