"""
Module: LMR_driver_callable.py

Purpose: This is the "main" module of the LMR code.
         Generates a paleoclimate reconstruction (single Monte-Carlo
         realization) through the assimilation of a set of proxy data.

Options: None.
         Experiment parameters defined in LMR_config.

Originators: Greg Hakim    | Dept. of Atmospheric Sciences, Univ. of Washington
             Robert Tardif | January 2015

Revisions:
  April 2015:
            - This version is callable by an outside script, accepts a single
              object, called state, which has everything needed for the driver
              (G. Hakim - U. of Washington)

            - Re-organisation of code around PSM calibration and calculation of
              Ye. Code now assumes PSM parameters have been pre-calulated and
              Ye's are calculated up-front for all proxy types/sites. All
              proxy data are now also loaded up-front, prior to any loops.
              Ye's are appended to state vector to form an augmented state
              vector and are also updated by DA. (R. Tardif - U. of Washington)
    May 2015:
            - Bug fix in calculation of global mean temperature + function
              now part of LMR_utils.py (G. Hakim - U. of Washington)
   July 2015:
            - Switched time & proxy loops, simplified logic so more of the
              proxy and PSM specifics are contained within their classes,
              formatted to mostly adhere to PEP8 guidlines
              (A. Perkins - U. of Washington)
  April 2016:
            - Added handling of the "sensitivity" attribute now attached to
              proxy psm objects that defines the climate variable to which
              each proxy record is deemed sensitive to.
              (R. Tardif - U. of Washington)
   July 2016:
            - Slight code adjustments for handling possible use of PSM calibrated 
              on the basis of proxy records seasonality metadata.
              (R. Tardif - U. of Washington)
 August 2016:
            - Introduced new function that loads pre-calculated Ye values 
              generated using psm types assigned to individual proxy types
              as defined in the experiment configuration. 
              (R. Tardif - U. of Washington)
   Feb. 2017:
            - Modifications to temporal loop to allow the production of 
              reconstructions at lower temporal resolution (i.e. other
              than annual).
              (R. Tardif - U. of Washington)
  March 2017:
            - Added possibility to by-pass the regridding (truncation of the state).
              (R. Tardif - U. of Washington)
            - Added another option for regridding that works on gridded 
              fields with missing values (masked grid points. e.g. ocean fields) 
              (R. Tardif - U. of Washington)
            - Replaced the hared-coded truncation resolution (T42) of spatial fields 
              updated during the DA (i.e. reconstruction resolution) by a 
              user-specified value set in the configuration.
 August 2017:
            - Included the Ye's from withheld proxies to state vector so they get 
              updated during DA as well for easier & complete proxy-based evaluation
              of reconstruction. (R. Tardif - U. of Washington)
"""
import numpy as np
from os.path import join
from time import time

import LMR_proxy2
import LMR_gridded
from LMR_utils2 import global_mean2
import LMR_outputs as lmr_out
import LMR_config as BaseCfg
import LMR_forecaster
from LMR_DA import enkf_update_array_xb_blend, cov_localization


# *** main driver
def LMR_driver_callable(cfg=None):

    if cfg is None:
        cfg = BaseCfg.Config()  # Use base configuration from LMR_config

    # Temporary fix for old 'state usage'
    core_cfg = cfg.core
    prior_cfg = cfg.prior
    output_avg_interval = prior_cfg.avg_interval

    # verbose controls print comments (0 = none; 1 = most important;
    #  2 = many; 3 = a lot; >=4 = all)
    verbose = cfg.LOG_LEVEL

    nexp = core_cfg.nexp
    workdir = core_cfg.datadir_output
    recon_period = core_cfg.recon_period
    online = core_cfg.online_reconstruction
    hybrid_update = core_cfg.hybrid_update
    hybrid_a_val = core_cfg.hybrid_a
    blend_prior = core_cfg.blend_prior
    reg_inf = core_cfg.reg_inflate
    inf_factor = core_cfg.inflation_factor
    nens = core_cfg.nens
    loc_rad = core_cfg.loc_rad
    inflation_fact = core_cfg.inflation_factor
    outputs = prior_cfg.outputs
    save_analysis_ye = outputs['analysis_Ye']

    # ==========================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ==========================================================================
    # TODO: AP Logging instead of print statements
    if verbose > 0:
        print('')
        print('=====================================================')
        print('Running LMR reconstruction...')
        print('=====================================================')
        print('Name of experiment: ', nexp)
        print(' Monte Carlo iter : ', core_cfg.curr_iter)
        print('')
        
    begin_time = time()

    # Define the number of years of the reconstruction (nb of assimilation
    # times)
    ntimes = recon_period[1] - recon_period[0] + 1
    recon_times = np.arange(recon_period[0], recon_period[1]+1)

    # ==========================================================================
    # Get information on proxies to assimilate ---------------------------------
    # ==========================================================================

    begin_time_proxy_load = time()
    if verbose > 0:
        print('')
        print('-----------------------------------')
        print('Uploading proxy data & PSM info ...')
        print('-----------------------------------')

    # Build dictionaries of proxy sites to assimilate and those set aside for
    # verification
    prox_manager = LMR_proxy2.ProxyManager(cfg.proxies, cfg.psm,
                                           recon_period)
    req_avg_intervals = prox_manager.avg_interval_by_psm_type

    # Convert the required average interval keys to the prior variable name
    psm_var_map = prior_cfg.psm_var_map
    for key in list(req_avg_intervals.keys()):
        curr_avg_intervals = req_avg_intervals.pop(key)
        psm_sensitivity, psm_generic_var = key
        prior_varname = psm_var_map[psm_sensitivity][psm_generic_var]
        req_avg_intervals[prior_varname] = curr_avg_intervals

    type_site_assim = prox_manager.assim_ids_by_group
    # count the total number of proxies
    assim_proxy_count = len(prox_manager.ind_assim)
    # count the total witheld proxies
    if prox_manager.ind_eval:
        eval_proxy_count = len(prox_manager.ind_eval)
    else:
        eval_proxy_count = None

    if verbose > 0:
        print('Assimilating proxy types/sites:', type_site_assim)
        print('--------------------------------------------------------------------')
        print('Proxy counts for experiment:')
        for pkey, plist in sorted(type_site_assim.items()):
            print(('%45s : %5d' % (pkey, len(plist))))
        print(('%45s : %5d' % ('TOTAL', assim_proxy_count)))
        print('--------------------------------------------------------------------')

    if verbose > 2:
        proxy_load_time = time() - begin_time_proxy_load
        print('-----------------------------------------------------')
        print('Loading completed in ' + str(proxy_load_time) + ' seconds')
        print('-----------------------------------------------------')

    # ==========================================================================
    # Load prior data ----------------------------------------------------------
    # ==========================================================================
    if verbose > 0:
        print('-------------------------------------------')
        print('Uploading gridded (model) data as prior ...')
        print('-------------------------------------------')
        print('Source for prior: ', prior_cfg.prior_source)

    # Create initial state vector of desired variables at smallest time res
    Xb_one = LMR_gridded.State.from_config(prior_cfg,
                                           req_avg_intervals=req_avg_intervals)
    state_vars = '_'.join([var_key[0]
                           for var_key in Xb_one.base_prior_keys])
    h5f_path = join(workdir,
                    'recon_output_{}_{}.h5'.format(state_vars,
                                                   output_avg_interval))

    [calc_and_store_scalars,
     scalar_containers] = \
        lmr_out.prepare_scalar_calculations(outputs['scalar_ens'], Xb_one,
                                            prior_cfg, ntimes, nens)

    [field_hdf5_outputs,
     field_get_ens_func] = lmr_out.prepare_field_output(outputs, Xb_one, ntimes,
                                                        nens, h5f_path)

    load_time = time() - begin_time
    if verbose > 2:
        print('-----------------------------------------------------')
        print('Loading completed in ' + str(load_time)+' seconds')
        print('-----------------------------------------------------')

    # check covariance inflation from config
    if inflation_fact is not None and verbose > 2:
        print(('\nUsing covariance inflation factor: %8.2f' % inflation_fact))

    # Keep dimension of pre-augmented version of state vector
    state_dim = Xb_one.shape[0]

    if save_analysis_ye:
        assim_ye_path = join(workdir, 'assim_ye_ens_output.zarr')
        assim_ye_out = lmr_out.create_Ye_output(assim_ye_path,
                                                assim_proxy_count,
                                                nens, ntimes)

        if eval_proxy_count is not None:
            eval_ye_path = join(workdir, 'eval_ye_ens_output.zarr')
            eval_ye_out = lmr_out.create_Ye_output(eval_ye_path,
                                                   eval_proxy_count,
                                                   nens, ntimes)
        else:
            eval_ye_out = None
    else:
        assim_ye_out = None
        eval_ye_out = None

    # ----------------------------------
    # Augment state vector with the Ye's
    # ----------------------------------

    # TODO: Figure out how to handle precalculated YE Vals
    # Extract all the Ye's from master list of proxy objects into numpy array
    ye_all = LMR_proxy2.calc_assim_ye_vals(prox_manager, Xb_one)
    Xb_one.augment_state(ye_all)

    # TODO: Switch to cPickled prior object... right now hardcoded for annual
    # case saving
    # Dump prior state vector (Xb_one) to file 
    # filen = workdir + '/' + 'Xb_one'
    # state = Xb_one.get_var_data('state').copy()
    # aug_state = Xb_one.state.copy()
    # nan_vals = np.isnan(state)
    # if np.any(nan_vals):
    #     state[nan_vals] = 1.0e20
    #     aug_state[np.isnan(aug_state)] = 1.0e20
    # else:
    #     state = Xb_one.get_var_data('state')
    #     aug_state = Xb_one.state
    # np.savez(filen, Xb_one=state,
    #          Xb_one_aug=aug_state,
    #          stateDim=state_dim,
    #          Xb_one_coords=Xb_one.var_coords,
    #          state_info=Xb_one.old_state_info)

    # TODO: replicate single variable prior saving

    # Initialize forecaster for online reconstructions
    if online:
        print('\n Initializing LMR forecasting for online reconstruction')
        key = cfg.forecaster.use_forecaster
        fcastr_class = LMR_forecaster.get_forecaster_class(key)
        forecaster = fcastr_class(cfg.forecaster, Xb_one.var_keys)

    # ==========================================================================
    # Loop over all years and proxies, and perform assimilation ----------------
    # ==========================================================================
    Xb_one.stash_state('orig')

    ens_calib_check = np.zeros((assim_proxy_count, 5))

    start_yr, end_yr = recon_period
    assim_times = np.arange(start_yr, end_yr+1)

    # ---------------------
    # Loop over proxy types
    # ---------------------
    lasttime = time()
    for iyr, t in enumerate(assim_times):

        if verbose > 0:
            print('working on year: ' + str(t))
            # Store original annual for hybrid update
        if hybrid_update:
            if iyr == 0:
                # Creates a copy for use as our static prior
                Xb_one.stash_state('orig_aug')
                Xb_static = Xb_one.state
                Yevals_static = Xb_one.get_var_data('ye_vals')
                Xb_one.stash_recall_state_list('orig_aug',
                                               copy=True)
            else:
                Xb_one.stash_state('tmp')
                Xb_one.stash_recall_state_list('orig_aug', copy=True)
                Xb_static = Xb_one.state
                Yevals_static = Xb_one.get_var_data('ye_vals')
                Xb_one.stash_pop_state_list('tmp')

            if blend_prior:
                xbf = Xb_one.state
                blend_forecast = (hybrid_a_val * xbf +
                                  (1-hybrid_a_val) * Xb_static)
                Xb_one.state = blend_forecast

        # Save output fields for the prior
        lmr_out.save_field_output(iyr, 'prior', Xb_one, field_hdf5_outputs,
                                  output_def=outputs['prior'])

        # Update Xb with each proxy
        for iproxy, Y in enumerate(prox_manager.sites_assim_proxy_objs()):

            # Crude check if we have proxy ob for current time
            try:
                Y.values[t]
            except KeyError:
                continue

            if verbose > 1:
                print('--------------- Processing proxy: ' + Y.id)
            if verbose > 2:
                print('Site:', Y.id, ':', Y.type)
                print(' latitude, longitude: ' + str(Y.lat), str(Y.lon))

            loc = None
            if loc_rad is not None:
                if verbose > 2:
                    print('...computing localization...')
                    raise NotImplementedError('Covariance localization'
                                              ' not properly implemented'
                                              ' yet.')
                    # loc = cov_localization(loc_rad, X, Y)

            # Get Ye values for current proxy
            Ye = Xb_one.get_var_data('ye_vals')[iproxy]

            # Define the ob error variance
            ob_err = Y.psm_obj.R

            # TODO: If I ever do subannual forecasts need to adjust this
            mse = np.mean((Y.values[t] - Ye)**2)
            y_ye_var = Ye.var(ddof=1) + ob_err
            ens_calib_check[iproxy, iyr % 5] = mse / y_ye_var
            if not np.all(np.isfinite(ens_calib_check)):
                raise FilterDivergenceError('Filter divergence detected'
                                            ' during year {}. Skipping '
                                            'iteration.'.format(t))

            # --------------------------------------------------------------
            # Do the update (assimilation) ---------------------------------
            # --------------------------------------------------------------
            if verbose > 2:
                print(('updating time: ' + str(t) + ' proxy value : ' +
                       str(Y.values[t]) + ' | mean prior proxy estimate: ' +
                       str(Ye.mean())))

            # Get static Ye for hybrid update
            if hybrid_update:
                Ye_static = Yevals_static[iproxy]
                hybrid_tup = (Xb_static, Ye_static)
            else:
                hybrid_tup = None

            # Update the state
            Xa = enkf_update_array_xb_blend(Xb_one.state,
                                            Y.values[t], Ye, ob_err, loc,
                                            static_prior=hybrid_tup,
                                            a=hybrid_a_val)

            Xb_one.state = Xa


            # check the variance change for sign
            thistime = time()
            lasttime = thistime


        # Check for filter divergence
        if ens_calib_check.mean() > 50:
            raise FilterDivergenceError('Filter divergence detected'
                                        ' during year {}. Skipping '
                                        'iteration.'.format(t))

        # Calculate and store index values from field
        calc_and_store_scalars(Xb_one, iyr)
        # Calculate and store posterior field reductions
        lmr_out.save_field_output(iyr, 'posterior', Xb_one, field_hdf5_outputs,
                                  output_def=outputs['posterior'])

        # Save field ensemble members
        if outputs['field_ens_output'] is not None:
            lmr_out.save_field_output(iyr, 'field_ens_output', Xb_one,
                                      field_hdf5_outputs,
                                      ens_out_func=field_get_ens_func)

        # Save Ye Information
        if save_analysis_ye:
            assim_ye = Xb_one.get_var_data('ye_vals')
            assim_ye_out[:, iyr] = assim_ye

            if eval_proxy_count is not None:
                eval_ye = LMR_proxy2.calc_eval_ye_vals(prox_manager, Xb_one)
                eval_ye_out[:, iyr] = eval_ye

        if online:

            forecaster.forecast(Xb_one)

            # Recalculate Ye values
            ye_all = LMR_proxy2.calc_assim_ye_vals(prox_manager, Xb_one)
            Xb_one.reset_augmented_ye(ye_all)

            # Inflation Adjustment
            if reg_inf:
                Xb_one.reg_inflate_xb(inf_factor)

    end_time = time() - begin_time

    # End of loop on proxy types
    if verbose > 0:
        print('')
        print('=====================================================')
        print('Reconstruction completed in ' + str(end_time/60.0)+' mins')
        print('=====================================================')

    if verbose > 0:
        if assim_ye_out is not None:
            print('-----------------------------------')
            print('Assimilated proxy Ye output info...')
            print(assim_ye_out.info)
            print('-----------------------------------')

        if eval_ye_out is not None:
            print('-----------------------------------')
            print('Witheld proxy Ye output info...')
            print(eval_ye_out.info)
            print('-----------------------------------')

    # Save Scalar information and proxies assimilated/withheld
    lmr_out.save_scalar_ensembles(workdir, recon_times, scalar_containers)
    lmr_out.save_recon_proxy_information(prox_manager, workdir)

    field_hdf5_outputs.close()

    exp_end_time = time() - begin_time
    if verbose > 0:
        print('')
        print('=====================================================')
        print('Experiment completed in ' + str(exp_end_time/60.0) + ' mins')
        print('=====================================================')

# ------------------------------------------------------------------------------
# --------------------------- end of main code ---------------------------------
# ------------------------------------------------------------------------------


class FilterDivergenceError(ValueError):
    pass


if __name__ == '__main__':
    LMR_driver_callable()
