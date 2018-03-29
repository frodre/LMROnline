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

from . import LMR_proxy2
from . import LMR_gridded
from .LMR_utils2 import global_mean2
from . import LMR_config as BaseCfg
from . import LMR_forecaster
from .LMR_DA import enkf_update_array_xb_blend, cov_localization


# *** Helper Methods
def _calc_yevals_from_prior(ye_shp, xb_state, prox_manager):
    ye_all = np.empty(shape=ye_shp)
    for i, pobj in enumerate(prox_manager.sites_assim_proxy_objs()):
        ye_all[i, :] = pobj.psm(xb_state)

    return ye_all

# *** main driver
def LMR_driver_callable(cfg=None):

    if cfg is None:
        cfg = BaseCfg.Config()  # Use base configuration from LMR_config

    # Temporary fix for old 'state usage'
    core_cfg = cfg.core
    prior_cfg = cfg.prior

    # verbose controls print comments (0 = none; 1 = most important;
    #  2 = many; 3 = a lot; >=4 = all)
    verbose = cfg.LOG_LEVEL

    nexp = core_cfg.nexp
    workdir = core_cfg.datadir_output
    recon_period = core_cfg.recon_period
    online = core_cfg.online_reconstruction
    persistence = core_cfg.persistence_forecast
    hybrid_update = core_cfg.hybrid_update
    hybrid_a_val = core_cfg.hybrid_a
    blend_prior = core_cfg.blend_prior
    reg_inf = core_cfg.reg_inflate
    inf_factor = core_cfg.inflation_factor
    nens = core_cfg.nens
    loc_rad = core_cfg.loc_rad
    inflation_fact = core_cfg.inflation_factor
    state_backend = prior_cfg.backend_type

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
    # Load prior data ----------------------------------------------------------
    # ==========================================================================
    if verbose > 0:
        print('-------------------------------------------')
        print('Uploading gridded (model) data as prior ...')
        print('-------------------------------------------')
        print('Source for prior: ', prior_cfg.prior_source)

    # Create initial state vector of desired variables at smallest time res
    Xb_one = LMR_gridded.State.from_config(prior_cfg)

    load_time = time() - begin_time
    if verbose > 2:
        print('-----------------------------------------------------')
        print('Loading completed in ' + str(load_time)+' seconds')
        print('-----------------------------------------------------')

    # check covariance inflation from config
    if inflation_fact is not None and verbose > 2:
        print(('\nUsing covariance inflation factor: %8.2f' %inflate))

    # ==========================================================================
    # Calculate regridded state from prior, if option chosen -------------------
    # ==========================================================================

    # TODO: Regridding here
    # if trunc_state:
    #     Xb_one = Xb_one_full.truncate_state()
    # else:
    #     Xb_one = Xb_one_full.copy()

    # Keep dimension of pre-augmented version of state vector
    state_dim = Xb_one.shape[0]

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
    prox_manager = LMR_proxy2.ProxyManager(cfg.proxies, recon_period,
                                           cfg.psm)
    type_site_assim = prox_manager.assim_ids_by_group
    # count the total number of proxies
    assim_proxy_count = len(prox_manager.ind_assim)

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

    # ----------------------------------
    # Augment state vector with the Ye's
    # ----------------------------------

    # TODO: Figure out how to handle precalculated YE Vals
    # TODO: append eval proxy YE values
    # Extract all the Ye's from master list of proxy objects into numpy array
    ye_shp = (assim_proxy_count, nens)
    ye_all = _calc_yevals_from_prior(ye_shp, Xb_one, prox_manager)
    Xb_one.augment_state(ye_all)

    # TODO: Switch to cPickled prior object... right now hardcoded for annual
    # case saving
    # Dump prior state vector (Xb_one) to file 
    filen = workdir + '/' + 'Xb_one'
    if isinstance(Xb_one.state, np.ma.MaskedArray):
        state = Xb_one.get_var_data('state').filled()
        aug_state = Xb_one.state.filled()
    else:
        state = Xb_one.get_var_data('state')
        aug_state = Xb_one.state
    np.savez(filen, Xb_one=state,
             Xb_one_aug=aug_state,
             stateDim=state_dim,
             Xb_one_coords=Xb_one.var_coords,
             state_info=Xb_one.old_state_info)

    # TODO: replicate single variable prior saving

    # Initialize forecaster for online reconstructions
    if online:
        print('\n Initializing LMR forecasting for online reconstruction')
        key = cfg.forecaster.use_forecaster
        fcastr_class = LMR_forecaster.get_forecaster_class(key)
        forecaster = fcastr_class(cfg.forecaster)

    # ==========================================================================
    # Loop over all years and proxies, and perform assimilation ----------------
    # ==========================================================================
    Xb_one.stash_state('orig')

    # Array containing the global and hemispheric-mean state
    # Now doing surface air temperature only (var = tas_sfc_Amon)!
    gmt_save = np.zeros([assim_proxy_count+1, ntimes])
    nhmt_save = np.zeros([assim_proxy_count+1, ntimes])
    shmt_save = np.zeros([assim_proxy_count+1, ntimes])

    xbm = Xb_one.get_var_data('tas_sfc_Amon').mean(axis=1)  # ensemble mean
    gmt, nhmt, shmt = global_mean2(xbm,
                                   Xb_one.var_coords['tas_sfc_Amon']['lat'],
                                   output_hemispheric=True)
    # First row is prior GMT
    gmt_save[0, :] = gmt
    nhmt_save[0, :] = nhmt
    shmt_save[0, :] = shmt
    # Prior for first proxy assimilated
    gmt_save[1, :] = gmt 
    nhmt_save[1, :] = nhmt
    shmt_save[1, :] = shmt

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


        # Save prior variance if online assimilation
        if online:
            xbv = Xb_one.get_var_data('tas_sfc_Amon')
            xbv = xbv.var(ddof=1, axis=1)
            grd_shp = Xb_one.var_space_shp['tas_sfc_Amon']
            xbv = xbv.reshape(grd_shp)
            if iyr == 0:
                xbv_out = np.zeros((len(assim_times), grd_shp[0],
                                    grd_shp[1]))
            xbv_out[iyr] = xbv

        # Update Xb with each proxy
        for iproxy, Y in enumerate(prox_manager.sites_assim_proxy_objs()):

            # Crude check if we have proxy ob for current time
            try:
                Y.values[t]
            except KeyError:
                # Make sure GMT spot filled from previous proxy
                gmt_save[iproxy+1, iyr] = gmt_save[iproxy, iyr]
                nhmt_save[iproxy+1, iyr] = nhmt_save[iproxy, iyr]
                shmt_save[iproxy+1, iyr] = shmt_save[iproxy, iyr]
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

            xam = Xb_one.get_var_data('tas_sfc_Amon').mean(axis=1)
            gmt, nhmt, shmt = \
                global_mean2(xam,
                             Xb_one.var_coords['tas_sfc_Amon']['lat'],
                             output_hemispheric=True)
            gmt_save[iproxy+1, iyr] = gmt
            nhmt_save[iproxy+1, iyr] = nhmt
            shmt_save[iproxy+1, iyr] = shmt


            # check the variance change for sign
            thistime = time()
            # if verbose > 2:
            #     xbvar = Xb.var(axis=1, ddof=1)
            #     xavar = Xa.var(ddof=1, axis=1)
            #     vardiff = xavar - xbvar
            #     print 'max change in variance:' + str(np.max(vardiff))
            #     print 'update took ' + str(thistime-lasttime) + 'seconds'
            lasttime = thistime


        # Check for filter divergence
        if ens_calib_check.mean() > 50:
            raise FilterDivergenceError('Filter divergence detected'
                                        ' during year {}. Skipping '
                                        'iteration.'.format(t))
        # Save annual data to file
        ypad = '{:04d}'.format(int(t))
        filen = join(workdir, 'year' + ypad + '.npy')
        if isinstance(Xb_one.state, np.ma.MaskedArray):
            state = Xb_one.state.filled()
        else:
            state = Xb_one.state
        np.save(filen, state)

        if online:
            # Push sub_base prior to next year
            inext_yr = iyr + 1

            if not persistence:
                # Forecast
                fcast_out = forecaster.forecast(Xb_one)

                # Recall orig state, any vars not forecast are climatological
                #  sample
                Xb_one.stash_recall_state_list('orig', copy=True)
                Xb_one.update_var_data(fcast_out)

                #Recalculate Ye values
                ye_all = _calc_yevals_from_prior(ye_shp, Xb_one, prox_manager)
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

    if online:
        # Save prior ensemble variance
        filen = join(workdir, 'prior_ensvar_tas_sfc_Amon')
        tmp_coords = Xb_one.var_coords['tas_sfc_Amon']
        np.savez(filen, xbv=xbv_out,
                 lat=tmp_coords['lat'].reshape(xbv.shape),
                 lon=tmp_coords['lon'].reshape(xbv.shape))

    # 3 July 2015: compute and save the GMT for the full ensemble
    gmt_ensemble = np.zeros([ntimes, nens])
    nhmt_ensemble = np.zeros([ntimes, nens])
    shmt_ensemble = np.zeros([ntimes, nens])
    for iyr, yr in enumerate(assim_times):
        filen = join(workdir, 'year{:04d}'.format(int(yr)))
        Xb_one.state = np.load(filen+'.npy')
        xam = Xb_one.get_var_data('tas_sfc_Amon').mean(axis=1)
        gmt, nhmt, shmt = \
            global_mean2(xam,
                         Xb_one.var_coords['tas_sfc_Amon']['lat'],
                         output_hemispheric=True)
        gmt_ensemble[iyr] = gmt
        nhmt_ensemble[iyr] = nhmt
        shmt_ensemble[iyr] = shmt

    filen = join(workdir, 'gmt_ensemble')
    np.savez(filen, gmt_ensemble=gmt_ensemble, nhmt_ensemble=nhmt_ensemble,
             shmt_ensemble=shmt_ensemble, recon_times=recon_times)

    # save global mean temperature history and the proxies assimilated
    print(('saving global mean temperature update history and ',
           'assimilated proxies...'))
    filen = join(workdir, 'gmt')
    np.savez(filen, gmt_save=gmt_save, nhmt_save=nhmt_save, shmt_save=shmt_save,
             recon_times=recon_times,
             apcount=assim_proxy_count,
             tpcount=assim_proxy_count)

    # TODO: (AP) The assim/eval lists of lists instead of lists of 1-item dicts
    assimilated_proxies = [{p.type: [p.id, p.lat, p.lon, p.time,
                                     p.psm_obj.sensitivity]}
                           for p in prox_manager.sites_assim_proxy_objs()]
    filen = join(workdir, 'assimilated_proxies')
    np.save(filen, assimilated_proxies)
    
    # collecting info on non-assimilated proxies and save to file
    nonassimilated_proxies = [{p.type: [p.id, p.lat, p.lon, p.time,
                                        p.psm_obj.sensitivity]}
                              for p in prox_manager.sites_eval_proxy_objs()]
    if nonassimilated_proxies:
        filen = join(workdir, 'nonassimilated_proxies')
        np.save(filen, nonassimilated_proxies)

    exp_end_time = time() - begin_time
    if verbose > 0:
        print('')
        print('=====================================================')
        print('Experiment completed in ' + str(exp_end_time/60.0) + ' mins')
        print('=====================================================')

    # TODO: best method for Ye saving?
    return (prox_manager.sites_assim_proxy_objs(),
            prox_manager.sites_eval_proxy_objs())
# ------------------------------------------------------------------------------
# --------------------------- end of main code ---------------------------------
# ------------------------------------------------------------------------------


class FilterDivergenceError(ValueError):
    pass


if __name__ == '__main__':
    LMR_driver_callable()
