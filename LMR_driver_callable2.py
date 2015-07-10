
# ==============================================================================
# Program: LMR_driver_callable.py
# 
# Purpose: 
#
# Options: None. 
#          Experiment parameters through namelist, passed through object called
#          "state"
# 
# Originators: Greg Hakim   | Dept. of Atmospheric Sciences, Univ. of Washington
#              Robert Tardif | January 2015
# 
# Revisions: 
#  April 2015:
#            - This version is callable by an outside script, accepts a single
#              object, called state, which has everything needed for the driver
#              (G. Hakim)

#            - Re-organisation of code around PSM calibration and calculation of
#              Code now assumes PSM parameters have been pre-calulated and
#              Ye's are calculated up-front for all proxy types/sites. All
#              proxy data are now also loaded up-front, prior to any loops.
#              Ye's are appended to state vector to form an augmented state
#              vector and are also updated by DA. (R. Tardif)
#  May 2015:
#            - Bug fix in calculation of global mean temperature + function
#              now part of LMR_utils.py (G. Hakim)
#  July 2015:
#            - Switched time & proxy loops, simplified logic so more of the
#              proxy and PSM specifics are contained within their classes,
#              formatted to mostly adhere to PEP8 guidlines
#              (A. Perkins)
# ==============================================================================

import numpy as np
from os.path import join
from time import time

import LMR_proxy2
import LMR_prior
import LMR_utils
import LMR_config as BaseCfg
from LMR_DA import enkf_update_array, cov_localization


def LMR_driver_callable(cfg=None):

    if cfg is None:
        cfg = BaseCfg  # Use base configuration from LMR_config

    # Temporary fix for old 'state usage'
    core = cfg.core
    prior = cfg.prior

    # verbose controls print comments (0 = none; 1 = most important;
    #  2 = many; >=3 = all)
    verbose = 1

    # TODO: AP Fix Configuration
    # daylight the variables passed in the state object (easier for code
    # migration than leaving attached)
    nexp = core.nexp
    workdir = core.datadir_output
    recon_period = core.recon_period
    online = core.online_reconstruction
    nens = core.nens
    loc_rad = core.loc_rad
    prior_source = prior.prior_source
    datadir_prior = prior.datadir_prior
    datafile_prior = prior.datafile_prior
    state_variables = prior.state_variables

    # ==========================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ==========================================================================
    # TODO: AP Logging instead of print statements
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Running LMR reconstruction...'
        print '====================================================='
        print 'Name of experiment: ', nexp
        print ' Monte Carlo iter : ', 
        print ''
        
    begin_time = time()

    # Define the number of years of the reconstruction (nb of assimilation
    # times)
    # Note: recon_period is defined in namelist
    recon_times = np.arange(recon_period[0], recon_period[1]+1)

    # ==========================================================================
    # Load calibration data ----------------------------------------------------
    # # ========================================================================
    # if verbose > 0:
    #     print '------------------------------'
    #     print 'Creating calibration object...'
    #     print '------------------------------'
    #     print 'Source for calibration: ' + datatag_calib
    #     print ''

    # TODO: Doesn't appear to use C at all...
    # Assign calibration object according to "datatag_calib" (from namelist)
    # C = LMR_calibrate.calibration_assignment(datatag_calib)
    #
    # TODO: AP Required attributes need explicit declaration in method/class
    # # the path to the calibration directory is specified in the namelist file;
    #  bind it here
    # C.datadir_calib = datadir_calib;
    #
    # # read the data !!!!!!!!!!!!!!!!!! don't need this with all pre-calculated
    #  PSMs !!!!!!!!!!!!!!!!!!
    # C.read_calibration()

    # ==========================================================================
    # Load prior data ----------------------------------------------------------
    # ==========================================================================
    if verbose > 0:
        print '-------------------------------------------'
        print 'Uploading gridded (model) data as prior ...'
        print '-------------------------------------------'
        print 'Source for prior: ', prior_source

    # Assign prior object according to "prior_source" (from namelist)
    X = LMR_prior.prior_assignment(prior_source)

    # TODO: AP explicit requirements
    # add namelist attributes to the prior object
    X.prior_datadir = datadir_prior
    X.prior_datafile = datafile_prior
    X.statevars = state_variables
    X.Nens = nens

    # Read data file & populate initial prior ensemble
    X.populate_ensemble(prior_source)
    Xb_one_full = X.ens

    # number of lats and lons 
    nlat = X.nlat
    nlon = X.nlon

    # Prepare to check for files in the prior (work) directory (this object just
    #  points to a directory)
    prior_check = np.DataSource(workdir)

    load_time = time() - begin_time
    if verbose > 2:
        print '-----------------------------------------------------'
        print 'Loading completed in ' + str(load_time)+' seconds'
        print '-----------------------------------------------------'

    # ==========================================================================
    # Get information on proxies to assimilate ---------------------------------
    # ==========================================================================

    begin_time_proxy_load = time()
    if verbose > 0:
        print ''
        print '-----------------------------------'
        print 'Uploading proxy data & PSM info ...'
        print '-----------------------------------'

    # Build dictionaries of proxy sites to assimilate and those set aside for
    # verification
    prox_manager = LMR_proxy2.ProxyManager(BaseCfg, recon_period)
    type_site_assim = prox_manager.all_ids_by_group

    if verbose > 0:
        print 'Assimilating proxy types/sites:', type_site_assim

    # ==========================================================================
    # Calculate all Ye's (for all sites in sites_assim) ------------------------
    # ==========================================================================

    print '--------------------------------------------------------------------'
    print 'Proxy counts for experiment:'
    # count the total number of proxies
    total_proxy_count = len(prox_manager.all_proxies)
    for pkey, plist in type_site_assim.iteritems():
        print('%45s : %5d' % (pkey, len(plist)))
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print '--------------------------------------------------------------------'

    proxy_load_time = time() - begin_time_proxy_load
    if verbose > 2:
        print '-----------------------------------------------------'
        print 'Loading completed in ' + str(proxy_load_time) + ' seconds'
        print '-----------------------------------------------------'

    # ==========================================================================
    # Calculate truncated state from prior, if option chosen -------------------
    # ==========================================================================

    [Xb_one, lat_new, lon_new] = LMR_utils.regrid_sphere(nlat, nlon, nens,
                                                         Xb_one_full, 42)
    nlat_new = lat_new.shape[0]
    nlon_new = lat_new.shape[1]

    # Keep dimension of pre-augmented version of state vector
    [stateDim, _] = Xb_one.shape

    # ----------------------------------
    # Augment state vector with the Ye's
    # ----------------------------------

    # Extract all the Ye's from master list of proxy objects into numpy array
    if not online:
        Ye_all = np.empty(shape=[total_proxy_count, nens])
        for k, proxy in enumerate(prox_manager.sites_assim_proxy_objs()):
            Ye_all[k, :] = proxy.psm(X)

        # Append ensemble of Ye's to prior state vector
        Xb_one_aug = np.append(Xb_one, Ye_all, axis=0)
    else:
        Xb_one_aug = Xb_one

    # Dump prior state vector (Xb_one) to file 
    filen = workdir + '/' + 'Xb_one'
    np.savez(filen, Xb_one=Xb_one, Xb_one_aug=Xb_one_aug, stateDim=stateDim,
             lat=lat_new, lon=lon_new, nlat=nlat_new, nlon=nlon_new)

    # ==========================================================================
    # Loop over all proxies and perform assimilation ---------------------------
    # ==========================================================================

    # ---------------------
    # Loop over proxy types
    # ---------------------

    # Array containing the global-mean state (for diagnostic purposes)
    gmt_save = np.zeros([total_proxy_count+1,
                         recon_period[1] - recon_period[0] + 1])
    xbm = Xb_one[0:stateDim, :].mean(axis=1)  # ensemble-mean
    xbm_lalo = xbm.reshape(nlat_new, nlon_new)
    gmt = LMR_utils.global_mean(xbm_lalo, lat_new[:, 0], lon_new[0, :])
    gmt_save[0, :] = gmt  # First prior
    gmt_save[1, :] = gmt  # Prior for first proxy assimilated

    lasttime = time()
    for yr_idx, t in enumerate(xrange(recon_period[0], recon_period[1]+1)):

        if verbose > 2:
            print 'working on year: ' + str(t)

        ypad = '{:04d}'.format(t)
        filen = join(workdir, 'year' + ypad + '.npy')
        if prior_check.exists(filen):
            if verbose > 2:
                print 'prior file exists: ' + filen
            Xb = np.load(filen)
        else:
            if verbose > 2:
                print 'Prior file ', filen, ' does not exist...'
            Xb = Xb_one_aug.copy()

        for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
            # Crude check if we have ob for this time
            try:
                Y.values[t]
            except KeyError:
                continue

            if verbose > 0:
                print '--------------- Processing proxy: ' + Y.id

            if verbose > 1:
                print ''
                print 'Site:', Y.id, ':', Y.type
                print ' latitude, longitude: ' + str(Y.lat), str(Y.lon)

            loc = None
            if loc_rad is not None:
                if verbose > 2:
                    print '...computing localization...'
                    loc = cov_localization(loc_rad, X, Y)

            # Get Ye values for current proxy
            if online:
                Ye = Y.psm(Xb)
            else:
                Ye = Xb[proxy_idx+stateDim]

            # Define the ob error variance
            ob_err = Y.psm_obj.R

            # ------------------------------------------------------------------
            # Do the update (assimilation) -------------------------------------
            # ------------------------------------------------------------------
            if verbose > 2:
                print ('updating time: ' + str(t) + ' proxy value : ' +
                       str(Y.values[t]) + ' | mean prior proxy estimate: ' +
                       str(Ye.mean()))

            # Update the state
            Xa = enkf_update_array(Xb, Y.values[t], Ye, ob_err, loc)
            Xb = Xa
            xam = Xa[0:stateDim].mean(axis=1)
            xam_lalo = xam.reshape((nlat_new, nlon_new))
            gmt = LMR_utils.global_mean(xam_lalo, lat_new[:, 0], lon_new[0, :])
            gmt_save[proxy_idx, yr_idx] = gmt

            # check the variance change for sign
            thistime = time()
            if verbose > 2:
                xbvar = Xb.var(axis=1, ddof=1)
                xavar = Xa.var(ddof=1, axis=1)
                vardiff = xavar - xbvar
                print 'max change in variance:' + str(np.max(vardiff))
                print 'update took ' + str(thistime-lasttime) + 'seconds'
            lasttime = thistime

            if proxy_idx+1 < total_proxy_count:
                gmt_save[proxy_idx+1, yr_idx] = gmt

        # Dump Xa to file (to be used as prior for next assimilation)
        np.save(filen, Xa)

    end_time = time() - begin_time

    # End of loop on proxy types
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Reconstruction completed in ' + str(end_time/60.0)+' mins'
        print '====================================================='

    # save global mean temperature history and the proxies assimilated
    print ('saving global mean temperature update history and ',
           'assimilated proxies...')
    filen = join(workdir, 'gmt')
    np.savez(filen, gmt_save=gmt_save, recon_times=recon_times,
             apcount=total_proxy_count, tpcount=total_proxy_count)

    assimilated_proxies = [{p.type: [p.site, p.lat, p.lon, p.time]}
                           for p in prox_manager.sites_assim_proxy_objs()]
    filen = join(workdir, 'assimilated_proxies')
    np.save(filen, assimilated_proxies)
    
    # collecting info on non-assimilated proxies and save to file
    nonassimilated_proxies = [{p.type: [p.site, p.lat, p.lon, p.time]}
                              for p in prox_manager.sites_eval_proxy_objs()]
    if nonassimilated_proxies:
        filen = join(workdir, 'nonassimilated_proxies')
        np.save(filen, nonassimilated_proxies)

    exp_end_time = time() - begin_time
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Experiment completed in ' + str(exp_end_time/60.0) + ' mins'
        print '====================================================='

# ------------------------------------------------------------------------------
# --------------------------- end of main code ---------------------------------
# ------------------------------------------------------------------------------