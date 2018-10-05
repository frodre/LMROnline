#==========================================================================================
# Data assimilation function. 
#
# Define the data assimilation function
# This version uses passed arrays, and updates the ensemble for a single time
# (& single ob).
#
#==========================================================================================

import numpy as np
from time import time

import LMR_utils as LMR_utils

def enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None):
    """
    Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington

    Revisions:

    1 September 2017: 
                    - changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1) 
                    for an unbiased calculation of the variance. 
                    (G. Hakim - U. Washington)
    
    -----------------------------------------------------------------
     Inputs:
          Xb: background ensemble estimates of state (Nx x Nens) 
     obvalue: proxy value
          Ye: background ensemble estimate of the proxy (Nens)
      ob_err: proxy error variance
         loc: localization vector (Nx x 1) [optional]
     inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,
    # ens. members]
    nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = Xb.mean(axis=1, keepdims=True)
    Xbp = Xb - xbm

    # ensemble mean and variance of the background estimate of the proxy 
    mye = Ye.mean()
    varye = Ye.var(ddof=1)

    # lowercase ye has ensemble-mean removed 
    ye = Ye - mye

    # innovation
    try:
        innov = obvalue - mye
    except ValueError as e:
        print(f'innovation error. obvalue = {obvalue} mye = {mye}')
        print('returning Xb unchanged...')
        return Xb
    
    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp, ye) / (nens - 1)

    # Option to localize the gain
    if loc is not None:
        kcov = kcov * loc
   
    # Kalman gain
    kmat = kcov / kdenom

    # update ensemble mean
    xam = xbm + kmat * innov

    # update the ensemble members using the square-root approach
    beta = 1 / (1 + np.sqrt(ob_err / (varye + ob_err)))
    kmat *= beta
    kmat = kmat[:, np.newaxis]  # Nx x 1
    ye = ye[np.newaxis]         # 1 x Nens
    Xap = Xbp - np.dot(kmat, ye)

    # full state
    Xa = xam + Xap

    # Return the full state
    return Xa


def enkf_update_array_xb_blend(Xb, obvalue, Ye, ob_err, loc=None, inflate=None,
                               static_prior=None, a=1):
    """
    Temporary second function to ensure that nothing changes when updating
    the syntax... AndreP

    Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
          Xb: background ensemble estimates of state (Nx x Nens)
     obvalue: proxy value
          Ye: background ensemble estimate of the proxy (Nens x 1)
      ob_err: proxy error variance
         loc: localization vector (Nx x 1) [optional]
     inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array
    #  Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = Xb.mean(axis=1)
    Xbp = Xb - xbm[:, None]  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy
    mye = Ye.mean()

    # lowercase ye has ensemble-mean removed
    ye = Ye - mye

    # innovation  (Why is this in a try except?)
    try:
        innov = obvalue - mye
    except:
        print(('innovation error. obvalue = ' + str(obvalue) + ' mye = ' +
               str(mye)))
        print('returning Xb unchanged...')
        return Xb

    # numerator of serial Kalman gain (cov(x,Hx))
    if static_prior is not None:
        # Hybrid prior update method
        Xb_static, Ye_static = static_prior
        xbm_static = Xb_static.mean(axis=1)
        Xbp_static = Xb_static - xbm_static[:, None]
        ye_static = Ye_static - Ye_static.mean()

        kcov_f = np.dot(Xbp, ye) / (Nens - 1)
        kcov_s = np.dot(Xbp_static, ye_static) / (Nens - 1)

        kcov = a * kcov_f + (1-a) * kcov_s
        varye = a * Ye.var(ddof=1) + (1-a) * Ye_static.var(ddof=1)
    else:
        # Standard update method
        kcov = np.dot(Xbp, ye) / (Nens-1)
        varye = Ye.var(ddof=1)  # TODO: this should probably switch to unbiased

    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # Option to inflate the covariances by a certain factor
    if inflate is not None:
        kcov = inflate * kcov

    # Option to localize the gain
    if loc is not None:
        kcov = kcov * loc

    # Kalman gain
    kmat = kcov / kdenom

    # update ensemble mean
    mean_update = kmat * innov
    xam = xbm + mean_update

    if static_prior is not None:
        xam_static = xbm_static + mean_update

    # update the ensemble members using the square-root approach
    beta = 1. / (1. + np.sqrt(ob_err / (varye + ob_err)))
    kmat *= beta
    kmat = kmat[:, np.newaxis]

    if static_prior is not None:
        ye_static = ye_static[None]
        Xap_static = Xbp_static - np.dot(kmat, ye_static)
        Xb_static[:] = Xap_static + xam_static[:, None]

    ye = ye[np.newaxis]
    Xap = Xbp - np.dot(kmat, ye)

    # full state
    Xa = xam[:, None] + Xap

    # Return the full state
    return Xa


def kalman_optimal(Xb, proxy_obs, proxy_errors, ye_ens, num_svals=None,
                   verbose=False):
    """
    Originator
    ==========
    Greg Hakim
    University of Washington
    26 February 2018

    -- Adapted by AndreP September 2018

    Parameters
    ----------
    Xb: ndarray
        State data to be updated by assimilation (Nx x Nens)
    proxy_obs: ndarray
        Proxy obervations used to update the state (Nobs)
    proxy_errors: ndarray
        Proxy error values for each proxy (Nobs)
    ye_ens: ndarray
        Estimated observations from the state for each proxy (Nobs x Nens)
    num_svals: int, Optional
        Number of singular values to use in the transformed update space
    verbose: bool, Optional
        Print verbose information about the assimilation update

    Returns
    -------
    ndarray
        The updated state data

    """

    if verbose:
        print('Updating state using Kalman Optimal solver all at once.')

    begin_time = time()

    nobs = ye_ens.shape[0]
    nens = ye_ens.shape[1]
    num_dof = min(nobs, nens)

    if verbose:
        print(f'number of obs: {nobs:d}')
        print(f'number of ensemble members: {nens:d}')

    # ensemble prior mean and perturbations
    xbm = Xb.mean(axis=1, keepdims=True)
    Xbp = Xb - xbm

    R = np.diag(proxy_errors)
    Risr = np.diag(1 / np.sqrt(proxy_errors))

    Yem = ye_ens.mean(axis=1, keepdims=True)
    Yep = ye_ens - Yem
    Htp = np.dot(Risr, Yep) / np.sqrt(nens - 1)
    Htm = np.dot(Risr, Yem)
    Yt = np.dot(Risr, proxy_obs[:, None])

    U, s, VT = np.linalg.svd(Htp, full_matrices=True)

    if not num_svals:
        num_svals = len(s) - 1

    if verbose:
        print(f'ndof : {num_dof}')
        print(f'U : {U.shape}')
        print(f's : {s.shape}')
        print(f'V : {VT.shape}')
        print(f'recontructing using {num_svals} singular values')

    innov = np.dot(U.T, Yt - Htm)

    # Kalman gain
    Kpre = s[0:num_svals] / (s[0:num_svals] * s[0:num_svals] + 1)
    K = np.zeros([nens, nobs])
    np.fill_diagonal(K, Kpre)

    # ensemble-mean analysis increment in transformed space
    xhatinc = np.dot(K, innov)
    # ensemble-mean analysis increment in the transformed ensemble space
    xtinc = np.dot(VT.T, xhatinc) / np.sqrt(nens - 1)

    # ensemble-mean analysis increment in the original space
    xinc = np.dot(Xbp, xtinc)
    # ensemble mean analysis in the original space
    xam = xbm + xinc

    # transform the ensemble perturbations
    lam = np.zeros([nobs, nens])
    np.fill_diagonal(lam, s[0:num_svals])
    tmp = np.linalg.inv(np.dot(lam, lam.T) + np.identity(nobs))
    sigsq = np.identity(nens) - np.dot(np.dot(lam.T, tmp), lam)
    sig = np.sqrt(sigsq)
    T = np.dot(VT.T, sig)
    Xap = np.dot(Xbp, T)

    # perturbations must have zero mean
    Xap = Xap - Xap.mean(axis=1, keepdims=True)

    if verbose:
        print(f'min s: {min(s)}')

    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print(f'completed in {elapsed_time:2.2f} seconds')
        print('-----------------------------------------------------')

    readme = '''
        The SVD dictionary contains the SVD matrices U,s,V where V 
        is the transpose of what numpy returns. xtinc is the ensemble-mean
        analysis increment in the intermediate space; *any* state variable 
        can be reconstructed from this matrix.
        '''
    SVD = {'U': U, 's': s, 'V': VT.T, 'xtinc': xtinc, 'readme': readme}

    Xa = xam + Xap

    return Xa


def kalman_ensrf_serial():
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
        mse = np.mean((Y.values[t] - Ye) ** 2)
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


# ==============================================================================
# DA Utility Functions
# ==============================================================================


def get_valid_proxies_info(target_year, proxy_manager, verbose=False):

    begin_time = time()

    if verbose:
        print(f'Finding proxy records for year: {target_year:4d}')

    proxy_vals = []
    proxy_errs = []
    proxy_idxs = []

    for pidx, proxy_obj in enumerate(proxy_manager.sites_assim_proxy_objs()):

        try:
            proxy_val = proxy_obj.values[target_year]
        except KeyError:
            if verbose:
                print(f'No obs in target year {target_year:4d} for proxy '
                      f'{proxy_obj.type}.')
            continue

        proxy_vals.append(proxy_val)
        proxy_errs.append(proxy_obj.psm_obj.error())
        proxy_idxs.append(pidx)

    proxy_vals = np.array(proxy_vals)
    proxy_errs = np.array(proxy_errs)

    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print(f'completed in {elapsed_time:2.2f} seconds')
        print('-----------------------------------------------------')

    return proxy_vals, proxy_errs, proxy_idxs


def process_hybrid_static_prior(yr_idx, prior_state, blend_prior, hybrid_a_val):
    if yr_idx == 0:
        # Creates a copy for use as our static prior
        prior_state.stash_state('orig_aug')
        Xb_static = prior_state.state
        Ye_vals_static = prior_state.get_var_data('ye_vals')
        prior_state.stash_recall_state_list('orig_aug',
                                            copy=True)
    else:
        prior_state.stash_state('tmp')
        prior_state.stash_recall_state_list('orig_aug', copy=True)
        Xb_static = prior_state.state
        Ye_vals_static = prior_state.get_var_data('ye_vals')
        prior_state.stash_pop_state_list('tmp')

    if blend_prior:
        xbf = prior_state.state
        blend_forecast = (hybrid_a_val * xbf +
                          (1 - hybrid_a_val) * Xb_static)
        prior_state.state = blend_forecast

    hybrid_update_kwargs = {'Xb_static': Xb_static,
                            'Ye_vals_static': Ye_vals_static}

    return hybrid_update_kwargs, prior_state


def cov_localization(locRad, Y, X, X_coords):
    """

    Originator: R. Tardif, 
                Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
        locRad : Localization radius (distance in km beyond which cov are forced to zero)
             Y : Proxy object, needed to get ob site lat/lon (to calculate distances w.r.t. grid pts
             X : Prior object, needed to get state vector info. 
      X_coords : Array containing geographic location information of state vector elements

     Output:
        covLoc : Localization vector (weights) applied to ensemble covariance estimates.
                 Dims = (Nx x 1), with Nx the dimension of the state vector.

     Note: Uses the Gaspari-Cohn localization function.

    """

    # declare the localization array, filled with ones to start with (as in no localization)
    stateVectDim, nbdimcoord = X_coords.shape
    covLoc = np.ones(shape=[stateVectDim],dtype=np.float64)

    # Mask to identify elements of state vector that are "localizeable"
    # i.e. fields with (lat,lon)
    localizeable = covLoc == 1. # Initialize as True
    
    for var in X.trunc_state_info.keys():
        [var_state_pos_begin,var_state_pos_end] =  X.trunc_state_info[var]['pos']
        # if variable is not a field with lats & lons, tag localizeable as False
        if X.trunc_state_info[var]['spacecoords'] != ('lat', 'lon'):
            localizeable[var_state_pos_begin:var_state_pos_end+1] = False

    # array of distances between state vector elements & proxy site
    # initialized as zeros: this is important!
    dists = np.zeros(shape=[stateVectDim])

    # geographic location of proxy site
    site_lat = Y.lat
    site_lon = Y.lon
    # geographic locations of elements of state vector
    X_lon = X_coords[:,1]
    X_lat = X_coords[:,0]

    # calculate distances for elements tagged as "localizeable".
    dists[localizeable] = np.array(LMR_utils.haversine(site_lon, site_lat,
                                                       X_lon[localizeable],
                                                       X_lat[localizeable]),dtype=np.float64)

    # those not "localizeable" are assigned with a disdtance of "nan"
    # so these elements will not be included in the indexing
    # according to distances (see below)
    dists[~localizeable] = np.nan
    
    # Some transformation to variables used in calculating localization weights
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    # values for distances very near the localization radius
    # TODO: revisit calculations to minimize round-off errors
    covLoc[covLoc < 0.0] = 0.0


    return covLoc
