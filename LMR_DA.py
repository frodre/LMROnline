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

def enkf_update_array(Xb, obvalue, Ye, ob_err, kalman_gain=None, loc=None):
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
    October 2018
      - Generalized the Kalman gain calculation so I could use a single update
        function for hybrid and regular EnSRF updates. (A. Perkins)

    Parameters
    ----------
    Xb: ndarray
        background ensemble estimates of state (Nx x Nens)
    obvalue: float
        proxy value
    Ye: ndarray
        background ensemble estimate of the proxy (Nens)
    ob_err: float
        proxy error variance
    kalman_gain: ndarray, Optional
        The kalman gain to be used for the update step if a non-standard gain
        is desired
    loc: float, Optional
        The localization radius to enforce for the Kalman gain term

    Returns
    -------
    ndarray:
        The updated ensemble of state (Nx x Nens)
    """

    # ensemble mean background and perturbations
    xbm = Xb.mean(axis=1, keepdims=True)
    Xbp = Xb - xbm

    # ensemble mean and variance of the background estimate of the proxy 
    mye = Ye.mean()
    varye = Ye.var(ddof=1)

    # lowercase ye has ensemble-mean removed 
    ye = Ye - mye

    # innovation
    innov = obvalue - mye

    if kalman_gain is None:
        kmat = get_serial_kalman_gain(Xbp, ye, ob_err)
    else:
        kmat = kalman_gain

    # Option to localize the gain
    if loc is not None:
        kmat = kmat * loc

    # update ensemble mean
    xam = xbm + kmat * innov

    # update the ensemble members using the square-root approach
    beta = 1 / (1 + np.sqrt(ob_err / (varye + ob_err)))
    kmat *= beta
    ye = ye[np.newaxis]         # 1 x Nens
    Xap = Xbp - np.dot(kmat, ye)

    # full state
    Xa = xam + Xap

    # Return the full state
    return Xa


def kalman_optimal(Xb, proxy_obs, proxy_errors, valid_proxy_idxs, ye_start_idx,
                   num_svals=None, verbose=False):
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
        background ensemble to be updated during assimilation
    proxy_obs: ndarray
        Proxy obervations used to update the state (Nobs)
    proxy_errors: ndarray
        Proxy error values for each proxy (Nobs)
    valid_proxy_idxs: ndarray
        Indices corresponding to the proxy observations
    ye_start_idx: int
        Index where Ye values start in the state array
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

    ye_vals = Xb[ye_start_idx:, :]
    ye_ens = ye_vals[valid_proxy_idxs, :]

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


def kalman_ensrf_serial(Xb, proxy_obs, proxy_errors, valid_proxy_idxs,
                        ye_start_idx, verbose=False, loc_rad=None,
                        Xb_static=None, hybrid_a_val=None):
    """
    Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington

    -- Adapted by Andre P

    Xb: ndarray
        State data to be updated by assimilation (Nx x Nens)
    proxy_obs: ndarray
        Proxy obervations used to update the state (Nobs)
    proxy_errors: ndarray
        Proxy error values for each proxy (Nobs)
    ye_ens: ndarray
        Estimated observations from the state for each proxy (Nobs x Nens)
    verbose: bool, Optional
        Print verbose update information
    loc_rad: float, Optional
        Localization radius to use during the assimilation update. NOT
        CURRENTLY IMPLEMENTED
    Xb_static: ndarray, Optional
        Climatological data (not forecasted) used for hybrid forecast update
        as in Perkins & Hakim 2017
    hybrid_a_val: float, Optional
        Hybrid blending coefficient between 0.0 and 1.0. Only used if Xb_static
        is not None

    Returns
    -------
    ndarray
        The updated state data
    """

    if verbose:
        print('Updating state using serial Kalman EnSRF solver')

    begin_time = time()

    if loc_rad is not None:
        raise NotImplementedError('Covariance localization has not yet been '
                                  'implemented in this version.')

    for i in range(len(proxy_obs)):

        y_val = proxy_obs[i]
        ob_err = proxy_errors[i]
        ye_idx = ye_start_idx + valid_proxy_idxs[i]
        ye_val = Xb[ye_idx]

        if Xb_static is not None:
            ye_static = Xb_static[ye_idx]

            kalman_gain = get_serial_kalman_gain_hybrid_update(Xb, Xb_static,
                                                               ye_val,
                                                               ye_static,
                                                               ob_err,
                                                               hybrid_a_val)
        else:
            # Let default kalman gain in enkf_update do the work
            kalman_gain = None

        Xa = enkf_update_array(Xb, y_val, ye_val, ob_err,
                               kalman_gain=kalman_gain)

        Xb = Xa

    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print(f'completed in {elapsed_time:2.2f} seconds')
        print('-----------------------------------------------------')

    return Xa


# ==============================================================================
# DA Utility Functions
#
#
# ==============================================================================


def get_solver(solver_key):
    if solver_key == 'serial':
        return kalman_ensrf_serial
    elif solver_key == 'optimal':
        return kalman_optimal
    else:
        raise KeyError('Unrecognized solver specification for data '
                       'assimilation: {}'.format(solver_key))


def get_serial_kalman_gain(Xb, ye, ob_error):
    """
    Determine the kalman gain term for the serial EnSRF update.
    Parameters
    ----------
    Xb: ndarray
        State array composed of field values being updated. (Nx x Nens)
    ye: ndarray
        Ensemble of estimated observations. (Nens)
    ob_error: float
        Error of the observation being used to compute the innovation

    Returns
    -------
    ndarray
        Kalman gain matrix computed from the state and ye_vals

    """

    Xbp = Xb - Xb.mean(axis=1, keepdims=True)
    yep = ye - ye.mean()

    nens = Xbp.shape[1]
    var_ye = yep.var(ddof=1)

    # innovation variance (denominator of serial Kalman gain)
    kdenom = (var_ye + ob_error)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp, yep) / (nens - 1)

    # Kalman gain
    kmat = kcov / kdenom

    kmat = kmat[:, None]

    return kmat


def get_serial_kalman_gain_hybrid_update(Xb, Xb_static, ye, ye_static,
                                         ob_error, a):
    """
    Determine the kalman gain term for the serial EnSRF update using the
    hybrid blending between forecast and static climatological data.

    Parameters
    ----------
    Xb: ndarray
        State array composed of field values being updated (Nx x Nens)
    Xb_static: ndarray
        Same as Xb_pert, but the static climatological state
    ye: ndarray
        Ensemble of estimated observations.
        (Nens)
    ye_static: ndarray
        Same as ye_pert but ye values from the static climatological state.
    ob_error: float
        Error of the observation being used to compute the innovation
    a: float
        Blending coefficient for the hybrid update between 0.0 and 1.0.

    Returns
    -------
    ndarray
        Kalman gain matrix computed from the state and ye_vals

    """

    nens = Xb.shape[1]

    Xbp = Xb - Xb.mean(axis=1, keepdims=True)
    yep = ye - ye.mean()

    Xbp_static = Xb_static - Xb_static.mean(axis=1, keepdims=True)
    yep_static = ye_static - ye_static.mean()

    kcov_f = np.dot(Xbp, yep) / (nens - 1)
    kcov_s = np.dot(Xbp_static, yep_static) / (nens - 1)

    ye_var = yep.var(ddof=1)
    ye_static_var = yep_static.var(ddof=1)

    var_ye_blend = a * ye_var + (1 - a) * ye_static_var

    kcov = a * kcov_f + (1 - a) * kcov_s

    kmat = kcov / (var_ye_blend + ob_error)

    kmat = kmat[:, None]

    return kmat


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
