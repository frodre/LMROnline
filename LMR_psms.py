"""
Module: LMR_psms.py

Purpose: Module containing methods for various Proxy System Models (PSMs)
         Adapted from LMR_proxy and LMR_calibrate using OOP by Andre Perkins

Originator: Andre Perkins, U. of Washington

Revisions: 
          - Use of new more efficient "get_distance" function to calculate the
            distance between proxy sites and analysis grid points.
            [R. Tardif, U. of Washington, February 2016]
          - Added the "LinearPSM_TorP" class, allowing the use of
            temperature-calibrated *OR* precipitation-calibrated linear PSMs.
            For each proxy record to be assimilated, the selection of the PSM
            (i.e. T vs P) is perfomed on the basis of the smallest MSE of
            regression residuals.
            [R. Tardif, U. of Washington, April 2016]
          - Added the "h_interpPSM" psm class for use of isotope-enabled GCM 
            data as prior: Ye values are taken as the prior isotope field either 
            at the nearest grid pt. or as the weighted-average of values at grid 
            points surrounding the assimilated isotope proxy site.
            [ R. Tardif, U. of Washington, June 2016 ]
          - Added the "BilinearPSM" class for PSMs based on bivariate linear 
            regressions w/ temperature AND precipitation/PSDI as independent 
            variables.
            [ R. Tardif, Univ. of Washington, June 2016 ]
          - Added the capability of calibrating/using PSMs calibrated on the basis 
            of a proxy record seasonality metadata.
            [ R. Tardif, Univ. of Washington, July 2016 ]
          - Added the capability of objectively calibrating/using PSMs calibrated 
            on the basis objectively-derived seasonality. 
            [ R. Tardif, Univ. of Washington, December 2016 ]
          - Added the "BayesRegUK37PSM" class, the forward model used in 
            the assimilation of alkenone uk37 proxy data. Code based on 
            spline coefficients provided by J. Tierney (U of Arizona).
            [ R. Tardif, Univ. of Washington, January 2017 ]
          - Calibration of statistical PSMs now all referenced to anomalies w.r.t.
            20th century.
            [ R. Tardif, Univ. of Washington, August 2017 ]
"""
import numpy as np
import logging
import LMR_gridded
import pickle
import os.path
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from sklearn.linear_model import LinearRegression
from abc import ABCMeta, abstractmethod
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from functools import lru_cache

from LMR_utils import (haversine, get_distance, smooth2D,
                       get_data_closest_gridpt, class_docs_fixer,
                       PSMFitThresholdError, PSMTooFewObsError,
                       PSMTorPCalibrationError)

# Logging output utility, configuration controlled by driver
logger = logging.getLogger(__name__)
REQ_NOBS = 25


class BasePSM(metaclass=ABCMeta):
    """
    Proxy system model.

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_kwargs: dict (unpacked)
        Specfic arguments for the target PSM
    """

    def __init__(self, config, proxy_obj, **psm_kwargs):
        self.lat = None
        self.lon = None
        pass

    @abstractmethod
    def psm(self, prior_obj):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        prior_obj BasePriorObject
            Prior to be mapped to observation space (Ye).

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """
        pass

    @abstractmethod
    def error(self):
        """
        Error model for given PSM.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_kwargs(config):
        """
        Returns keyword arguments required for instantiation of given PSM.

        Parameters
        ----------
        config: LMR_config
            Configuration module used for current LMR run.

        Returns
        -------
        kwargs: dict
            Keyword arguments for given PSM
        """
        pass

    def get_close_grid_point_data(self, data, lon, lat):
        """
        Extracts data along the sampling dimension that is closest to the
        grid point (lat/lon) of the current PSM object.

        Parameters
        ----------
        data: ndarray
            Gridded data matching dimensions of (sample, lat, lon) or
            (sample, lat*lon).
        lon: ndarray
            Longitudes pertaining to the input data.  Can have shape as a single
            vector (lat), a grid (lat, lon), or a flattened grid (lat*lon).
        lat: ndarray
            Latitudes pertaining to the input data. Can have shape as a single
            vector (lon), a grid (lat, lon), or a flattened grid (lat*lon).

        Returns
        -------
        tmp_dat: ndarray
            Grid point data closes to the lat/lon of the current PSM object
        """

        lonshp = lon.shape
        latshp = lat.shape

        # If not equal we got a single vector?
        if lonshp != latshp:
            lon, lat = np.meshgrid(lon, lat)

        if len(lon.shape) > 1:
            lon = lon.ravel()
            lat = lat.ravel()

        # Calculate distance
        dist = haversine(self.lon, self.lat, lon, lat)

        if len(dist) in data.shape:
            loc_idx = dist.argmin()
            tmp_dat = data[..., loc_idx]
        else:
            # TODO: This is not general lat/lon being swapped, OK for now...
            [min_dist_lat_idx,
             min_dist_lon_idx] = np.unravel_index(dist.argmin(),
                                                  data.shape[-2:])
            tmp_dat = data[..., min_dist_lat_idx, min_dist_lon_idx]

        return tmp_dat

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_psm_data(pre_calib_file):
        """Helper method for loading from dataframe"""

        if pre_calib_file is None:
            raise IOError('No pre-calibration file specified.')

        with open(pre_calib_file, mode='rb') as f:
            data = pickle.load(f)

        return data

    def _get_gridpoint_data_from_state(self, var_key, state_object):
        var_data = state_object.get_var_data(var_key)
        var_name, avg_interval = var_key
        var_lat = state_object.var_coords[var_name]['lat']
        var_lon = state_object.var_coords[var_name]['lon']

        return self.get_close_grid_point_data(var_data.T, var_lon, var_lat)

    def _handle_single_input_avg_key(self, avg_key, psm_config, specific_config):
        avg_kwargs = psm_config.get_avg_def(avg_key)
        elem_to_avg = avg_kwargs['elem_to_avg']
        specific_config.update_avg_interval(avg_key, avg_kwargs)

        return elem_to_avg


@class_docs_fixer
class LinearPSM(BasePSM):
    """
    PSM based on linear regression.

    Attributes
    ----------
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy sitex
    corr: float
        Correlation of proxy record against calibration data
    slope: float
        Linear regression slope of proxy/calibration fit
    intercept: float
        Linear regression y-intercept of proxy/calibration fit
    R: float
        Mean-squared error of proxy/calibration fit

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_data: dict
        Pre-calibrated PSM dictionary containing current associated
        proxy site's calibration information

    Raises
    ------
    ValueError
        If PSM is below critical correlation threshold.
    """

    def __init__(self, psm_config, proxy_obj, psm_data=None,
                 calib_obj=None, diag_out=None, diag_fig=None,
                 avg_key=None,
                 on_the_fly_calib=False):

        self.psm_key = 'linear'
        linear_psm_cfg = psm_config.linear

        ignore_pre_calib = linear_psm_cfg.ignore_pre_calib

        proxy = proxy_obj.type
        site = proxy_obj.id
        r_crit = linear_psm_cfg.psm_r_crit
        self.lat = proxy_obj.lat
        self.lon = proxy_obj.lon
        self.elev = proxy_obj.elev

        # Variable is used temporarily
        self.nobs = None

        self.datatag = linear_psm_cfg.datatag
        self.avg_type = linear_psm_cfg.avg_type

        self.datainfo = linear_psm_cfg.datainfo
        # Variable mapping connected to state information set in the
        # configuration
        # TODO: switch to a tuple format in the config
        self.psm_vartype = self.datainfo['psm_vartype']

        # all required types for the PSM
        self.psm_req_types = (self.psm_vartype,)

        self.sensitivity = list(self.psm_vartype.keys())[0]

        # PSM Model Info
        self.SSE = None
        self.AIC = None
        self.BIC = None
        self.R2 = None
        self.R2adj = None

        # For diagnostics
        self.calib_time = None
        self.calib_refer_values = None
        self.calib_proxy_values = None
        self.calib_proxy_fit = None

        try:
            if ignore_pre_calib:
                raise IOError('Ignore pre-calibrated PSMs is specified.')

            # Try using pre-calibrated psm_data
            if psm_data is None:
                pre_calib_fpath = linear_psm_cfg.pre_calib_datafile
                psm_data = self._load_psm_data(pre_calib_fpath)

            psm_site_data = psm_data[(proxy, site)]
            self.corr = psm_site_data['PSMcorrel']
            self.slope = psm_site_data['PSMslope']
            self.intercept = psm_site_data['PSMintercept']
            self.R = psm_site_data['PSMmse']
            self.avg_interval = psm_site_data['avg_interval']
            # TODO: Set seasonality for calibration on the fly?
            self.seasonality = psm_site_data.get('Seasonality', None)
            psm_config.handle_proxy_elem_list(self.seasonality)

        except KeyError:
            raise KeyError('Proxy in database but not found in '
                             'pre-calibration file... Skipping: '
                             '{}'.format(proxy_obj.id))
        except IOError as e:
            print('No pre-calibration found for {}'.format(proxy_obj.id))
            if not on_the_fly_calib:
                raise e
            else:
                print(' ... calibrating on the fly')

                # Get the averaging interval information from the configuration
                if self.avg_type == 'annual':
                    avg_key = 'annual_std'
                elif self.avg_type == 'seasonal':
                    if avg_key is None:
                        elem_to_avg = proxy_obj.seasonality
                        [avg_key, _] = psm_config.handle_proxy_elem_list(
                            elem_to_avg)
                else:
                    raise KeyError(
                        'Unrecognized average type in PSM initialization...'
                        '\nExpected "annual" or "seasonal". Got {}'
                        ''.format(self.avg_type))

                # Handle seasonality
                self.avg_interval = avg_key
                self.seasonality = self._handle_single_input_avg_key(avg_key,
                                                                     psm_config,
                                                                     linear_psm_cfg)

                if calib_obj is None:
                    source = linear_psm_cfg.datatag
                    calib_class = LMR_gridded.get_analysis_var_class(source)
                    calib_obj = calib_class.load(linear_psm_cfg, anomaly=True)

                self.calibrate(calib_obj, proxy_obj,
                               diag_output=diag_out, diag_output_figs=diag_fig)

        vartype_as_tuple = tuple(self.psm_vartype.items())[0]
        self.req_avg_intervals = {vartype_as_tuple: self.avg_interval}

        # Raise exception if critical correlation value not met
        if abs(self.corr) < r_crit:
            raise PSMFitThresholdError('Proxy model correlation ({:.2f}) does '
                                       'not meet critical threshold ({:.2f}).'
                                       ''.format(self.corr, r_crit))

    def psm(self, state_object):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        state_object: LMR_gridded.State
            State vector to be mapped into observation space (stateDim x ensDim)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Get state var for psm

        state_var = state_object.get_psm_var_key(self.psm_vartype)
        var_avg_key = (state_var, self.avg_interval)
        gridpoint_data = self._get_gridpoint_data_from_state(var_avg_key,
                                                             state_object)

        Ye = self.basic_psm(gridpoint_data)

        return Ye

    def basic_psm(self, data):
        """
        A PSM that doesn't need to do the state unpacking steps...

        Parameters
        ----------
        data: ndarray
            Data to be used in the psm calculation of estimated observations
            (Ye)

        Returns
        -------
        Ye: ndarray
            Estimated observations from the proxy system model
        """

        return self.slope * data + self.intercept

    def error(self):
        """
        Error model for the proxy.
        Returns
        -------

        """
        return self.R

    # TODO: Simplify a lot of the actions in the calibration
    def calibrate(self, calib_obj, proxy, diag_output=False, diag_output_figs=False):
        """
        Calibrate given proxy record against observation data and set relevant
        PSM attributes.

        Parameters
        ----------
        calib_obj: calibration_master like
            Calibration object containing data, time, lat, lon info
        proxy: BaseProxyObject like
            Proxy object to fit to the calibration data
        diag_output, diag_output_figs: bool, optional
            Diagnostic output flags for calibration method
        """

        # --------------------------------------------
        # Use linear model (regression) as default PSM
        # --------------------------------------------

        calvals = self.get_close_grid_point_data(calib_obj.data,
                                                 calib_obj.lon,
                                                 calib_obj.lat)
        cyears = np.array([t.year for t in calib_obj.time])

        valid_idx = np.isfinite(calvals)
        calvals = calvals[valid_idx]
        cyears = cyears[valid_idx]

        # ------------------------
        # Set-up linear regression
        # ------------------------
        # Use pandas DataFrame to store proxy & calibration data side-by-side
        # Fill-in proxy data
        df = pd.DataFrame({'time': proxy.time, 'y': proxy.values})
        # Add calibration data
        frame = pd.DataFrame({'time': cyears, 'Calibration': calvals})
        df = df.merge(frame, how='inner', on='time')

        df.set_index('time', drop=True, inplace=True)
        df.sort_index(inplace=True)

        # ensure all df entries are floats: if not, sm.ols gives garbage
        df = df.astype(np.float)

        nobs = len(df.index)
        if nobs < REQ_NOBS:  # skip rest if insufficient overlapping data
            raise PSMTooFewObsError('Less than {:d} overlapping observations '
                                    'available to calibrate PSM. Nobs={:d}'
                                    ''.format(REQ_NOBS, nobs))

        reg_xa = df['Calibration'].values
        reg_ya = df['y'].values

        model = LinearRegression(fit_intercept=True, normalize=False)
        model.fit(reg_xa[:, None], reg_ya[:])
        self.intercept = model.intercept_
        self.slope = model.coef_[0]
        self.NbPts = nobs

        yhat = model.predict(reg_xa[:, None])
        ss_tot = ((reg_ya - reg_ya.mean())**2).sum()
        ss_res = ((reg_ya - yhat)**2).sum()

        # Model fit
        self.R2 = 1 - (ss_res / ss_tot)
        # denom in last term is (n - p - 1) where n is nsamples and p is num
        # explanatory variables (which is 1 for univariate reg
        self.R2adj = 1 - (1 - self.R2) * ((nobs - 1) / (nobs - 1 - 1))

        # BIC assumes errors are IID
        # k is the number of estimated parameters (intercept, slope, ss_res)
        k = 3
        self.BIC = nobs * np.log(ss_res/nobs) + k * np.log(nobs)
        self.AIC = 2 * k - 2 * np.log(ss_res)



        # Perform the regression
        # regress = sm.ols(formula="y ~ Calibration", data=df).fit()
        # # number of data points used in the regression
        #
        # # Assign PSM calibration attributes
        # # Extract the needed regression parameters
        # self.intercept = regress.params[0]
        # self.slope = regress.params[1]
        # self.NbPts = nobs
        self.corr = np.sqrt(self.R2)
        if self.slope < 0:
            self.corr = -self.corr

        # Stats on fit residuals
        # MSE = np.mean(regress.resid ** 2)
        self.R = ss_res / nobs
        self.SSE = ss_res

        # Model information
        # self.AIC = regress.aic
        # self.BIC = regress.bic
        # self.R2 = regress.rsquared
        # self.R2adj = regress.rsquared_adj

        # Extra diagnostics
        reg_xa = df['Calibration'].values
        reg_ya = df['y'].values
        self.calib_time = df.index.values
        self.calib_refer_values = reg_xa
        self.calib_proxy_values = reg_xa
        fit = self.slope * self.calib_refer_values + self.intercept
        self.calib_proxy_fit = fit

        if diag_output:
            # Diagnostic output
            print("***PSM stats:")
            # print(regress.summary())

            if diag_output_figs:
                # Figure (scatter plot w/ summary statistics)
                line = self.slope * reg_xa + self.intercept
                plt.plot(reg_xa, line, 'r-', reg_xa, reg_ya, 'o',
                           markersize=7, markerfacecolor='#5CB8E6',
                           markeredgecolor='black', markeredgewidth=1)
                # GH: I don't know how this ran in the first place; must exploit
                # some global namespace
                plt.title('%s: %s' % (proxy.type, proxy.id))
                plt.xlabel('Calibration data')
                plt.ylabel('Proxy data')
                xmin, xmax, ymin, ymax = plt.axis()
                # Annotate with summary stats
                ypos = ymax - 0.05 * (ymax - ymin)
                xpos = xmin + 0.025 * (xmax - xmin)
                plt.text(xpos, ypos, 'Nobs = %s' % str(nobs), fontsize=12,
                           fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos,
                           'Slope = %s' % "{:.4f}".format(self.slope),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos,
                           'Intcpt = %s' % "{:.4f}".format(self.intercept),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos, 'Corr = %s' % "{:.4f}".format(self.corr),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos, 'Res.MSE = %s' % "{:.4f}".format(MSE),
                           fontsize=12, fontweight='bold')

                plt.savefig('proxy_%s_%s_LinearModel_calib.png' % (
                    proxy.type.replace(" ", "_"), proxy.id),
                              bbox_inches='tight')
                plt.close()

    @staticmethod
    def get_kwargs(config):
        try:
            psm_data = LinearPSM._load_psm_data(config)
            return {'psm_data': psm_data}
        except IOError as e:
            print(e)
            return {}

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_psm_data(pre_calib_file):
        """Helper method for loading from dataframe"""

        if pre_calib_file is None:
            raise IOError('No pre-calibration file specified.')

        with open(pre_calib_file, mode='rb') as f:
            data = pickle.load(f)

        return data


class LinearPSM_TorP(BasePSM):
    """
    TODO: FIX
    PSM based on linear regression w.r.t. temperature or precipitation
    
    **Important note: This class assumes that all linear PSMs have been 
                      pre-calibrated w.r.t. to temperature and precipitation, 
                      using the "LinearPSM" class defined above.
                      No attempts at calibration are performed here! 
                      The following code only assigns PSM linear regression 
                      parameters to individual proxies from fits to temperature
                      OR precipitation/moisture according to a measure of
                      "goodness-of-fit" criteria (here, MSE of the resisduals).

    Attributes
    ----------
    sensitivity: string
        Indicates the sensitivity (temperature vs moisture) of the proxy record
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    corr: float
        Correlation of proxy record against calibration data
    slope: float
        Linear regression slope of proxy/calibration fit
    intercept: float
        Linear regression y-intercept of proxy/calibration fit
    R: float
        Mean-squared error of proxy/calibration fit

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_data: dict
        Pre-calibrated PSM dictionary containing current associated
        proxy site's calibration information

    Raises
    ------
    ValueError
        If PSM is below critical correlation threshold.
    """

    def __init__(self, psm_config, proxy_obj, psm_data_T=None,
                 psm_data_P=None, diag_out=None, diag_fig=None,
                 avg_key=None, on_the_fly_calib=False):

        self.psm_key = 'linear_TorP'
        linearTorP_cfg = psm_config.linear_TorP

        proxy = proxy_obj.type
        site = proxy_obj.id
        r_crit = linearTorP_cfg.psm_r_crit
        metric = linearTorP_cfg.metric
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        self.elev = proxy_obj.elev

        self.datatag_T = linearTorP_cfg.datatag_T
        self.datatag_P = linearTorP_cfg.datatag_P

        self.avg_type = linearTorP_cfg.avg_type

        # Get the averaging interval information from the configuration
        if self.avg_type == 'annual':
            avg_key = 'annual_std'
        elif self.avg_type == 'seasonal':
            if avg_key is None:
                elem_to_avg = proxy_obj.seasonality
                [avg_key, _] = psm_config.handle_proxy_elem_list(elem_to_avg)
        else:
            raise KeyError('Unrecognized average type in PSM initialization...'
                           '\nExpected "annual" or "seasonal". Got {}'
                           ''.format(self.avg_type))

        self.avg_interval = avg_key
        # Update seasonality
        self.seasonality = self._handle_single_input_avg_key(avg_key,
                                                             psm_config,
                                                             linearTorP_cfg)

        try:
            psm_obj_T = LinearPSM(linearTorP_cfg.tempearature,
                                  proxy_obj, psm_data=psm_data_T,
                                  diag_out=diag_out, diag_fig=diag_fig,
                                  avg_key=avg_key,
                                  on_the_fly_calib=on_the_fly_calib)
        except (IOError, ValueError) as e:
            psm_obj_T = None

        try:
            psm_obj_P = LinearPSM(linearTorP_cfg.moisture, proxy_obj,
                                  psm_data=psm_data_P,
                                  diag_out=diag_out, diag_fig=diag_fig,
                                  avg_key=avg_key,
                                  on_the_fly_calib=on_the_fly_calib)
        except (IOError, ValueError) as e:
            psm_obj_P = None

        if psm_obj_T is not None and psm_obj_P is not None:
            if metric == 'corr':
                compare_T = psm_obj_T.corr
                compare_P = psm_obj_P.corr
            else:
                compare_T = psm_obj_T.R
                compare_P = psm_obj_P.R

            if abs(compare_T) >= abs(compare_P):
                self.sensitivity = 'temperature'
            else:
                self.sensitivity = 'moisture'
        elif psm_obj_T is not None:
            self.sensitivity = 'temperature'
        elif psm_obj_P is not None:
            self.sensitivity = 'moisture'
        else:
            raise PSMTorPCalibrationError('Proxy PSM could not be calibrated...'
                                          ' Skipping: {}'.format(proxy_obj.id))

        if self.sensitivity == 'temperature':
            self.psm_obj = psm_obj_T
            self.psm = psm_obj_T.psm
        else:
            self.psm_obj = psm_obj_P
            self.psm = psm_obj_P.psm

        self.datainfo = self.psm_obj.datainfo

        # Variable mapping connected to state information
        self.psm_vartype = self.datainfo['psm_vartype']

        # All variable types required to use the psm
        self.psm_req_types = (self.psm_vartype,)

        self.corr = self.psm_obj.corr
        self.slope = self.psm_obj.slope
        self.intercept = self.psm_obj.intercept
        self.R = self.psm_obj.R
        self.seasonality = self.psm_obj.seasonality

        vartype_as_tuple = tuple(self.psm_vartype.items())[0]
        self.req_avg_intervals = {vartype_as_tuple: self.avg_interval}

        # Raise exception if critical correlation value not met
        if abs(self.psm_obj.corr) < r_crit:
            raise PSMFitThresholdError('Proxy model correlation ({:.2f}) does '
                                       'not meet critical threshold ({:.2f}).'
                                       ''.format(self.corr, r_crit))

    # Define the error model for this proxy
    def error(self):
        return self.R

    def calibrate(self, calib_obj, proxy, diag_output=False, diag_output_figs=False):
        """
        Calibrate given proxy record against observation data and set relevant
        PSM attributes.

        Parameters
        ----------
        calib_obj: calibration_master like
            Calibration object containing data, time, lat, lon info
        proxy: BaseProxyObject like
            Proxy object to fit to the calibration data
        diag_output, diag_output_figs: bool, optional
            Diagnostic output flags for calibration method
        """

        print('Calibration not performed in this psm class!')
        pass


class BilinearPSM(BasePSM):
    """
    TODO: FIX
    PSM based on bivariate linear regression w.r.t. temperature AND precipitation/moisture
    

    Attributes
    ----------
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    corr: float
        Correlation of proxy record against calibration data
    slope: float
        Linear regression slope of proxy/calibration fit
    intercept: float
        Linear regression y-intercept of proxy/calibration fit
    R: float
        Mean-squared error of proxy/calibration fit

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_data: dict
        Pre-calibrated PSM dictionary containing current associated
        proxy site's calibration information

    Raises
    ------
    ValueError
        If PSM is below critical correlation threshold.
    """

    def __init__(self, psm_config, proxy_obj, psm_data=None,
                 avg_key_T=None, calib_obj_T=None,
                 avg_key_P=None, calib_obj_P=None,
                 on_the_fly_calib=False):

        self.psm_key = 'bilinear'
        bilinear_cfg = psm_config.bilinear

        ignore_pre_calib = bilinear_cfg.ignore_pre_calib

        proxy = proxy_obj.type
        site = proxy_obj.id
        r_crit = bilinear_cfg.psm_r_crit
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        self.elev = proxy_obj.elev

        self.avg_type = bilinear_cfg.avg_type

        self.psm_vartype_T = bilinear_cfg.temperature.datainfo['psm_vartype']
        self.psm_vartype_P = bilinear_cfg.moisture.datainfo['psm_vartype']

        self.psm_req_types = (self.psm_vartype_T, self.psm_vartype_P)

        # Assign sensitivity as temperature_moisture
        self.sensitivity = 'temperature_moisture'

        # PSM Model Info
        self.SSE = None
        self.AIC = None
        self.BIC = None
        self.R2 = None
        self.R2adj = None

        # For diagnostics
        self.calib_time = None
        self.calib_refer_values = None
        self.calib_proxy_values = None
        self.calib_proxy_fit = None

        # Try using pre-calibrated psm_data
        try:
            if ignore_pre_calib:
                raise IOError('Ignore pre-calibrated PSMs is specified.')

            # Try using pre-calibrated psm_data
            if psm_data is None:
                pre_calib_fpath = bilinear_cfg.pre_calib_datafile
                psm_data = self._load_psm_data(pre_calib_fpath)

            psm_site_data = psm_data[(proxy, site)]

            self.corr = psm_site_data['PSMcorrel']
            self.slope_temperature = psm_site_data['PSMslope_temperature']
            self.slope_moisture = psm_site_data['PSMslope_moisture']
            self.intercept = psm_site_data['PSMintercept']
            self.R = psm_site_data['PSMmse']

            self.calib_temperature = psm_site_data['calib_temperature']
            self.calib_moisture = psm_site_data['calib_moisture']

            # check if seasonality defined in the psm data
            # if it is, return as an attribute of psm object
            if 'Seasonality' in list(psm_site_data.keys()):
                self.seasonality_T = psm_site_data['Seasonality']
                self.seasonality_P = self.seasonality_T

                avg_interval, _ = psm_config.handle_proxy_elem_list(self.seasonality_T)
                self.avg_interval_T = avg_interval
                self.avg_interval_P = avg_interval

            if 'Seasonality_T' in list(psm_site_data.keys()):
                self.seasonality_T = psm_site_data['Seasonality_T']
                self.avg_interval_T = psm_site_data['avg_interval_T']
                psm_config.handle_proxy_elem_list(self.seasonality_T)

            if 'Seasonality_P' in list(psm_site_data.keys()):
                self.seasonality_P = psm_site_data['Seasonality_P']
                self.avg_interval_P = psm_site_data['avg_interval_P']
                psm_config.handle_proxy_elem_list(self.seasonality_P)

        except KeyError as e:
            raise KeyError('Proxy in database but not found in pre-calibration '
                           'file... Skipping: {}'.format(proxy_obj.id))
        except (IOError, FileNotFoundError) as e:
            # No precalibration found, have to do it for this proxy
            print('No pre-calibration found for {}:{}'.format(proxy_obj.id,
                                                              proxy_obj.type))
            if not on_the_fly_calib:
                raise e
            else:
                print(' ... calibrating on the fly')

                # Get the averaging interval information from the configuration
                if self.avg_type == 'annual':
                    avg_key_T = 'annual_std'
                    avg_key_P = None
                elif self.avg_type == 'seasonal':
                    if avg_key_T is None and avg_key_P is None:
                        elem_to_avg = proxy_obj.seasonality
                        [avg_key_T, _] = psm_config.handle_proxy_elem_list(
                            elem_to_avg)
                        avg_key_P = None
                else:
                    raise KeyError(
                        'Unrecognized average type in PSM initialization...'
                        '\nExpected "annual" or "seasonal". Got {}'
                        ''.format(self.avg_type))

                # Set the seasonality based on averaging keys
                self._handle_input_avg_keys(avg_key_T, avg_key_P, psm_config,
                                            bilinear_cfg)

                if calib_obj_T is None:
                    source = bilinear_cfg.temperature.datatag
                    calib_class = LMR_gridded.get_analysis_var_class(source)
                    calib_obj_T = calib_class.load(bilinear_cfg.temperature,
                                                   anomaly=True)

                if calib_obj_P is None:
                    source = bilinear_cfg.moisture.datatag
                    calib_class = LMR_gridded.get_analysis_var_class(source)
                    calib_obj_P = calib_class.load(bilinear_cfg.moisture,
                                                   anomaly=True)

            self.calibrate(calib_obj_T, calib_obj_P, proxy_obj)

        vartype_as_tuple_T = tuple(self.psm_vartype_T.items())[0]
        vartype_as_tuple_P = tuple(self.psm_vartype_P.items())[0]
        self.req_avg_intervals = {vartype_as_tuple_T: self.avg_interval_T,
                                  vartype_as_tuple_P: self.avg_interval_P}

        # Raise exception if critical correlation value not met
        if abs(self.corr) < r_crit:
            raise PSMFitThresholdError('Proxy model correlation ({:.2f}) does '
                                       'not meet critical threshold ({:.2f}).'
                                       ''.format(self.corr, r_crit))

    def _handle_input_avg_keys(self, avg_key_T, avg_key_P, psm_config,
                               bilinear_cfg):

        if avg_key_T is not None and avg_key_P is None:
            avg_key_P = avg_key_T
        elif avg_key_P is not None and avg_key_T is None:
            avg_key_T = avg_key_P

        seasonality_T = self._handle_single_input_avg_key(avg_key_T,
                                                          psm_config,
                                                          bilinear_cfg.temperature)
        seasonality_P = self._handle_single_input_avg_key(avg_key_P,
                                                          psm_config,
                                                          bilinear_cfg.moisture)

        self.seasonality_T = seasonality_T
        self.seasonality_P = seasonality_P
        self.avg_interval_T = avg_key_T
        self.avg_interval_P = avg_key_P

    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, state_object):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        state_object: LMR_gridded.State
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------
        state_var_T = state_object.get_psm_var_key(self.psm_vartype_T)
        state_var_P = state_object.get_psm_var_key(self.psm_vartype_P)

        var_avg_key_T = (state_var_T, self.avg_interval_T)
        var_avg_key_P = (state_var_P, self.avg_interval_P)

        gridpoint_data_T = self._get_gridpoint_data_from_state(var_avg_key_T,
                                                               state_object)

        gridpoint_data_P = self._get_gridpoint_data_from_state(var_avg_key_P,
                                                               state_object)

        Ye = self.basic_psm(gridpoint_data_T, gridpoint_data_P)

        return Ye

    def basic_psm(self, data_T, data_P):

        coef_T = self.slope_temperature
        coef_P = self.slope_moisture
        return coef_T*data_T + coef_P*data_P + self.intercept

    # Define the error model for this proxy
    def error(self):
        return self.R

    # TODO: Simplify a lot of the actions in the calibration
    def calibrate(self, cobj_T, cobj_P, proxy, diag_output=False, diag_output_figs=False):
        """
        Calibrate given proxy record against observation data and set relevant
        PSM attributes.

        Parameters
        ----------
        cobj_T: calibration_master like
            Calibration object containing temperature data, time, lat, lon info
        cobj_P: calibration_master like
            Calibration object containing precipitation/moisture data, time, lat, lon info
        proxy: BaseProxyObject like
            Proxy object to fit to the calibration data
        diag_output, diag_output_figs: bool, optional
            Diagnostic output flags for calibration method
        """

        # ----------------------------------------------
        # Use bilinear model (regression) as default PSM
        # ----------------------------------------------

        calvals_T = self.get_close_grid_point_data(cobj_T.data,
                                                   cobj_T.lon,
                                                   cobj_T.lat)
        cyears_T = np.array([t.year for t in cobj_T.time])

        valid_idx = np.isfinite(calvals_T)
        calvals_T = calvals_T[valid_idx]
        cyears_T = cyears_T[valid_idx]

        calvals_P = self.get_close_grid_point_data(cobj_P.data,
                                                   cobj_P.lon,
                                                   cobj_P.lat)
        cyears_P = np.array([t.year for t in cobj_P.time])

        valid_idx = np.isfinite(calvals_P)
        calvals_P = calvals_P[valid_idx]
        cyears_P = cyears_P[valid_idx]

        # ---------------------------
        # Perform bilinear regression
        # ---------------------------

        # Use panda DataFrame to store proxy & calibration data side-by-side
        # Fill-in proxy data
        df = pd.DataFrame({'time': proxy.time, 'y': proxy.values})
        # Add temperature calibration data
        frameT = pd.DataFrame({'time': cyears_T, 'Temperature': calvals_T})
        df = df.merge(frameT, how='inner', on='time')
        # Add precipitation/moisture calibration data
        frameP = pd.DataFrame({'time': cyears_P, 'Moisture': calvals_P})
        df = df.merge(frameP, how='inner', on='time')

        df.set_index('time', drop=True, inplace=True)
        df.sort_index(inplace=True)

        # ensure all df entries are floats: if not, sm.ols gives garbage
        df = df.astype(np.float)

        nobs = len(df.index)
        if nobs < REQ_NOBS:  # skip rest if insufficient overlapping data
            raise PSMTooFewObsError('Less than {:d} overlapping observations '
                                    'available to calibrate PSM. Nobs={:d}'
                                    ''.format(REQ_NOBS, nobs))
        reg_xa = df[['Temperature', 'Moisture']].values
        reg_ya = df['y'].values

        model = LinearRegression(fit_intercept=True, normalize=False)
        model.fit(reg_xa, reg_ya[:])
        self.intercept = model.intercept_
        self.slope_temperature = model.coef_[0]
        self.slope_moisture = model.coef_[1]
        self.NbPts = nobs

        yhat = model.predict(reg_xa)
        ss_tot = ((reg_ya - reg_ya.mean()) ** 2).sum()
        ss_res = ((reg_ya - yhat) ** 2).sum()

        # Model fit
        self.R2 = 1 - (ss_res / ss_tot)
        # denom in last term is (n - p - 1) where n is nsamples and p is num
        # explanatory variables (which is 1 for univariate reg
        self.R2adj = 1 - (1 - self.R2) * ((nobs - 1) / (nobs - 2 - 1))

        # BIC assumes errors are IID
        # k is the number of estimated parameters (intercept, slope, ss_res)
        k = 4
        self.BIC = nobs * np.log(ss_res / nobs) + k * np.log(nobs)
        self.AIC = 2 * k - 2 * np.log(ss_res)
        # Perform the regression
        # regress = sm.ols(formula="y ~ Temperature + Moisture", data=df).fit()
        #regress = sm.ols(formula="y ~ Temperature * Moisture", data=df).fit() # w/ interaction term
        #regress = sm.ols(formula="y ~ Temperature + Moisture +Temperature:Moisture", data=df).fit() # w/ interaction term

        # extract the needed regression parameters
        # self.intercept = regress.params[0]
        # self.slope_temperature = regress.params[1]
        # self.slope_moisture = regress.params[2]
        #
        # self.NbPts = nobs
        self.corr = np.sqrt(self.R2)

        # Stats on fit residuals
        self.R = ss_res / nobs
        self.SSE = ss_res
        #
        # # Model information
        # self.AIC = regress.aic
        # self.BIC = regress.bic
        # self.R2 = regress.rsquared
        # self.R2adj = regress.rsquared_adj

        reg_xa_T = df['Temperature'].values
        reg_xa_P = df['Moisture'].values
        reg_ya = df['y'].values

        self.calib_time = df.index.values
        self.calib_proxy_values = reg_ya
        self.calib_temperature_refer_values = reg_xa_T
        self.calib_moisture_refer_values = reg_xa_P
        fit = self.slope_temperature * reg_xa_T + self.slope_moisture * reg_xa_P + self.intercept
        self.calib_proxy_fit = fit

        if diag_output:
            # Diagnostic output
            print("***PSM stats:")
            print(regress.summary())
            print(' ')
            print('Pairwise correlations:')
            print('----------------------')
            print(df.corr())
            print(' ')
            print(' ')
            if diag_output_figs:
                # Figure (scatter plot w/ summary statistics)

                y = df['y']
                X = df[['Temperature','Moisture']]

                xx1, xx2 = np.meshgrid(np.linspace(X.Temperature.min(),X.Temperature.max(),200),
                                       np.linspace(X.Moisture.min(),X.Moisture.max(),200))
                Model = self.intercept + self.slope_temperature*xx1 + self.slope_moisture*xx2

                fig = plt.figure(figsize=(10,7))
                ax = Axes3D(fig,azim=-130,elev=15)

                # plot hyperplane
                surf = ax.plot_surface(xx1,xx2,Model,cmap=plt.cm.RdBu_r,alpha=0.6,linewidth=0)

                # plot data points
                resid = y - (self.intercept + self.slope_temperature*X['Temperature'] + self.slope_temperature*X['Moisture'])
                ax.scatter(X[resid>=0.].Temperature,X[resid>=0.].Moisture,y[resid>=0.], color='black', alpha=1.0,facecolor='white')
                ax.scatter(X[resid<0.].Temperature,X[resid<0.].Moisture,y[resid<0.], color='black', alpha=1.0)

                # axis labels
                plt.suptitle('%s: %s' % (proxy.type, proxy.id),fontsize=16,fontweight='bold')

                ax.set_xlabel('Calibration temperature')
                ax.set_ylabel('Calibration moisture')
                ax.set_zlabel('Proxy data')

                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                zmin, zmax = ax.get_zlim()

                ax.text2D(0.05,0.92, 'Nobs=%s  Correlation=%s  MSE=%s' %(self.NbPts, "{:.4f}".format(self.corr),"{:.4f}".format(self.R)), transform=ax.transAxes, fontsize=14)
                ax.text2D(0.05,0.89, 'proxy = %s + (%s x Temperature) + (%s x Moisture)' %("{:.4f}".format(self.intercept),
                                                                                                 "{:.4f}".format(self.slope_temperature),
                                                                                                 "{:.4f}".format(self.slope_moisture)), transform=ax.transAxes, fontsize=14)

                plt.savefig('proxy_%s_%s_BilinearModel_calib.png' % (
                    proxy.type.replace(" ", "_"), proxy.id),
                              bbox_inches='tight')
                plt.close()


    @staticmethod
    def get_kwargs(config):
        try:
            psm_data = BilinearPSM._load_psm_data(config)
            return {'psm_data': psm_data}
        except IOError as e:
            print(e)
            return {}


class h_interpPSM(BasePSM):
    """
    Horizontal interpolator as a PSM.
    Interpolator is a distance-weighted average of surrounding gridpoint values, 
    with weights determined by exponentially-decaying function calculated using
    a user-defined influence radius.

    Attributes
    ----------
    RadiusInfluence: float
        Distance-scale used the calculation of exponentially-decaying weights in interpolator (in km)
    sensitivity: string
        Indicates the sensitivity (temperature vs moisture) of the proxy record
        Here set to "None" as this information is irrelevant, but given for 
        compatibility with other PSM classes
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    R: float
        Obs. error variance associated to proxy site

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to

    Raises
    ------
    ValueError
        ...
    """

    def __init__(self, config, proxy_obj, R_data=None):

        self.psm_key = 'h_interp'

        self.proxy = proxy_obj.type
        self.site = proxy_obj.id
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        try:
            self.elev = proxy_obj.elev
        except:
            self.elev = None
        self.sensitivity = None
        self.RadiusInfluence = config.psm.h_interp.radius_influence

        # Try finding file containing obs. error variance info for **d18O** ??
        # and assign the R value to proxy psm object
        try:
            if R_data is None:
                R_data = self._load_psm_data(config)
            self.R = R_data[(self.proxy, self.site)]
        except (KeyError, IOError) as e:
            # No obs. error variance file found
            print(e)
            print(('Cannot find obs. error variance data for:' + str((
                   self.proxy, self.site))))


    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # define state variable (which isotope) that should be interpolated to proxy site
        # Now, d18O is the only isotope considered, irrespective of the proxy type to be assimilated.
        # (TODO: more comprehensive & flexible way to do this...)
        state_var = 'd18O_sfc_Amon'

        if state_var not in list(X_state_info.keys()):
            raise KeyError('Needed variable not in state vector for Ye'
                           ' calculation.')

        # TODO: end index should already be +1, more pythonic
        statevar_startidx, statevar_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')

        # Find row index of X for which [X_lat,X_lon] corresponds to closest
        # grid point to location of proxy site [self.lat,self.lon]

        # Calclulate distances from proxy site.
        stateDim = statevar_endidx - statevar_startidx + 1
        ensDim = Xb.shape[1]
        dist = np.empty(stateDim)
        dist = haversine(self.lon, self.lat,
                          X_coords[statevar_startidx:(statevar_endidx+1), ind_lon],
                          X_coords[statevar_startidx:(statevar_endidx+1), ind_lat])

        # Check if RadiusInfluence is defined (not None). Use weighted-averaging as interpolator if it is.
        # Otherwise, seek value at nearest gridpoint to proxy location
        if self.RadiusInfluence:
            # Calculate weighted-average
            #  exponential decay
            L =  self.RadiusInfluence
            weights = np.exp(-np.square(dist)/np.square(L))
            # make the weights sum to one
            weights /=weights.sum(axis=0)

            Ye = np.dot(weights.T,Xb[statevar_startidx:(statevar_endidx+1), :])
        else:
            # Pick value at nearest grid point
            # row index of nearest grid pt. in prior (minimum distance)
            kind = np.unravel_index(dist.argmin(), dist.shape)[0] + statevar_startidx

            Ye = np.squeeze(Xb[kind, :])


        return Ye

    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1

    @staticmethod
    def get_kwargs(config):
        try:
            # obs. error variabce data
            R_data = h_interpPSM._load_psm_data(config)
            return {'R_data': R_data}
        except IOError as e:
            print(e)
            print('In "h_interpPSM" class: Cannot find PSM calibration file: '
                  '%s. Exiting!' % pre_calib_file)
            raise SystemExit
    
    @staticmethod
    def _load_psm_data(config):
        """Helper method for loading from dataframe"""

        R_data_file = config.psm.h_interp.datafile_obsError

        if R_data_file:
            # check if file exists
            if not os.path.isfile(R_data_file):
                print('In "h_interpPSM" class: Cannot find PSM calibration file: %s. Exiting!' % pre_calib_file)
                raise SystemExit(1)

            else:
                # This returns a python dictionary with entries as: {(proxy type, proxy name): Rvalue}
                Rdata_dict={}
                with open(R_data_file,'r') as f:
                    linenb = 0
                    for line in f:
                        try:
                            ptype,pname,Rval = line.split(',') # expects commas as field separator
                            Rval = Rval.strip('\n') # remove end-or-line chars
                            # removes quotes (single or double) around the R value if present
                            if (Rval.startswith('"') and Rval.endswith('"')) or \
                               (Rval.startswith("'") and Rval.endswith("'")):
                                Rval = Rval[1:-1]
                            # populate dictionary entry, make sure R value is a float
                            Rdata_dict[(ptype,pname)] = float(Rval)
                        except:
                            print('WARNING: In "h_interpPSM", load_psm_data: Error reading/parsing data on line', linenb, ':', line)

                linenb +=1

        else:
            Rdata_dict = {}

        return Rdata_dict


@class_docs_fixer
class BayesRegUK37PSM(BasePSM):
    """
    ... 

    Attributes
    ----------

    ...

    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    R: float
        Obs. error variance associated to proxy site

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to.

    Raises
    ------
    ValueError
        ...
    """

    def __init__(self, config, proxy_obj, Bayes_data=None):

        self.psm_key = 'bayesreg_uk37'

        proxy = proxy_obj.type
        site = proxy_obj.id
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        self.elev = proxy_obj.elev

        self.sensitivity = 'sst'

        # Matlab engine # for old matlab implementation
        #self.MatlabEng = config.psm.bayesreg_uk37.MatlabEng

        self.psm_required_variables = config.psm.bayesreg_uk37.psm_required_variables
        self.datafile_BayesRegressionData = config.psm.bayesreg_uk37.datafile_BayesRegressionData

        try:
            if Bayes_data is None:
                Bayes_data = self._load_psm_data(config)
            self.tau2 = Bayes_data['tau2']
            self.Bspline = Bayes_data['Bspline']
            # Getting the info for obs. error variance (R)
            self.R = np.mean(Bayes_data['tau2'])
        except (KeyError, IOError) as e:
            # No obs. error variance file found
            logger.error(e)
            logger.info('Cannot find obs. error variance data for:' + str((self.proxy, self.site)))


    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Defining state variables to consider in the calculation of Ye's

        state_var = list(self.psm_required_variables.keys())[0]

        if state_var not in list(X_state_info.keys()):
            raise KeyError('Needed variable not in state vector for Ye'
                           ' calculation.')

        var_startidx, var_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')
        X_lons = X_coords[var_startidx:(var_endidx+1), ind_lon]
        X_lats = X_coords[var_startidx:(var_endidx+1), ind_lat]
        var_data = Xb[var_startidx:(var_endidx+1), :]

        # get prior data at proxy site
        gridpoint_data = get_data_closest_gridpt(var_data,X_lons,X_lats,self.lon,self.lat,getvalid=True)

        # check if gridpoint_data is in K: need deg. C for forward model
        # crude check...
        if np.nanmin(var_data) > 200.0:
            gridpoint_data = gridpoint_data - 273.15

        order = 2  # 3 in MATLAB
        knots = np.array([-0.4, 15, 24, 26, 29.6])
        heads = [knots[0]] * order
        tails = [knots[-1]] * order
        tck = [np.concatenate([heads, knots, tails]), self.Bspline, order]
        outdat = interpolate.splev(x = gridpoint_data, tck = tck, ext = 0)
        out_ens = np.random.normal(outdat, np.sqrt(self.tau2))
        Ye_ens = out_ens.T

        """
        # Matlab implementation ...
        # Making the input array matlab-friendly. Can do it through a list. 
        tmp = np.array(gridpoint_data)[np.newaxis]        
        MATgridpoint_data = matlab.double((tmp.T).tolist())
        # Ye_mat is a matlab array. It contains an *ensemble* of estimates for every
        # prior ensemble member.
        Ye_mat = self.MatlabEng.UK37_forward_model_func(self.datafile_BayesRegressionData,
                                                        MATgridpoint_data)
        # convert to numpy array for good measure
        Ye_ens = np.array(Ye_mat)
        """

        # take the mean of the ensemble of estimates
        Ye = np.mean(Ye_ens, axis=1)

        return Ye


    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1

    @staticmethod
    def _load_psm_data(config):
        """Helper method for loading from dataframe"""

        data_file = config.psm.bayesreg_uk37.datafile_BayesRegressionData

        if data_file:
            # check if file exists
            if not os.path.isfile(data_file):
                print('In "BayesRegUK37PSM" class: Cannot find file containing obs. error info.: %s. Exiting!' % data_file)
                raise SystemExit(1)
            else:
                # Load in the data
                regression_data = loadmat(data_file)
                tau2_data = regression_data['tau2_draws_final']
                B_data = regression_data['b_draws_final']
                BayesData_dict = {'Bspline': B_data, 'tau2': tau2_data}
        else:
            BayesData_dict = {}

        return BayesData_dict

    @staticmethod
    def get_kwargs(config):
        pass


# Mapping dict to PSM object type, this is where proxy_type/psm relations
# should be specified (I think.) - AP
_psm_classes = {'linear': LinearPSM, 'linear_TorP': LinearPSM_TorP,
                'bilinear': BilinearPSM, 'h_interp': h_interpPSM,
                'bayesreg_uk37': BayesRegUK37PSM}


def get_psm_class(psm_type):
    """
    Retrieve psm class type to be instantiated.

    Parameters
    ----------
    psm_type: str
        Dict key to retrieve correct PSM class type.

    Returns
    -------
    BasePSM like:
        Class type to be instantiated and attached to a proxy.
    """
    return _psm_classes[psm_type]

