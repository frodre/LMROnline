from abc import ABCMeta, abstractmethod
import numpy as  np
import os.path as path
import os
import logging

import pylim.LIM as LIM
import pylim.DataTools as DT
from LMR_utils2 import class_docs_fixer, augment_docstr, regrid_sphere
import LMR_gridded

logger = logging.getLogger(__name__)

class BaseForecaster:
    """
    Class defining methods for LMR forecasting
    """

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def forecast(self, state_obj):
        """
        Perform forecast.  Should edit the state vector in place.
        TODO: might change this so that these are attached to
        a state since they depend on them for input. See PSM obj.
        for a similar situation....

        Parameters
        ----------
        state_obj: LMR_gridded.State
            Current state to be forecasted.
        """
        pass


@class_docs_fixer
class LIMForecaster(BaseForecaster):
    """
    Linear Inverse Model Forecaster
    """

    def __init__(self, forecaster_config):
        nelem_in_yr = 1
        lim_cfg = forecaster_config.lim
        detrend = lim_cfg.detrend
        num_pcs = lim_cfg.fcast_num_pcs
        prior_map = lim_cfg.prior_mapping
        fcast_lead = lim_cfg.fcast_lead
        fcast_type = lim_cfg.fcast_type
        self.use_ens_mean_fcast = lim_cfg.use_ens_mean_fcast

        fcast_calib_vars = LMR_gridded.ForecasterVariable.load_allvars(lim_cfg)
        fcast_calib_dobjs = {key: var.forecast_var_to_pylim_dataobj()
                             for key, var in fcast_calib_vars.items()}
        self.calib_dobjs = fcast_calib_dobjs

        var_order = []
        calib_state = []
        calib_eofs = {}
        fcast_state_bnds = {}
        start = 0
        for key, dobj in fcast_calib_dobjs.items():
            dobj.calc_anomaly(nelem_in_yr)  # TODO: hardcoded annual
            if detrend:
                dobj.detrend_data()
            dobj.area_weight_data()
            dobj.eof_proj_data(num_pcs)
            data = dobj.eof_proj[:]

            var_order.append(key)
            calib_state.append(data)
            calib_eofs[key] = dobj._eofs
            end = start + data.shape[-1]
            fcast_state_bnds[key] = (start, end)
            start = end

        calib_state = np.concatenate(calib_state, axis=-1)


        # Search for pre-calibrated LIM to save time
        # fpath, fname = path.split(infile)
        # precalib_path = path.join(fpath, 'lim_precalib')
        # if do_detrend:
        #     precalib_fname = path.splitext(fname)[0] + '.lim.pckl'
        # else:
        #     precalib_fname = path.splitext(fname)[0] + '_nodetr.lim.pckl'
        # precalib_pathfname = path.join(precalib_path, precalib_fname)
        #
        # if not ignore_precalib:
        #     try:
        #         self.lim = LIM.LIM.from_precalib(precalib_pathfname)
        #         print ('Pre-calibrated lim loaded from '
        #                '{}'.format(precalib_pathfname))
        #         return
        #     except IOError:
        #         print ('No pre-calibrated LIM found.')

        if fcast_type == 'noise_integrate':
            fit_noise = True
        else:
            fit_noise = False

        self.lim = LIM.LIM(calib_state, nelem_in_tau1=nelem_in_yr,
                           fit_nosie=fit_noise)
        self.var_order = var_order
        self.var_eofs = calib_eofs
        self.fcast_state_bnds = fcast_state_bnds
        self.prior_map = prior_map
        self.fcast_lead = fcast_lead

        if fcast_type == 'perfect':
            self._fcast_func = self._perf_forecast
        elif fcast_type == 'ens_mean_perfect':
            self._fcast_func = self._perf_ensmean_forecast
        elif fcast_type == 'noise_integrate':
            self._fcast_func = self._noise_integration
        else:
            raise ValueError('Unrecognized forecast type specification: '
                             '{}'.format(fcast_type))

        # if not path.exists(precalib_path):
        #     os.makedirs(precalib_path)
        #
        # self.lim.save_precalib(precalib_pathfname)

    def forecast(self, state_obj):
        """Perfect no-noise forecast. Drastically reduces output variance."""
        fcast_state = []
        is_compressed = []
        for var in self.var_order:
            data = state_obj.get_var_data(self.prior_map[var])
            if np.ma.is_masked(data):
                nens = data.shape[-1]
                data = data.compressed().reshape(-1, nens)
                is_compressed.append(var)
            # Transposes sampling dimension and projects into EOF space
            eof_proj = np.dot(data.T, self.var_eofs[var])
            fcast_state.append(eof_proj)

        fcast_state = np.concatenate(fcast_state, axis=-1)

        fcast_data = self._fcast_func(fcast_state)

        # var_data is returned as a view for annual, so this re-assigns
        fcast_state_out = {}
        for var in self.var_order:
            start, end = self.fcast_state_bnds[var]
            fcast_out = fcast_data[:, start:end]
            phys_space_fcast = np.dot(fcast_out, self.var_eofs[var].T)
            if var in is_compressed:
                dobj = self.calib_dobjs[var]
                phys_space_fcast = dobj.inflate_full_grid(data=phys_space_fcast)
                phys_space_fcast = np.ma.masked_invalid(phys_space_fcast)
            fcast_state_out[self.prior_map[var]] = phys_space_fcast.T

        return fcast_state_out

    def _perf_forecast(self, state_arr):
        return self.lim.forecast(state_arr, [self.fcast_lead])[0]

    def _perf_ensmean_forecast(self, state_arr):
        # Take the ensemble mean and forecast on that
        fcast_data_ensmean = state_arr.mean(axis=-1, keepdims=True)
        fcast_data_enspert = state_arr - fcast_data_ensmean

        fcast_data = self.lim.forecast(fcast_data_ensmean,
                                       [self.fcast_lead])[0]

        fcast_data = fcast_data + fcast_data_enspert

        return fcast_data

    def _noise_integration(self, state_arr):
        return self.lim.noise_integration(state_arr, self.fcast_lead,
                                          timesteps=1440)



@class_docs_fixer
class PersistanceForecaster(BaseForecaster):
    """
    Persistance Forecaster

    Does nothing but persist the current state to the next time.
    """

    def __init__(self, config):
        pass

    def forecast(self, state_obj):
        pass


_forecaster_classes = {'lim': LIMForecaster,
                       'persist': PersistanceForecaster}


def get_forecaster_class(key):
    """
    Retrieve forecaster class type to be instantiated.

    Parameters
    ----------
    key: str
        Dict key to retrieve correct Forecaster class type.

    Returns
    -------
    BaseForecaster-like:
        Forecaster class to be instantiated
    """
    return _forecaster_classes[key]




