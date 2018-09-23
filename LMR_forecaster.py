from abc import abstractmethod
import numpy as np
import logging

import pylim.LIM as LIM
import pylim.Stats as plstat
from LMR_utils2 import class_docs_fixer
import LMR_gridded

logger = logging.getLogger(__name__)


class BaseForecaster:
    """
    Class defining methods for LMR forecasting
    """

    @abstractmethod
    def __init__(self, config, state_var_keys):
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

    def __init__(self, forecaster_config, state_var_keys):
        # TODO: hardcoded annual or greater
        nelem_in_yr = 1
        lim_cfg = forecaster_config.lim
        match_prior = lim_cfg.match_prior
        detrend = lim_cfg.detrend
        num_pcs = lim_cfg.fcast_num_pcs
        dobj_num_pcs = lim_cfg.dobj_num_pcs
        prior_map = lim_cfg.prior_mapping
        fcast_lead = lim_cfg.fcast_lead
        fcast_type = lim_cfg.fcast_type

        FcastVar = LMR_gridded.ForecasterVariable

        if match_prior:
            load_vars = state_var_keys
        else:
            load_vars = FcastVar.get_fcast_prior_match_vars(lim_cfg.fcast_varnames)

        save_attrs = [lim_cfg.datatag, num_pcs, dobj_num_pcs] + list(load_vars)

        self.valid_data_mask = {}
        self.var_eofs = {}
        self.var_order = []
        self.var_span = {}
        self._pre_concat_data = []
        self._start = 0
        self._end = 0

        for key, fcast_var in fcast_var_gen:

            # Convert data into pylim.DataObject and then perform processing
            dobj = self._process_forecast_variable(key, fcast_var, detrend,
                                                   nelem_in_yr, dobj_num_pcs)

            # Handle combination into LIMState
            self._process_lim_state_params(key, dobj, 'data')

        calib_state_std = self._combine_lim_state_data()

        # Calculate multi-variate EOFs
        [calib_state_eofs,
        calib_state_svals,
        calib_state_variance_stats] = _calc_limstate_eofs(calib_state_std,
                                                          num_pcs)
        self.calib_eofs = calib_state_eofs

        # Collect dobject data for projecting EOFs upon and calibrating LIM
        # [calib_state_reg,
        #  _, _] = _combine_dobjs_into_limstate(fcast_calib_dobjs, 'eof_proj')
        #
        # # Project unstandardized ata on the calculated EOFs
        # eof_proj_calib = calib_state_reg @ calib_state_eofs

        # Project unstandardized ata on the calculated EOFs
        eof_proj_calib = calib_state_std @ calib_state_eofs

        if fcast_type == 'noise_integrate':
            fit_noise = True
        else:
            fit_noise = False

        self.lim = LIM.LIM(eof_proj_calib, nelem_in_tau1=nelem_in_yr,
                           fit_noise=fit_noise)
        self.prior_map = prior_map
        self.fcast_lead = fcast_lead
        self.match_prior = match_prior

        if fcast_type == 'perfect':
            self._fcast_func = self._perf_forecast
        elif fcast_type == 'ens_mean_perfect':
            self._fcast_func = self._perf_ensmean_forecast
        elif fcast_type == 'noise_integrate':
            self._fcast_func = self._noise_integration
        else:
            raise ValueError('Unrecognized forecast type specification: '
                             '{}'.format(fcast_type))

    def forecast(self, state_obj):
        """Perfect no-noise forecast. Drastically reduces output variance."""
        fcast_state = []
        is_compressed = []
        for var_key in self.var_order:
            var, avg_interval = var_key
            prior_var_key = self.prior_map[var]
            data = state_obj.get_var_data(var_key)
            if np.any(np.isnan(data)):
                # Get the LIM calibration defined mask
                valid_mask = self.valid_data_mask[var_key]
                data = data[valid_mask, :]
                is_compressed.append(var_key)
            # Transposes sampling dimension and projects into EOF space
            eof_proj = np.dot(data.T, self.var_eofs[var_key])
            fcast_state.append(eof_proj)

        fcast_state = np.concatenate(fcast_state, axis=-1)
        fcast_state = fcast_state @ self.calib_eofs

        fcast_data = self._fcast_func(fcast_state)

        # translate forecast out of state eof basis
        fcast_data = fcast_data @ self.calib_eofs.T

        # gather forecast back in original physical space from prior
        fcast_state_out = {}
        for var_key in self.var_order:
            var, avg_interval = var_key
            prior_var_key = self.prior_map[var]
            fcast_out = _get_var_from_limstate(var_key, fcast_data,
                                               self.var_span)
            phys_space_fcast = fcast_out @ self.var_eofs[var_key].T
            if var_key in is_compressed:
                phys_space_fcast = self._decompress_field(var_key,
                                                          phys_space_fcast)

            fcast_state_out[(prior_var_key, avg_interval)] = phys_space_fcast.T

        # restore original data (climatological) when forecast prior vars
        # do not necessarily match full state data
        if not self.match_prior:
            state_obj.restore_orig_state()

        # Overwrite forecasted state data
        state_obj.update_var_data(fcast_state_out)

        return state_obj

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
        # TODO: need to add a seed list generated for each recon year
        return self.lim.noise_integration(state_arr, self.fcast_lead,
                                          timesteps=1440)

    def _decompress_field(self, key, data):
        # Simplified pylim.DataObject.inflate_full_grid because I don't want to
        # store the entire calibration object for each field

        valid_data = self.valid_data_mask[key]

        if data.ndim > 2:
            raise AttributeError('Function created to work for 2D data only.')

        new_data = np.empty((data.shape[0], len(valid_data))) * np.nan
        new_data[:, valid_data] = data

        return new_data

    def _process_forecast_variable(self, key, fcast_var, detrend, nelem_in_yr,
                                   dobj_num_pcs):

        # Convert data into pylim.DataObject and then perform processing
        data_obj = fcast_var.forecast_var_to_pylim_dataobj()

        if data_obj.valid_data is not None:
            self.valid_data_mask[key] = data_obj.valid_data

        data_obj.calc_anomaly(nelem_in_yr, save=(not detrend))

        if detrend:
            data_obj.detrend_data()
            proj_key = data_obj._DETRENDED
        else:
            proj_key = data_obj._ANOMALY

        # TODO: Fix cell area loading in LMR_gridded
        data_obj.area_weight_data(save=False)
        data_obj.eof_proj_data(dobj_num_pcs, proj_key=proj_key, save=False)
        self.var_eofs[key] = data_obj._eofs
        data_obj.standardize_data(save=False)

        return data_obj

    def _process_lim_state_params(self, key, data_obj, data_obj_dkey):
        self.var_order.append(key)
        dobj_data = getattr(data_obj, data_obj_dkey)
        self._pre_concat_data.append(dobj_data)
        self._end = self._start + dobj_data.shape[1]
        self.var_span[key] = (self._start, self._end)
        self._start = self._end

    def _combine_lim_state_data(self):

        curr_min = 1e10
        for var_data in self._pre_concat_data:
            curr_min = min(curr_min, var_data.shape[0])

        for i, var_data in enumerate(self._pre_concat_data):
            shave_yr = var_data.shape[0] - curr_min
            if shave_yr > 1:
                raise ValueError('Havent planned for more than 1 year being '
                                 'removed from data after it is averaged...')

            self._pre_concat_data[i] = var_data[shave_yr:]

        lim_state_data = np.concatenate(self._pre_concat_data, axis=1)
        self._pre_concat_data = None

        return lim_state_data


@class_docs_fixer
class PersistanceForecaster(BaseForecaster):
    """
    Persistance Forecaster

    Does nothing but persist the current state to the next time.
    """

    def __init__(self, config, state_var_keys):
        pass

    def forecast(self, state_obj):
        return state_obj


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


def _combine_dobjs_into_limstate(dobjs, dobj_key):
    var_order = []
    var_span = {}
    data = []

    start = 0
    for key, dobj in dobjs.items():
        var_order.append(key)
        dobj_data = getattr(dobj, dobj_key)
        data.append(dobj_data[:])
        end = start + dobj_data.shape[1]
        var_span[key] = (start, end)
        start = end

    # Address the differences in number of times due to time averages taken
    # that span a calendar year
    curr_min = 1e10
    for var_data in data:
        curr_min = min(curr_min, var_data.shape[0])
    for i, var_data in enumerate(data):
        shave_yr = var_data.shape[0] - curr_min
        if shave_yr > 1:
            raise ValueError('Havent planned for more than 1 year being '
                             'removed from data after it is averaged...')

        data[i] = var_data[shave_yr:]

    data = np.concatenate(data, axis=1)

    return data, var_order, var_span


def _get_var_from_limstate(var_key, data, var_span):

    start, end = var_span[var_key]
    return data[..., start:end]


def _calc_limstate_eofs(data, num_eofs):
    var_stats = {}
    state_eofs, state_svals = plstat.calc_eofs(data, num_eofs,
                                               var_stats_dict=var_stats)

    return state_eofs, state_svals, var_stats
