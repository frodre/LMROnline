from abc import abstractmethod
from copy import deepcopy
from collections import defaultdict
import numpy as np
import logging
import hashlib
import os
import pickle

import pylim.LIM as LIM
import pylim.Stats as plstat
from LMR_utils import class_docs_fixer
import LMR_gridded

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


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

    def __init__(self, lim_config, load_vars, var_groups, nelem_in_yr,
                 dobj_num_pcs, multivar_num_pcs, detrend, fcast_type, prior_map,
                 fcast_lead, match_prior, save_attrs, std_before_eof_vars=None,
                 store_calib=False, uncouple_groups=None):

        FcastVar = LMR_gridded.ForecasterVariable

        self.valid_data_mask = {}
        self.var_eofs = {}
        self.var_order = []
        self.var_span = {}
        self.calib_yr_range = {}
        self.grp_calib_eofs = {}
        self.grp_multivar_eof_stats = {}
        self.grp_calib_data = {}
        self.grp_span = {}
        self.var_eof_std_factor = {}
        self.var_std_factor = {}
        self.var_eof_stats = {}
        self._start = 0
        self._end = 0

        # Used for data concatenation
        self._pre_eof_data_std = defaultdict(list)

        # list of datatag, multivar_num_pcs, dobj_num_pcs, load_vars
        self._save_attrs = save_attrs

        for grp_key, (grp_nmodes, grp_var_list) in var_groups.items():

            fcast_var_gen = FcastVar.load_all_gen(lim_config, grp_var_list)

            for key, fcast_var in fcast_var_gen:

                # Convert data into pylim.DataObject and then perform processing
                dobj = self._process_forecast_variable(key, fcast_var, detrend,
                                                       nelem_in_yr,
                                                       dobj_num_pcs,
                                                       std_before_eof_vars)

                # Handle combination into LIMState
                self._process_lim_state_params(key, grp_key, dobj)

        self._combine_lim_state_data()
        eof_proj_calib = self._form_mvar_eof_state(var_groups, store_calib)

        if fcast_type == 'noise_integrate':
            fit_noise = True
        else:
            fit_noise = False

        # component indices to remove forward influence on other fields
        if uncouple_groups:
            uncouple_grp_ranges = [self.grp_span[key]
                                   for key in uncouple_groups]
        else:
            uncouple_grp_ranges = None

        # TODO: add in the component zeroing to G or L?
        self.lim = LIM.LIM(eof_proj_calib, nelem_in_tau1=nelem_in_yr,
                           fit_noise=fit_noise, max_neg_Qeval=40,
                           uncouple_ranges=uncouple_grp_ranges)
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

    @classmethod
    def from_config(cls, forecaster_config, state_var_keys,
                    psm_req_var_keys=None, store_calib=False):
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
        ignore_precalib = lim_cfg.ignore_precalib_lim
        save_precalib = lim_cfg.save_precalib_lim
        var_to_std_before_eof = lim_cfg.var_to_std_before_eof
        var_to_separate = lim_cfg.var_to_separate
        uncouple_groups = lim_cfg.uncouple_groups
        base_avg_interval = lim_cfg.avg_interval

        FcastVar = LMR_gridded.ForecasterVariable

        if match_prior:
            load_vars = state_var_keys
        else:
            # TODO: I'm not sure this works correcly with psm required variables
            load_vars = FcastVar.get_fcast_prior_match_vars(
                lim_cfg.fcast_varnames, lim_cfg.prior_mapping,
                state_var_keys)

        [var_load_groups,
         save_str_items] = cls._handle_config_var_separation(var_to_separate,
                                                             load_vars,
                                                             num_pcs,
                                                             psm_req_var_keys,
                                                             base_avg_interval)

        # Create file string using hashing
        save_attrs = [lim_cfg.datatag, str(dobj_num_pcs)]
        if uncouple_groups is not None and uncouple_groups:
            save_attrs += ['uncoupled', *uncouple_groups]
        save_attrs += save_str_items
        save_str = '_'.join(save_attrs)
        save_str = save_str.encode('utf-8')
        save_hasher = hashlib.md5()
        save_hasher.update(save_str)
        save_filename = save_hasher.hexdigest() + '.pkl'

        output_dir = lim_cfg.lim_precalib_dir
        output_path = os.path.join(output_dir, save_filename)

        try:
            if ignore_precalib:
                raise IOError('Ignore precalibrated LIM files specified.')

            with open(output_path, 'rb') as f:
                lim_obj = pickle.load(f)

            print('Successfully loaded pre-calibrated lim!')

        except IOError as e:
            print(e)
            lim_obj = cls(lim_cfg, load_vars, var_load_groups, nelem_in_yr,
                          dobj_num_pcs, num_pcs, detrend, fcast_type, prior_map,
                          fcast_lead, match_prior, save_attrs,
                          std_before_eof_vars=var_to_std_before_eof,
                          store_calib=store_calib,
                          uncouple_groups=uncouple_groups)

            if save_precalib:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(output_path, 'wb') as f:
                    print(f'Saving pre-calibrated LIM: {output_path}')
                    lim_obj._pre_eof_data_std = None
                    lim_obj._separate_vars = None
                    tmp_calib = lim_obj.grp_calib_data
                    lim_obj.grp_calib_data = None
                    pickle.dump(lim_obj, f)
                    lim_obj.grp_calib_data = tmp_calib

        lim_obj.print_lim_save_attrs()

        return lim_obj

    @classmethod
    def _handle_config_var_separation(cls, var_to_separate, all_load_keys,
                                      main_nmodes,
                                      psm_req_var_keys, avg_interval):
        """Process requested variables to separate into groups for the LIM"""
        if psm_req_var_keys is not None and psm_req_var_keys:
            all_load_keys += psm_req_var_keys

        all_load_keys = deepcopy(all_load_keys)
        sep_psms_key = 'psm_req_vars'

        grps_to_sep = {}
        not_assigned = set(all_load_keys)
        if sep_psms_key in var_to_separate:

            if psm_req_var_keys is None:
                raise ValueError('List of psm variable keys required if '
                                 'separation is requested')

            num_psm_grp_modes = var_to_separate.pop(sep_psms_key)
            psm_req_var_keys.sort()
            grps_to_sep[sep_psms_key] = (num_psm_grp_modes,
                                         psm_req_var_keys)
            not_assigned -= set(psm_req_var_keys)

        for grp_name, num_modes_ret in var_to_separate.items():
            if isinstance(num_modes_ret, int):
                # single variable
                var_key = (grp_name, avg_interval)
                not_assigned -= set(var_key)
                grps_to_sep[grp_name] = (num_modes_ret, [var_key])
                not_assigned -= {var_key}
            elif not isinstance(num_modes_ret, tuple):
                raise ValueError('var_to_separate should either contain an'
                                 'integer or tuple<int, list> to specify '
                                 'separation')
            else:
                num_modes_ret, grp_vars = num_modes_ret
                not_assigned -= set(grp_vars)

        not_assigned = list(not_assigned)
        not_assigned.sort()

        grps_to_sep['main'] = (main_nmodes, not_assigned)
        sep_save_str_items = cls._get_sep_var_string_items(grps_to_sep)

        return grps_to_sep, sep_save_str_items

    @staticmethod
    def _get_sep_var_string_items(var_to_sep):

        save_str_items = []
        for grp_key, (grp_nmodes, var_keys) in var_to_sep.items():
            save_str_items += ['_'.join([grp_key, str(grp_nmodes)])]
            var_key_strs = ['_'.join(item) for item in var_keys]
            save_str_items += var_key_strs

        # save_str_items.sort()
        # save_str = '_'.join(save_str_items)

        return save_str_items

    def forecast(self, state_obj):
        """Perfect no-noise forecast. Drastically reduces output variance."""

        # Translate state into LIM space for forecasting
        fcast_state, is_compressed = self.phys_space_data_to_fcast_space(state_obj)

        # Forecast data in LIM space
        fcast_data_trunc = self._fcast_func(fcast_state)

        # Translate data in LIM space back to physical space
        fcast_state_out = self.fcast_space_data_to_phys_space(fcast_data_trunc,
                                                              is_compressed)

        # restore original data (climatological) when forecast prior vars
        # do not necessarily match full state data
        if not self.match_prior:
            state_obj.restore_orig_state()

        # Overwrite forecasted state data
        state_obj.update_var_data(fcast_state_out)

        return state_obj

    def phys_space_data_to_fcast_space(self, state_obj, is_diff_model=False):
        fcast_state = []
        is_compressed = {}

        for var_key in self.var_order:
            var, avg_interval = var_key
            prior_var_key = self.prior_map[var]
            data = state_obj.get_var_data((prior_var_key, avg_interval))

            if np.any(np.isnan(data)):
                # Get the LIM calibration defined mask
                valid_mask = self.valid_data_mask[var_key]
                data = data[valid_mask, :]

                nan_check = np.isnan(data)
                if is_diff_model and np.any(nan_check):
                    # still NaN in the data
                    new_nan_mask = nan_check.sum(axis=1) > 0
                    new_valid_mask = np.logical_not(new_nan_mask)

                    data = data[new_valid_mask, :]
                else:
                    new_valid_mask = None

                is_compressed[var_key] = new_valid_mask
            else:
                new_valid_mask = None

            # Standardize prior to projection (helps for fields a lot of small
            #  values)
            if var_key in self.var_std_factor:
                data = data * self.var_std_factor[var_key]

            # Transposes sampling dimension and projects into EOF space
            eof_to_proj = self.var_eofs[var_key]
            if new_valid_mask is not None:
                eof_to_proj = eof_to_proj[new_valid_mask]

            eof_proj = np.dot(data.T, eof_to_proj)
            eof_proj_std = eof_proj * self.var_eof_std_factor[var_key]
            fcast_state.append(eof_proj_std)

        fcast_state = np.concatenate(fcast_state, axis=-1)
        fcast_state = fcast_state @ self.calib_eofs

        return fcast_state, is_compressed

    def fcast_space_data_to_phys_space(self, fcast_data, compressed_keys):
        # translate forecast out of state eof basis
        fcast_data = fcast_data @ self.calib_eofs.T

        # gather forecast back in original physical space from prior
        fcast_state_out = {}
        for var_key in self.var_order:
            var, avg_interval = var_key
            prior_var_key = self.prior_map[var]
            fcast_out = _get_var_from_limstate(var_key, fcast_data,
                                               self.var_span)
            fcast_out_un_std = fcast_out / self.var_eof_std_factor[var_key]
            phys_space_fcast = fcast_out_un_std @ self.var_eofs[var_key].T

            if var_key in self.var_std_factor:
                phys_space_fcast = phys_space_fcast / self.var_std_factor[
                    var_key]

            if var_key in compressed_keys:
                supplement_mask = compressed_keys[var_key]
                phys_space_fcast = self._decompress_field(var_key,
                                                          phys_space_fcast)

            fcast_state_out[(prior_var_key, avg_interval)] = phys_space_fcast.T

        return fcast_state_out

    def print_lim_save_attrs(self):
        datatag = self._save_attrs[0]
        dobj_npcs = int(self._save_attrs[1])
        load_vars = self._save_attrs[2:]

        dobj_eof_str = ''
        for var_key, eof_stats in self.var_eof_stats.items():
            tot_var_ret = eof_stats['var_expl_by_ret'] * 100
            dobj_eof_str += f'\t{var_key}: \t{tot_var_ret:3.1f}\n'

        mvar_eof_str = ''
        for grp_key, var_ret in self.grp_multivar_eof_stats.items():
            mvar_pct_ret = var_ret['var_expl_by_ret'] * 100
            mvar_eof_str += f'\t{grp_key}: \t{mvar_pct_ret:2.1f}\n'

        str = (f'\nCalibrated LIM Attributes:\n'
               f'==========================\n'
               f'data source: {datatag}\n'
               f'dobj num pcs: {dobj_npcs:3d}\n'
               f'dobj EOF percentage variance retained:\n'
               f'{dobj_eof_str}'
               f'percentage of multi-variate variance retained:\n'
               f'{mvar_eof_str}'
               f'included variable/avg_intervals: {load_vars}\n'
               f'==========================\n')

        print(str)

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
        out_arr = np.zeros((1441, *state_arr.shape), dtype=np.complex128)
        res = self.lim.noise_integration(state_arr, self.fcast_lead,
                                         timesteps=1440, out_arr=out_arr)
        avg = out_arr.real.mean(axis=0)
        return res

    def _decompress_field(self, key, data, supplement_mask=None):
        # Simplified pylim.DataObject.inflate_full_grid because I don't want to
        # store the entire calibration object for each field

        spatial_shp = list(data.shape[:-1])

        # Adjustment for fields from other models that may have different NaNs
        if supplement_mask is not None:
            tmp_data = np.empty(spatial_shp + [len(supplement_mask)]) * np.nan
            tmp_data[..., supplement_mask] = data
            data = tmp_data

        valid_data = self.valid_data_mask[key]
        
        new_data = np.empty((*data.shape[:-1], len(valid_data))) * np.nan
        new_data[..., valid_data] = data

        return new_data

    def _process_forecast_variable(self, key, fcast_var, detrend, nelem_in_yr,
                                   dobj_num_pcs, std_before_eof_vars):

        varname, avg_interval = key

        yr_start = fcast_var.time[0].year
        yr_end = fcast_var.time[-1].year
        self.calib_yr_range[key] = (yr_start, yr_end)
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

        if std_before_eof_vars is not None and varname in std_before_eof_vars:
            data_obj.standardize_data(save=True)
            self.var_std_factor[key] = data_obj._std_scaling
            proj_key = data_obj._STD

        # TODO: Fix cell area loading in LMR_gridded
        data_obj.area_weight_data(save=False)
        data_obj.eof_proj_data(dobj_num_pcs, proj_key=proj_key, save=True)
        self.var_eofs[key] = data_obj._eofs
        data_obj.calc_anomaly(nelem_in_yr, save=False)
        data_obj.standardize_data(save=False)

        self.var_eof_std_factor[key] = data_obj._std_scaling
        self.var_eof_stats[key] = data_obj.get_eof_stats()

        return data_obj

    def _process_lim_state_params(self, key, grp_key, data_obj):
        self.var_order.append(key)

        # Standardized data for calculating the multi-variate EOFs
        dobj_data_std = data_obj.data

        self._pre_eof_data_std[grp_key].append(dobj_data_std)
        self._end = self._start + dobj_data_std.shape[1]
        self.var_span[key] = (self._start, self._end)
        self._start = self._end

    def _combine_lim_state_data(self):

        late_start = -1e10
        early_end = 1e10
        for key, (yr_start, yr_end) in self.calib_yr_range.items():
            late_start = max(late_start, yr_start)
            early_end = min(early_end, yr_end)

        start_idx = 0
        for grp_key, concat_data in self._pre_eof_data_std.items():
            for i, var_data_std in enumerate(concat_data):

                key = self.var_order[i + start_idx]
                yr_start, yr_end = self.calib_yr_range[key]

                shave_start = late_start - yr_start
                shave_end = early_end - yr_end

                if shave_end == 0:
                    shave_end = None

                adj_slice = slice(shave_start, shave_end)

                concat_data[i] = var_data_std[adj_slice]

            start_idx += i + 1
            self._pre_eof_data_std[grp_key] = np.concatenate(concat_data, axis=1)

    def _form_mvar_eof_state(self, var_grps, store_calib):

        mvar_projected_data = []
        grp_span_start = 0
        for grp_key, (grp_neofs, grp_var_keys) in var_grps.items():
            grp_calib_state = self._pre_eof_data_std[grp_key]

            if len(grp_var_keys) > 1:
                grp_eof_proj = self._calc_mvar_eofs(grp_key, grp_calib_state,
                                                    grp_neofs)
            else:
                grp_eof_proj = self._handle_single_var_mvar_eofs(grp_key,
                                                                 grp_neofs,
                                                                 grp_calib_state)

            grp_span_end = grp_span_start + grp_eof_proj.shape[1]
            self.grp_span[grp_key] = (grp_span_start, grp_span_end)
            grp_span_start = grp_span_end

            mvar_projected_data.append(grp_eof_proj)
            if store_calib:
                self.grp_calib_data[grp_key] = grp_eof_proj

        mvar_projected_data = np.concatenate(mvar_projected_data, axis=1)

        # Combine EOFs into cohesive array
        # TODO: is this necessary any longer?
        i_len, j_len = 0, 0
        mvar_eof_lims = {}
        for grp_key, grp_eofs in self.grp_calib_eofs.items():
            i_shp, j_shp = grp_eofs.shape
            end_i = i_len + i_shp
            end_j = j_len + j_shp

            mvar_eof_lims[grp_key] = ((i_len, end_i), (j_len, end_j))
            i_len = end_i
            j_len = end_j

        combined_calib_eofs = np.zeros((i_len, j_len))
        for grp_key, grp_eofs in self.grp_calib_eofs.items():
            i_range, j_range = mvar_eof_lims[grp_key]
            combined_calib_eofs[slice(*i_range), slice(*j_range)] = grp_eofs
        self.calib_eofs = combined_calib_eofs

        return mvar_projected_data

    def _handle_single_var_mvar_eofs(self, grp_key, neofs, grp_calib_state):

        print(f'Handling multivar EOF translation for single var group: '
              f'{grp_key}')

        # don't need to do another EOF decomposition
        var_feature_len = grp_calib_state.shape[1]
        eof_translate = np.eye(var_feature_len)
        self.grp_calib_eofs[grp_key] = eof_translate[:, :neofs]
        eof_proj_calib = grp_calib_state[:, :neofs]

        return eof_proj_calib

    def _calc_mvar_eofs(self, grp_key, grp_calib_state, grp_neofs):

        # Calculate multi-variate EOFs
        [grp_state_eofs,
         grp_state_svals,
         grp_state_eof_stats] = _calc_limstate_eofs(grp_calib_state,
                                                    grp_neofs)

        self.grp_calib_eofs[grp_key] = grp_state_eofs
        ret_variance = grp_state_eof_stats['var_expl_by_ret'] * 100
        print(f'Variance % retained in multi-variate EOF truncation for '
              f'{grp_key} group: {ret_variance:3.1f}%')

        self.grp_multivar_eof_stats[grp_key] = grp_state_eof_stats

        # Project unstandardized data on the calculated EOFs
        eof_proj_calib = grp_calib_state @ grp_state_eofs

        return eof_proj_calib

    def _process_separate_vars(self, eof_proj_calib,
                               shave_yr_range, var_to_separate):

        """

        Parameters
        ----------
        eof_proj_calib: array-like
            The data to append the separated variables to
        shave_yr_range: tuple of years
            Year range from the coupled multi-variate state used to pare down
            the separate variables if necessary.
        var_to_separate: dict
            Mapping of the variable name to the number of standardized
            PCs to retain
        """

        new_lim_state = [eof_proj_calib]
        mvar_eof_shape = self.calib_eofs.shape
        var_eof_state_len = mvar_eof_shape[0]
        added_nmodes = 0

        for key, dobj in self._separate_vars.items():

            self.var_order.append(key)

            var_name, avg_interval = key
            num_eof_ret = var_to_separate[var_name]

            yr_start, yr_end = self.calib_yr_range[key]
            late_start, early_end = shave_yr_range
            shave_start = late_start - yr_start
            shave_end = early_end - yr_end

            if shave_end == 0:
                shave_end = None

            adj_slice = slice(shave_start, shave_end)

            # dobjs have been processed already in _process_fcast_variable
            data = dobj.data[adj_slice, :num_eof_ret]

            # add to eof proj calib
            new_lim_state.append(data)

            # add var range
            self._end = self._start + data.shape[1]
            added_nmodes += data.shape[1]
            self.var_span[key] = (self._start, self._end)
            self._start = self._end

            # adjust the dobj EOFs to omit left out modes
            curr_eofs = self.var_eofs[key]
            self.var_eofs[key] = curr_eofs[:, :num_eof_ret]

        # Adjust multivariate EOFs to a matrix
        # | Orig_mvar_eofs 0 |
        # |      0         I |
        new_mvar_eofs = np.zeros((var_eof_state_len + added_nmodes,
                                  mvar_eof_shape[1] + added_nmodes))
        new_mvar_eofs[:mvar_eof_shape[0], :mvar_eof_shape[1]] = self.calib_eofs
        new_mvar_eofs[mvar_eof_shape[0]:, mvar_eof_shape[1]:] = np.eye(added_nmodes)

        self.calib_eofs = new_mvar_eofs

        # Create mvar projected EOF state
        new_calib_state = np.concatenate(new_lim_state, axis=1)

        return new_calib_state


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
