"""
 Module: LMR_PSMbuild.py
 
   Stand-alone tool building linear forward models (Proxy System Models) relating surface
   temperature and/or moisture to various proxy measurements, through linear regression
   between proxy chronologies and historical gridded surface analyses.
   This updated version uses the Pandas DataFrame version of the proxy database and
   metadata, and can therefore be used on the PAGES2kS1 and NCDC pandas-formatted 
   proxy datafiles.
 
 Originator : Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                            | January 2016
"""
import os
import numpy as np
import pickle
import datetime
from time import time
from copy import deepcopy
from collections import defaultdict
from itertools import product


import LMR_proxy as LMR_proxy
import LMR_config
import LMR_gridded
import LMR_psms
from LMR_utils import PSMTooFewObsError, PSMFitThresholdError


# =========================================================================================
# START:  set user parameters here
# =========================================================================================

# USER NOTE:
# This script depends on defaults set in LMR_config for proxies and PSMs,
# if those values have changed it may not b

# Path to LMR data
lmr_path = '/home/katabatic/wperkins/data/LMR'

# Which proxy database to use for calibrating PSMs
use_from = 'LMRdb'
# use_from = 'PAGES2kv1'

# Where to output the pre-calibrated PSM file
output_dir = '/home/katabatic/wperkins/data/LMR/PSM/'

# Which type of PSM to create pre-calibrated data for
psm_type = 'linear'
# psm_type = 'bilinear'

# Use annual or seasonal distinctions to calibrate PSMs
# avg_type = 'annual'
avg_type = 'seasonal'

# Fraction of data required over the averaging interval to be a valid average
# 1.0 requires all data be non-NaN, 0.0 means no requirements
min_data_req_frac = 1.0

# Perform a series of calibrations using pre-defined seasons in order to
# determine an objective proxy seasonality definition.  Only used when
# avg_type='seasonal'.
test_proxy_seasonality = True

# Years over which calibration and proxy data are considered for fits
calib_period = (1850, 2015)

# The anomaly reference period that calibration data is centered to before a
# fit. This sets the anomaly reference for any reconstruction using a given
# PSM.
anom_reference_period = (1951, 1980)

# Dataset used as calibration data for linear PSM. See datasets.yml for
# dataset options
linear_datatag = 'GISTEMP'

# Datasets used as calibration for bilinear PSM.
bilinear_datatag_T = 'GISTEMP'
bilinear_datatag_P = 'GPCC'

# Regridding options, use None for no regridding
regrid_method = 'esmpy'

# Target grid to regrid to as defined in 'grid_def.yml'
regrid_grid = 'reg_2x2deg'

# ESMpy regridding interpolation method
esmpy_interp_method = 'bilinear'

# ===
# END: user parameters
# ===

if avg_type == 'seasonal':
    if test_proxy_seasonality:
        season_source = 'psm_calib'
    else:
        season_source = 'proxy_metadata'
else:
    test_proxy_seasonality = False
    season_source = None

load_psm_with_proxies = not test_proxy_seasonality


lmrdb_proxy_types = [
            'Tree Rings_WidthPages2',
            'Tree Rings_WidthBreit',
            'Tree Rings_Isotopes',
            'Tree Rings_Temperature',
            'Tree Rings_WoodDensity',
            'Corals and Sclerosponges_d18O',
            'Corals and Sclerosponges_SrCa',
            'Corals and Sclerosponges_Rates',
            # 'Corals and Sclerosponges_Composite',
            # 'Corals and Sclerosponges_Temperature',
            'Ice Cores_d18O',
            'Ice Cores_dD',
            'Ice Cores_Accumulation',
            'Ice Cores_MeltFeature',
            'Lake Cores_Varve',
            'Lake Cores_BioMarkers',
            'Lake Cores_GeoChem',
            'Lake Cores_Misc',
            'Marine Cores_d18O',
            'Marine Cores_tex86',
            'Marine Cores_uk37',
            'Bivalve_d18O',
            'Speleothems_d18O']

pages_proxy_types = ['Tree ring_Width', 'Tree ring_Density', 'Ice core_d18O',
                     'Ice core_d2H', 'Ice core_Accumulation', 'Coral_d18O',
                     'Coral_Luminescence', 'Lake sediment_All',
                     'Marine sediment_All', 'Speleothem_All']

lmrdb_psm_map = {proxy_type: psm_type for proxy_type in lmrdb_proxy_types}
pages_psm_map = {proxy_type: psm_type for proxy_type in pages_proxy_types}

proxy_kwargs = {'use_from': use_from,
                'proxy_frac': 1.0,
                'load_psm_with_proxies': load_psm_with_proxies,
                'on_the_fly_calib': True,
                'proxy_availability_filter': False,

                'PAGES2kv1': {'proxy_psm_type': pages_psm_map},
                'LMRdb': {'proxy_psm_type': lmrdb_psm_map}}

psm_cfg_kwargs = {'calib_period': calib_period,
                  'anom_reference_period': anom_reference_period,

                  'linear': {'datatag': linear_datatag,
                             'ignore_pre_calib': True,
                             'avg_type': avg_type,
                             'season_source': season_source,
                             'psm_r_crit': 0.0,
                             'min_data_req_frac': min_data_req_frac},
                  'bilinear': {'datatag_T': bilinear_datatag_T,
                               'datatag_P': bilinear_datatag_P,
                               'ignore_pre_calib': True,
                               'avg_type': avg_type,
                               'season_source': season_source,
                               'psm_r_crit': 0.0,
                               'min_data_req_frac': min_data_req_frac}}

regrid_kwargs = {'regrid_method': regrid_method,
                 'esmpy_regrid_to': regrid_grid,
                 'esmpy_interp_method': esmpy_interp_method}

test_seasons = ['annual_std', 'jja', 'jjason', 'djf', 'djfmam',
                'sh_growing', 'nh_growing']
default_season = 'annual_std'

if use_from == 'PAGES2kv1':
    test_season_proxy_types = [
    #    'Tree ring_Width',
        'Tree ring_Density'
    ]
else:
    test_season_proxy_types = ['Tree Rings_WidthBreit',
                               'Tree Rings_WidthPages2',
                               'Tree Rings_WoodDensity',
                               'Tree Rings_Isotopes']


def save_calib_no_testing(proxies, psm_file, psm_file_diag, psm_type):
    pids_by_ptype = defaultdict(list)
    psm_dict = defaultdict(dict)
    psm_dict_diag = {}

    for proxy in proxies:
        curr_ptype_list = pids_by_ptype[proxy.type]
        curr_ptype_list.append(proxy.id)

        curr_psm = proxy.psm_obj

        sitetag = (proxy.type, proxy.id)
        site_psm = psm_dict[sitetag]
        site_psm['lat'] = curr_psm.lat
        site_psm['lon'] = curr_psm.lon
        site_psm['elev'] = curr_psm.elev

        # selected PSM info into dictionary
        site_psm['NbCalPts'] = curr_psm.NbPts
        site_psm['PSMintercept'] = curr_psm.intercept
        site_psm['PSMcorrel'] = curr_psm.corr
        site_psm['PSMmse'] = curr_psm.R

        if psm_type == 'linear':
            site_psm['Seasonality'] = curr_psm.seasonality
            site_psm['avg_interval'] = curr_psm.avg_interval
            site_psm['calib'] = linear_datatag
            site_psm['PSMslope'] = curr_psm.slope

            print('=>',
                  "{:20s}\t".format(proxy.id),
                  "a={:12.4f}\t".format(curr_psm.slope),
                  "b={:12.4f}\t".format(curr_psm.intercept),
                  "corr={:12.4f}\t".format(curr_psm.corr),
                  "MSE={:12.4f}\t".format(curr_psm.R),
                  '(', "{:10.5f}".format(curr_psm.R2adj), ')')

        elif psm_type == 'bilinear':
            site_psm['Seasonality_T'] = curr_psm.seasonality_T
            site_psm['Seasonality_P'] = curr_psm.seasonality_P
            site_psm['avg_interval_T'] = curr_psm.avg_interval_T
            site_psm['avg_interval_P'] = curr_psm.avg_interval_P
            site_psm['calib_temperature'] = bilinear_datatag_T
            site_psm['calib_moisture'] = bilinear_datatag_P
            site_psm['PSMslope_temperature'] = curr_psm.slope_temperature
            site_psm['PSMslope_moisture'] = curr_psm.slope_moisture

            print('=>',
                  "{:20s}\t".format(proxy.id),
                  "a1={:12.4f}\t".format(curr_psm.slope_temperature),
                  "a2={:12.4f}\t".format(curr_psm.slope_moisture),
                  "b={:12.4f}\t".format(curr_psm.intercept),
                  "corr={:12.4f}\t".format(curr_psm.corr),
                  "MSE={:12.4f}\t".format(curr_psm.R),
                  '(', "{:10.5f}".format(curr_psm.R2adj), ')')
        else:
            raise KeyError('Unrecognized psm type key: {}'.format(psm_type))

        site_psm['PSMintercept'] = curr_psm.intercept
        site_psm['fitBIC'] = curr_psm.BIC
        site_psm['fitR2adj'] = curr_psm.R2adj

        # diagnostic information

        # copy main psm attributes
        psm_dict_diag[sitetag] = deepcopy(psm_dict[sitetag])
        site_diag = psm_dict_diag[sitetag]
        # add diagnostics
        site_diag['calib_time'] = curr_psm.calib_time
        site_diag['calib_proxy_values'] = curr_psm.calib_proxy_values
        site_diag['calib_fit_values'] = curr_psm.calib_proxy_fit

        if psm_type == 'linear':
            site_diag['calib_refer_values'] = curr_psm.calib_refer_values
        else:
            site_diag['calib_temperature_refer_values'] = \
                curr_psm.calib_temperature_refer_values
            site_diag['calib_moisture_refer_values'] = \
                curr_psm.calib_moisture_refer_values

    # Summary of calibrated proxy sites
    # ---------------------------------
    calibrated_sites = list(psm_dict.keys())
    calibrated_types = list(set([item[0] for item in calibrated_sites]))

    print('-------------------------------------------------------------------')
    print('Calibrated proxies : counts per proxy type:')
    # count the total number of proxies
    total_proxy_count = len(calibrated_sites)

    for ptype in sorted(calibrated_types):
        plist = [item[1] for item in calibrated_sites if item[0] == ptype]
        print('%45s : %5d' % (ptype, len(plist)))
    print('-------------------------------------------------------------------')
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print('-------------------------------------------------------------------')

    # Dump dictionaries to pickle files
    psm_path = os.path.join(output_dir, psm_file)
    print('Saving pre-calibrated psm file to: {}'.format(psm_path))
    with open(psm_path, 'wb') as f:
        pickle.dump(psm_dict, f, protocol=4)

    diag_path = os.path.join(output_dir, psm_file_diag)
    with open(diag_path, 'wb') as f:
        pickle.dump(psm_dict_diag, f, protocol=4)


def load_analysis_var(seasonality, avg_kwargs, psm_config):
    datatag = psm_config.datatag
    AnalysisVariable = LMR_gridded.get_analysis_var_class(datatag)
    psm_config.update_avg_interval(seasonality, avg_kwargs)
    return AnalysisVariable.load(psm_config, anomaly=True)


def get_calib_objects(psm_config, season_defs, seasonality_list):

    calib_objs = {}
    for seasonality in seasonality_list:
        avg_kwargs = season_defs[seasonality]
        analysis_var = load_analysis_var(seasonality, avg_kwargs, psm_config)
        calib_objs[seasonality] = analysis_var

    return calib_objs


def load_proxy_seasonal_calibs(proxies, seasonality_list, psm_config):

    proxy_calib_objs = {}
    for proxy in proxies:
        seasonality = proxy.seasonality
        [avg_interval,
         avg_int_kwarg] = psm_config.handle_proxy_elem_list(seasonality)

        if avg_interval not in seasonality_list:
            calib_obj = _load_proxy_calib(avg_interval, avg_int_kwarg,
                                          psm_config)
            proxy_calib_objs[proxy.id] = {avg_interval: calib_obj}

    return proxy_calib_objs


def _load_proxy_calib(seasonality, avg_kwargs, psm_config):

    if psm_type == 'bilinear':
        specific_config = psm_config.bilinear
        load_cfgs = [specific_config.temperature,
                     specific_config.moisture]
    else:
        specific_config = psm_config.linear
        load_cfgs = [specific_config]

    calibs = [load_analysis_var(seasonality, avg_kwargs, cfg)
              for cfg in load_cfgs]

    return calibs


def _get_bil_arg_combos(temp_objs, moist_objs):
    test_combos = {}
    for t_key, m_key in product(temp_objs.keys(), moist_objs.keys()):
        comb_key = (t_key, m_key)
        comb_val = {'avg_key_T': t_key,
                    'calib_obj_T': temp_objs[t_key],
                    'avg_key_P': m_key,
                    'calib_obj_P': moist_objs[m_key]}
        test_combos[comb_key] = comb_val

    return test_combos


def _get_lin_arg_combos(objs):
    test_combos = {}
    for t_key, calib_obj in objs.items():
        test_combos[t_key] = {'avg_key': t_key,
                              'calib_obj': calib_obj}

    return test_combos


def calib_seasonality_test(proxies, psm_config, seasonality_list, psm_type):

    season_defs = psm_config._avg_def_constants

    if psm_type == 'bilinear':
        specific_config = psm_config.bilinear
        temp_objs = get_calib_objects(specific_config.temperature,
                                      season_defs,
                                      seasonality_list)
        moist_objs = get_calib_objects(specific_config.moisture,
                                       season_defs,
                                       seasonality_list)

        combo_func = _get_bil_arg_combos
        test_combos = combo_func(temp_objs, moist_objs)

    else:
        specific_config = psm_config.linear
        temp_objs = get_calib_objects(specific_config,
                                      season_defs,
                                      seasonality_list)
        combo_func = _get_lin_arg_combos
        test_combos = combo_func(temp_objs)

    # Seasonal calibrations for specific proxies that aren't in the test list
    proxy_req_objs = load_proxy_seasonal_calibs(proxies, seasonality_list,
                                                psm_config)

    psm_class = LMR_psms.get_psm_class(psm_type)
    valid_proxies_with_psm = []

    for proxy in proxies:
        if proxy.id in proxy_req_objs:
            proxy_calib_dict = proxy_req_objs[proxy.id]
            avg_key = list(proxy_calib_dict.keys())[0]
            pcalib_objs = proxy_calib_dict[avg_key]

            # Create new test combination
            if len(pcalib_objs) > 1:
                t_obj, m_obj = pcalib_objs
                combined_temp = {avg_key: t_obj}
                combined_temp.update(temp_objs)
                combined_moisture = {avg_key: m_obj}
                combined_moisture.update(moist_objs)
                test_combos = combo_func(combined_temp, combined_moisture)
            else:
                new_calib_dict = {avg_key: pcalib_objs[0]}
                new_calib_dict.update(temp_objs)
                test_combos = combo_func(new_calib_dict)

        num_to_compare = len(test_combos)
        test_psms = {}
        psm_compare_metric = np.zeros(num_to_compare) * np.nan
        idx_to_key = {}

        for i, (key, psm_kwargs) in enumerate(test_combos.items()):
            idx_to_key[i] = key
            try:
                psm_obj = psm_class(psm_config, proxy,
                                    on_the_fly_calib=True, **psm_kwargs)
                test_psms[key] = psm_obj
                psm_compare_metric[i] = psm_obj.BIC
            except (PSMFitThresholdError, PSMTooFewObsError) as e:
                print('Could not calibrate test season(s) {} for {}'
                      ''.format(key, proxy.id))
                pass

        if np.any(np.isfinite(psm_compare_metric)):
            min_idx = np.nanargmin(psm_compare_metric)
            best_key = idx_to_key[min_idx]
            proxy.psm_obj = test_psms[best_key]
            valid_proxies_with_psm.append(proxy)

    return valid_proxies_with_psm


def calib_default_seasonality(proxies, psm_config, psm_type,
                              default_season):

    season_defs = psm_config._avg_def_constants

    psm_class = LMR_psms.get_psm_class(psm_type)

    valid_proxies_with_psm = []
    for proxy in proxies:

        seasonality = proxy.seasonality
        avg_interval, avg_kwargs = psm_config.handle_proxy_elem_list(seasonality)

        if psm_type == 'bilinear':
            temp_objs = get_calib_objects(psm_config.bilinear.temperature,
                                          season_defs, [avg_interval])
            avg_key_T, cobj_T = temp_objs.popitem()

            moist_objs = get_calib_objects(psm_config.bilinear.moisture,
                                           season_defs, [avg_interval])
            avg_key_P, cobj_P = moist_objs.popitem()

            psm_kwargs = {'avg_key_T': avg_key_T,
                          'calib_obj_T': cobj_T,
                          'avg_key_P': avg_key_P,
                          'calib_obj_P': cobj_P}
        else:
            temp_objs = get_calib_objects(psm_config.linear,
                                          season_defs, [avg_interval])
            avg_key_T, cobj_T = temp_objs.popitem()
            psm_kwargs = {'avg_key': avg_key_T,
                          'calib_obj': cobj_T}

        try:
            psm_obj = psm_class(psm_config, proxy,
                                on_the_fly_calib=True, **psm_kwargs)
            proxy.psm_obj = psm_obj
            valid_proxies_with_psm.append(proxy)
        except (PSMFitThresholdError, PSMTooFewObsError) as e:
            print(e)

    return valid_proxies_with_psm


def main():
    regrid_config = LMR_config.regrid(**regrid_kwargs)
    psm_config = LMR_config.psm(regrid_config, lmr_path=lmr_path,
                                proxy_use_from=use_from,
                                **psm_cfg_kwargs)
    proxy_config = LMR_config.proxies(lmr_path=lmr_path,
                                      **proxy_kwargs)

    begin_time = time()

    proxy_database = proxy_config.use_from

    print('Proxies             :', proxy_database)
    print('PSM type            :', psm_type)
    print('Calib. period       :', psm_config.calib_period)
    print('Anom. ref. period   :', psm_config.anom_reference_period)

    if not (proxy_database == 'PAGES2kv1' or proxy_database == 'LMRdb'):
        raise KeyError(f'Proxy database, {proxy_database}, is not a '
                       f'valid database key.')

    if psm_type == 'bilinear':
        psm_file = psm_config.bilinear.pre_calib_datafile
    elif psm_type == 'linear':
        psm_file = psm_config.linear.pre_calib_datafile
    else:
        raise KeyError(f'Designated psm_type, {psm_type}, is not a valid key.')

    # corresponding file containing complete diagnostics
    psm_file_diag = psm_file.replace('.pckl', '_diag.pckl')

    # Check if psm_file already exists, archive it with current date/time if
    # it exists and replace by new file
    if os.path.isfile(psm_file):        
        nowstr = datetime.datetime.now().strftime("%Y%m%d:%H%M")
        no_file_ext = psm_file.rstrip('.pckl')
        command = 'mv {} {}_{}.pckl'.format(psm_file, no_file_ext, nowstr)
        os.system(command)
        if os.path.isfile(psm_file_diag):
            diag_no_file_ext = psm_file_diag.rstrip('.pckl')
            command = 'mv {} {}_{}.pckl'.format(psm_file_diag,
                                                diag_no_file_ext,
                                                nowstr)
            os.system(command)

    proxy_class = LMR_proxy.get_proxy_class(use_from)

    proxies = proxy_class.load_all_annual_no_filtering(proxy_config, psm_config)

    if test_proxy_seasonality:

        test_proxies = []
        non_test_proxies = []
        for proxy in proxies:
            if proxy.type in test_season_proxy_types:
                test_proxies.append(proxy)
            else:
                non_test_proxies.append(proxy)

        test_proxies = calib_seasonality_test(test_proxies, psm_config,
                                              test_seasons, psm_type)

        non_test_proxies = calib_default_seasonality(non_test_proxies,
                                                     psm_config,
                                                     psm_type,
                                                     default_season)
        proxies = test_proxies + non_test_proxies

    save_calib_no_testing(proxies, psm_file, psm_file_diag, psm_type)

    end_time = time() - begin_time
    print('=========================================================')
    print('PSM calibration completed in '+ str(end_time/60.0)+' mins')
    print('=========================================================')

# =============================================================================


if __name__ == '__main__':
    main()
