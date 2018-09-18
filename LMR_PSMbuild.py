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

import LMR_proxy2 as LMR_proxy
import LMR_config


# =========================================================================================
# START:  set user parameters here
# =========================================================================================

# USER NOTE:
# This script depends on defaults set in LMR_config for proxies and PSMs,
# if those values have changed it may not b

# Path to LMR data
lmr_path = '/home/katabatic/wperkins/data/LMR'

# Which proxy database to use for calibrating PSMs
use_from = ['LMRdb']
# use_from = ['PAGES2kv1']

# Where to output the pre-calibrated PSM file
output_dir = '/home/katabatic/wperkins/data/LMR/PSM/'

# Which type of PSM to create pre-calibrated data for
psm_type = 'linear'
# psm_type = 'bilinear'

# Use annual or seasonal distinctions to calibrate PSMs
avg_type = 'annual'
# avg_type = 'seasonal'

# Perform a series of calibrations using pre-defined seasons in order to
# determine an objective proxy seasonality definition.  Only used when
# avg_type='seasonal'.
test_proxy_seasonality = False

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
    season_source = None

load_psm_with_proxies = not test_proxy_seasonality


lmrdb_proxy_types = ['Bivalve_d18O', 'Corals and Sclerosponges_d18O',
                     'Corals and Sclerosponges_SrCa',
                     'Corals and Sclerosponges_Rates', 'Ice Cores_d18O',
                     'Ice Cores_dD', 'Ice Cores_Accumulation',
                     'Ice Cores_MeltFeature',
                     'Lake Cores_Varve', 'Lake Cores_BioMarkers',
                     'Lake Cores_GeoChem', 'Lake Cores_Misc',
                     'Marine Cores_d18O', 'Tree Rings_WidthBreit',
                     'Tree Rings_WidthPages2', 'Tree Rings_WoodDensity',
                     'Tree Rings_Isotopes', 'Speleothems_d18O']

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

psm_kwargs = {'calib_period': calib_period,
              'anom_reference_period': anom_reference_period,

              'linear': {'datatag': linear_datatag,
                         'avg_type': avg_type,
                         'season_source': season_source,
                         'psm_r_crit': 0.0},
              'bilinear': {'datatag_T': bilinear_datatag_T,
                           'datatag_P': bilinear_datatag_P,
                           'avg_type': avg_type,
                           'season_source': season_source,
                           'psm_r_crit': 0.0}}

regrid_kwargs = {'regrid_method': regrid_method,
                 'esmpy_regrid_to': regrid_grid,
                 'esmpy_interp_method': esmpy_interp_method}

psm_config = LMR_config.psm(lmr_path=lmr_path, proxy_use_from=use_from,
                            **psm_kwargs)
proxy_config = LMR_config.proxies(lmr_path=lmr_path, **proxy_kwargs)
regrid_config = LMR_config.regrid(**regrid_kwargs)

proxy_psm_seasonality_pages = {
    'Tree ring_Width': {'flag': True,
                        'seasons': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                    [6, 7, 8], [6, 7, 8, 9, 10, 11],
                                    [-12, 1, 2], [-12, 1, 2, 3, 4, 5]]},
    'Tree ring_Density': {'flag': True,
                          'seasons': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                      [6, 7, 8], [6, 7, 8, 9, 10, 11],
                                      [-12, 1, 2], [-12, 1, 2, 3, 4, 5]]},
}

proxy_psm_seasonality_lmrdb = {
    'Tree Rings_WidthBreit': {'flag':True,
                              'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],
                                            [6,7,8], [3,4,5,6,7,8],
                                            [6,7,8,9,10,11],[-12,1,2],
                                            [-9,-10,-11,-12,1,2],
                                            [-12,1,2,3,4,5]],
                              'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],
                                            [6,7,8],[3,4,5,6,7,8],
                                            [6,7,8,9,10,11],[-12,1,2],
                                            [-9,-10,-11,-12,1,2],
                                            [-12,1,2,3,4,5]]},
    'Tree Rings_WidthPages2': {'flag': True,
                               'seasons_T': [
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                   [6, 7, 8], [3, 4, 5, 6, 7, 8],
                                   [6, 7, 8, 9, 10, 11], [-12, 1, 2],
                                   [-9, -10, -11, -12, 1, 2],
                                   [-12, 1, 2, 3, 4, 5]],
                               'seasons_M': [
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                   [6, 7, 8], [3, 4, 5, 6, 7, 8],
                                   [6, 7, 8, 9, 10, 11], [-12, 1, 2],
                                   [-9, -10, -11, -12, 1, 2],
                                   [-12, 1, 2, 3, 4, 5]]},
    'Tree Rings_WoodDensity': {'flag': True,
                               'seasons_T': [
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                   [6, 7, 8], [3, 4, 5, 6, 7, 8],
                                   [6, 7, 8, 9, 10, 11], [-12, 1, 2],
                                   [-9, -10, -11, -12, 1, 2],
                                   [-12, 1, 2, 3, 4, 5]],
                               'seasons_M': [
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                   [6, 7, 8], [3, 4, 5, 6, 7, 8],
                                   [6, 7, 8, 9, 10, 11], [-12, 1, 2],
                                   [-9, -10, -11, -12, 1, 2],
                                   [-12, 1, 2, 3, 4, 5]]},
    'Tree Rings_Isotopes': {'flag': True,
                            'seasons_T': [
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                [6, 7, 8], [3, 4, 5, 6, 7, 8],
                                [6, 7, 8, 9, 10, 11], [-12, 1, 2],
                                [-9, -10, -11, -12, 1, 2],
                                [-12, 1, 2, 3, 4, 5]],
                            'seasons_M': [
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                [6, 7, 8], [3, 4, 5, 6, 7, 8],
                                [6, 7, 8, 9, 10, 11], [-12, 1, 2],
                                [-9, -10, -11, -12, 1, 2],
                                [-12, 1, 2, 3, 4, 5]]},
}


def save_calib_no_testing(proxies, psm_file, psm_file_diag):
    pids_by_ptype = defaultdict(list)
    psm_dict = defaultdict(dict)
    psm_dict_diag = {}

    for proxy in proxies:
        curr_ptype_list = pids_by_ptype[proxy.type]
        curr_ptype_list.append(proxy.id)

        curr_psm = proxy.psm_obj

        print('=>',
              "{:45s}".format(curr_psm.avg_interval),
              "{:12.4f}".format(curr_psm.slope),
              "{:12.4f}".format(curr_psm.intercept),
              "{:12.4f}".format(curr_psm.corr),
              "{:12.4f}".format(curr_psm.R),
              '(', "{:10.5f}".format(curr_psm.R2adj), ')')

        sitetag = (proxy.type, proxy.id)
        site_psm = psm_dict[sitetag]
        site_psm['lat'] = curr_psm.lat
        site_psm['lon'] = curr_psm.lon
        site_psm['elev'] = curr_psm.elev

        # selected PSM info into dictionary
        site_psm['Seasonality'] = proxy.seasonality
        site_psm['NbCalPts'] = curr_psm.NbPts
        site_psm['PSMintercept'] = curr_psm.intercept
        site_psm['PSMcorrel'] = curr_psm.corr
        site_psm['PSMmse'] = curr_psm.R

        if psm_type == 'linear':
            site_psm['calib'] = linear_datatag
            site_psm['PSMslope'] = curr_psm.slope
        elif psm_type == 'bilinear':
            site_psm['calib_temperature'] = bilinear_datatag_T
            site_psm['calib_moisture'] = bilinear_datatag_P
            site_psm['PSMslope_temperature'] = curr_psm.slope_temperature
            site_psm['PSMslope_moisture'] = curr_psm.slope_moisture
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
    with open(psm_path, 'wb') as f:
        pickle.dump(psm_dict, f, protocol=4)

    diag_path = os.path.join(output_dir, psm_file_diag)
    with open(diag_path, 'wb') as f:
        pickle.dump(psm_dict_diag, f, protocol=4)


def calib_seasonality_test(proxies):
    NotImplementedError()


def main():

    begin_time = time()

    proxy_database = proxy_config.use_from[0]

    print('Proxies             :', proxy_database)
    print('PSM type            :', psm_type)
    print('Calib. period       :', psm_config.calib_period)
    print('Anom. ref. period   :', psm_config.anom_reference_period)

    if not (proxy_database == 'PAGES2kv1' or proxy_database == 'LMRdb'):
        raise KeyError(f'Proxy database, {proxy_database}, is not a '
                       f'valid database key.')

    if psm_type == 'bilnear':
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

    proxy_class = LMR_proxy.get_proxy_class(use_from[0])
    proxies = proxy_class.load_all_annual_no_filtering(proxy_config,
                                                       psm_config,
                                                       regrid_config)

    save_calib_no_testing(proxies, psm_file, psm_file_diag)

    end_time = time() - begin_time
    print('=========================================================')
    print('PSM calibration completed in '+ str(end_time/60.0)+' mins')
    print('=========================================================')

# =============================================================================


if __name__ == '__main__':
    main()
