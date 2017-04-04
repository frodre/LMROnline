"""
Class based config module to help with passing information to LMR modules for
paleoclimate reconstruction experiments.

NOTE:  All general user parameters that should be edited are displayed
       between the following sections:

       ##** BEGIN User Parameters **##

       parameters, etc.

       ##** END User Parameters **##

Adapted from LMR_exp_NAMELIST by AndreP
"""

from os.path import join


class constants:

    class file_types:
        netcdf = 'NCD'
        ascii = 'ASC'
        numpy = 'NPY'
        numpy_zip = 'NPZ'
        dataframe = 'DF'

    calib = {}
    calib['GISTEMP'] = {'fname': 'gistemp1200_ERSST.nc',
                       'varname': 'tempanomaly',
                       'type': 'NCD'}

    calib['HadCRUT'] = {'fname': 'HadCRUT.4.3.0.0.median.nc',
                        'varname': 'temperature_anomaly',
                        'type': 'NCD'}

    calib['BerkeleyEarth'] = {'fname': 'Land_and_Ocean_LatLong1.nc',
                              'varname': 'temperature',
                              'type': 'NCD'}

    calib['MLOST'] = {'fname': 'MLOST_air.mon.anom_V3.5.4.nc',
                      'varname': 'air',
                      'type': 'NCD'}

    calib['NOAA'] = {'fname': 'er-ghcn-sst.nc',
                     'varname': 'data',
                     'type': 'NCD'}

    prior = {}
    prior['ccsm4_last_millenium'] = \
        {'fname': '[vardef_template]_CCSM4_past1000_085001-185012.nc',
         'type': 'NCD',
         'state_vars': ['tas_sfc_Amon']}
    # ['tas_sfc_Amon', 'zg_500hPa_Amon', 'AMOCindex_Omon']

    prior['mpi-esm-p_last_millenium'] = \
        {'fname': '[vardef_template]_MPI-ESM-P_past1000_085001-185012.nc',
         'type': 'NCD',
         'state_vars': ['tas_sfc_Amon']}


class wrapper:
    """
    Parameters for reconstruction realization manager LMR_wrapper.

    Attributes
    ----------
    multi_seed: list(int), None
        List of RNG seeds to be used during a reconstruction for each
        realization.  This overrides the 'core.seed' parameter.
    param_search: dict{str: Iterable} or None
        Names of configuration parameters to iterate over when performing a
        reconstruction experiment

    Example
    -------
    The param_search dictionary should use the configuration object attribute
    syntax as the key to reference which parameters to iterate over. The
    value should be some iterable object of values to be covered in the
    parameter search::
        param_search = {'core.hybrid_a': (0.6, 0.7, 0.8),
                        'core.inf_factor': (1.1, 1.2, 1.3)}
    """

    ##** BEGIN User Parameters **##
    param_search = {'core.hybrid_a': (0.7, 0.8, 0.9)}
    multi_seed = [9271, 687, 4312, 7175, 4999, 3318, 3344, 3667, 6975, 1766, 7374, 1820,
                  2598, 1729, 9674, 3394, 239, 6039, 5670, 2679, 3334, 7684, 8701, 8719,
                  2767, 3988, 1341, 8734, 9880, 42, 2530, 6142, 5534, 1589, 7907, 8732, 5784,
                  1025, 6126, 6558, 3369, 8185, 9704, 6883, 9072, 7444, 9527, 1730, 567, 5294,
                  9677, 7105, 6497, 8558, 8651, 6829, 3944, 7014, 4166, 8141, 9964, 755, 872,
                  4372, 8599, 9030, 3291, 2659, 6914, 3874, 1227, 2239, 215, 9082, 1476, 2096,
                  1328, 5386, 6115, 5954, 3277, 8458, 6116, 3350, 7341, 1404, 8127, 9242, 2676,
                  5945, 3867, 4612, 810, 227, 422, 3830, 589, 2605, 8176, 7060]

    ##** END User Parameters **##

class core:
    """
    High-level parameters of reconstruction experiment

    Attributes
    ----------
    nexp: str
        Name of reconstruction experiment
    lmr_path: str
        Absolute path for the experiment
    online_reconstruction: bool
        Perform reconstruction with (True) or without (False) cycling
    persistence_forecast: bool
        If online is True, this flag defines whether to use a persistence (True)
        or LIM (False) forecast.
    clean_start: bool
        Delete existing files in output directory (otherwise they will be used
        as the prior!)
    recon_period: list(int)
        Time period for reconstruction
    seed: int, None
        RNG seed.  Passed to all random function calls. (e.g. Prior and proxy
        record sampling)  Overridden by wrapper.multi_seed.
    nens: int
        Ensemble size
    iter_range: list(int)
        Number of Monte-Carlo iterations to perform
    loc_rad: float
        Localization radius for DA (in km)
    assimilation_time_res: tup(float)
        Which resolution to assimilate data (in years)
    res_yr_shift: dict{float: float}
        Mapping dictionary for each assimilation resolution to a shifting
        coefficient in years. E.g. 0.5yr resolution could be shifted by 1/4
        year to roughly match with growing season
    hybrid_update: bool
        Use hybrid data assimilation technique for blending forecast and
        static information sources
    hybrid_a: float
        Blending coefficient between 0 and 1 for hybrid DA
    hybrid_blend_prior: bool
        Blend the prior state vector in addition to the covariance matrices.
        When this is True, the blending transitions the reconstruction from
        offline (a=0) to online(a=1)
    adaptive_inflate: bool
        DOES NOT CURRENTLY WORK
        Use EnKF adaptive inflation on the prior
    reg_inflate: bool
        Use ensemble variance inflation on the prior
    inf_factor: float
        Variance inflation factor to use when ``reg_inflate`` is True
    datadir_output: str
        Absolute path to working directory output for LMR
    archive_dir: str
        Absolute path to LMR reconstruction archive directory
    """

    ##** BEGIN User Parameters **##

    nexp = 'testdev_persistence'
    lmr_path = '/home/disk/chaos2/wperkins/data/LMR'
    online_reconstruction = True
    persistence_forecast = False
    clean_start = True
    ignore_pre_avg_file = False
    save_pre_avg_file = True
    # TODO: More pythonic to make last time a non-inclusive edge
    recon_period = [1950, 1960]
    nens = 10
    seed = None
    iter_range = [0, 0]
    curr_iter = iter_range[0]
    loc_rad = None
    assimilation_time_res = [1.0]  # in yrs
    # maps year shift (in years) to resolution
    res_yr_shift = {0.5: 0.25, 1.0: 0.0}

    # Forecasting Hybrid Update
    hybrid_update = True
    hybrid_update &= online_reconstruction
    hybrid_a = 0.85
    blend_prior = True

    # Adaptive Covariance Inflation
    adaptive_inflate = False
    reg_inflate = False
    inf_factor = 1.1

    # TODO: add rules for shift?
    # If shifting on smaller time scales than smallest time chunk it becomes
    # the base resolution
    sub_base_res = assimilation_time_res[0]
    for res, shift in res_yr_shift.iteritems():
        if (res in assimilation_time_res and
           shift < sub_base_res and
           shift != 0.0):
            sub_base_res = shift

    datadir_output = '/home/katabatic2/wperkins/LMR_output/working'

    archive_dir = '/home/katabatic2/wperkins/LMR_output/testing'

class proxies:
    """
    Parameters for proxy data

    Attributes
    ----------
    use_from: list(str)
        A list of keys for proxy classes to load from.  Keys available are
        stored in LMR_proxy2.
    proxy_frac: float
        Fraction of available proxy data (sites) to assimilate
    """

    ##** BEGIN User Parameters **##

    # =============================
    # Which proxy database to use ?
    # =============================
    use_from = ['pages']
    proxy_frac = 0.75

    ##** END User Parameters **##
    class pages:
        """
        Parameters for PagesProxy class

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data
        datafile_proxy: str
            Absolute path to proxy records file
        metafile_proxy: str
            Absolute path to proxy meta data
        dataformat_proxy: str
            File format of the proxy data
        regions: list(str)
            List of proxy data regions (data keys) to use.
        proxy_resolution: list(float)
            List of proxy time resolutions to use
        proxy_order: list(str):
            Order of assimilation by proxy type key
        proxy_assim2: dict{ str: list(str)}
            Proxy types to be assimilated.
            Uses dictionary with structure {<<proxy type>>: [.. list of measuremant
            tags ..] where "proxy type" is written as
            "<<archive type>>_<<measurement type>>"
        proxy_type_mapping: dict{(str,str): str}
            Maps proxy type and measurement to our proxy type keys.
            (e.g. {('Tree ring', 'TRW'): 'Tree ring_Width'} )
        simple_filters: dict{'str': Iterable}
            List mapping Pages2k metadata sheet columns to a list of values
            to filter by.
        """

        ##** BEGIN User Parameters **##

        datadir_proxy = join(core.lmr_path, 'data', 'proxies')
        # Pages 0.5yr resolution
        # datafile_proxy = join(datadir_proxy,
        #                       'Pages2k_Proxies_0pt5res.df.pckl')
        # metafile_proxy = join(datadir_proxy,
        #                       'Pages2k_Metadata_0pt5res.df.pckl')

        # Pages 1.0 yr res only
        datafile_proxy = join(datadir_proxy,
                              'Pages2k_Proxies.df.pckl')
        metafile_proxy = join(datadir_proxy,
                              'Pages2k_Metadata.df.pckl')
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']
        proxy_resolution = core.assimilation_time_res

        # DO NOT CHANGE FORMAT BELOW

        proxy_order = ['Tree ring_Width',
                       'Tree ring_Density',
                       'Ice core_d18O',
                       'Ice core_d2H',
                       'Ice core_Accumulation',
                       'Coral_d18O',
                       'Coral_Luminescence',
                       'Lake sediment_All',
                       'Marine sediment_All',
                       'Speleothem_All']

        proxy_assim2 = {
            'Tree ring_Width': ['Ring width',
                                'Tree ring width',
                                'Total ring width',
                                'TRW'],
            'Tree ring_Density': ['Maximum density',
                                  'Minimum density',
                                  'Earlywood density',
                                  'Latewood density',
                                  'MXD'],
            'Ice core_d18O': ['d18O'],
            'Ice core_d2H': ['d2H'],
            'Ice core_Accumulation': ['Accumulation'],
            'Coral_d18O': ['d18O'],
            'Coral_Luminescence': ['Luminescence'],
            'Lake sediment_All': ['Varve thickness',
                                  'Thickness',
                                  'Mass accumulation rate',
                                  'Particle-size distribution',
                                  'Organic matter',
                                  'X-ray density'],
            'Marine sediment_All': ['Mg/Ca'],
            'Speleothem_All': ['Lamina thickness'],
            }

        ##** END User Parameters **##

        # Create mapping for Proxy Type/Measurement Type to type names above
        proxy_type_mapping = {}
        for type, measurements in proxy_assim2.iteritems():
            # Fetch proxy type name that occurs before underscore
            type_name = type.split('_', 1)[0]
            for measure in measurements:
                proxy_type_mapping[(type_name, measure)] = type

        simple_filters = {'PAGES 2k Region': regions,
                          'Resolution (yr)': proxy_resolution}


class psm:
    """
    Parameters for PSM classes

    Attributes
    ----------
    use_psm: dict{str: str}
        Maps proxy class key to psm class key.  Used to determine which psm
        is associated with what Proxy type.
    """

    use_psm = {'pages': 'linear'}

    class linear:
        """
        Parameters for the linear fit PSM.

        Attributes
        ----------
        datatag_calib: str
            Source of calibration data for PSM
        datadir_calib: str
            Absolute path to calibration data
        datafile_calib: str
            Filename for calibration data
        dataformat_calib: str
            Data storage type for calibration data
        pre_calib_datafile: str
            Absolute path to precalibrated Linear PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """

        ##** BEGIN User Parameters **##

        datatag_calib = 'GISTEMP'
        sub_base_res = core.sub_base_res
        datadir_calib = join(core.lmr_path, 'data', 'analyses', datatag_calib)
        datafile_calib = constants.calib[datatag_calib]['fname']
        varname_calib = constants.calib[datatag_calib]['varname']
        dataformat_calib = constants.calib[datatag_calib]['type']

        ignore_pre_avg_file = core.ignore_pre_avg_file
        overwrite_pre_avg_file = core.save_pre_avg_file

        # pre_calib_datafile = join(core.lmr_path,
        #                           'PSM',
        #                           'PSMs_' + datatag_calib +
        #                           '_0pt5_1pt0_res.pckl')
        pre_calib_datafile = join(core.lmr_path,
                                  'PSM', 'test_psms',
                                  'PSMs_' + datatag_calib +
                                  '_1pt0res_1.00datfrac')
        psm_r_crit = 0.2
        min_data_req_frac = 1.0  # 0.0 no data required, 1.0 all data required
        ##** END User Parameters **##


class prior:
    """
    Parameters for the ensDA prior

    Attributes
    ----------
    prior_source: str
        Source of prior data
    datadir_prior: str
        Absolute path to prior data
    datafile_prior: str
        Name of prior file to use
    dataformat_prior: str
        Datatype of prior container
    truncate_state: bool
        Flag to truncate state vector to T42 spherical harmonic space
    backend_type: str
        Which backend to use for storing prior data during updates with
        shifted assimilation resolution.  Allowed flags are 'NPY' for numpy
        and 'H5' for HDF5 backends.
    state_variables: list(str)
        List of variables to use in the state vector for the prior
    """

    ##** BEGIN User Parameters **##


    # Prior data directory & model source
    prior_source = 'ccsm4_last_millenium'
    # prior_source = 'mpi-esm-p_last_millenium'

    datadir_prior = join(core.lmr_path, 'data', 'model', prior_source)
    datafile_prior   = constants.prior[prior_source]['fname']
    dataformat_prior = constants.prior[prior_source]['type']
    state_variables = constants.prior[prior_source]['state_vars']
    truncate_state = True
    backend_type = 'NPY'

    ##** END User Parameters **##

class forecaster:
    """
    Parameters for the online DA forecasting method.

    Attributes
    ----------
    use_forecaster: str
        Key of forecasting class to use for the current reconstruction.
    """

    # Which forecaster class to use
    use_forecaster = 'lim'

    class LIM:
        """
        calib_filename: Filename for LIM calibration data.  Should be netcdf
                        file or an HDF5 file from the DataTools.netcdf_to_hdf5_
                        container.
        calib_varname: Variable name to grab from calib_filename
        fcast_times: list(float)
            A list of lead times (in years) to forecast
        calib_is_anomaly: bool
            Flag if calibration data is in anomaly format
        calib_is_runmean: bool
            Flag if data has seasonal information removed. E.g. annual
            running mean was applied...
        wsize: int
            Window size for the annual running mean calculation.  Should be
            equivalent to the number of timesteps in a year for the source data.
        fcast_num_pcs: int
            Number of principle components to retain during LIM forecast
            calibration.
        detrend: bool
            Flag to detrend source data prior to calibration step.
        ignore_precalib: bool
            Ignore pre-calibrated LIM files
        use_ens_mean_fcast: bool
            Perform a forecast on the ensemble mean rather than each ensemble
            member
        eig_adjust: float
            CURRENTLY NOT USED.  Value to adjust eigenvalues of the forecast
            modes.
        """

        #calib_filename = ('/home/disk/chaos2/wperkins/data/LMR/data/model/20cr'
        #                  '/tas_sfc_Amon_20CR_185101-201112.nc')
        #calib_varname = 'tas'
        calib_filename = ('/home/disk/chaos2/wperkins/data/LMR/data/model'
                         '/ccsm4_last_millenium/'
                         'tas_sfc_Amon_CCSM4_past1000_085001-185012.nc')
        calib_varname = 'tas'
        # calib_filename = ('/home/disk/chaos2/wperkins/data/LMR/data/model'
        #                   '/mpi-esm-p_last_millenium/'
        #                   'tas_sfc_Amon_MPI-ESM-P_past1000_085001-185012.nc')
        # calib_varname = 'tas'

        # NOTE: for BerkeleyEarth data switch calib_is_anomaly and
        # calib_is_run_mean to TRUE
        #calib_filename = ('/home/disk/chaos2/wperkins/data/LMR/data/'
        #                  'analyses/Experimental/tas_run_mean_berkely_'
        #                  'earth_monthly_195701-201412.nc')
        #calib_varname = 'tas_run_mean'

        dataformat = 'NCD'
        calib_is_anomaly = False
        calib_is_runmean = False
        fcast_times = [1]
        wsize = 12
        fcast_num_pcs = 8
        detrend = True
        ignore_precalib = False
        use_ens_mean_fcast = False

        eig_adjust = None
