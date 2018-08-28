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
from copy import deepcopy
import yaml
import numpy as np
import os

from LMR_utils2 import get_averaging_period

# If true, uses only LMR_config.  No yaml loading
LEGACY_CONFIG = False

# Absolute path to LMR source code directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Control logging output. (0 = none; 1 = most important; 2 = many; 3 = a lot;
#   >=4 all)
LOG_LEVEL = 1

# Class for distinction of configuration classes
class ConfigGroup(object):

    def __init__(self, **kwargs):
        if kwargs:
            update_config_attrs_yaml(kwargs, self)


class _YamlStorage(object):
    """
    Generic object for loading in dictionaries from yaml files, 
    and a convenience function for returning loaded information.
    """

    def __init__(self, filename):
        print('Loading information from {}'.format(filename))
        try:
            f = open(join(SRC_DIR, filename), 'r')
            self.data = yaml.load(f)
        except IOError as e:
            raise SystemExit(('Could not load {} file when initializing the '
                              ' in LMR_config.py. File is required for locating'
                              ' dataset specific files. Please ensure '
                              'LMR_Config.SRC_DIR is correctly set. '
                              'Exiting...').format(filename))

    def get_info(self, tag):
        return dict(self.data[tag])


# TODO: I think these can just be a function with a special one for constants
class _DatasetDescriptors(_YamlStorage):
    """
    Loads and stores the datasets.yml file and return dictionaries of file
    specifications for each dataset. Information used by psm, prior,
    and forecaster configuration classes.
    """

    def __init__(self, filename=join(SRC_DIR, 'datasets.yml')):
        super(_DatasetDescriptors, self).__init__(filename)


class _GridDefinitions(_YamlStorage):
    """
    Loads and stores the grid_def.yml file and returns dictionaries of 
    information necessary to construct a given grid in ESMpy regridding.
    """

    def __init__(self, filename=join(SRC_DIR, 'grid_def.yml')):
        super(_GridDefinitions, self).__init__(filename)


class _ConstantDefinitions(_YamlStorage):
    """Stores constant information for LMR reconstructions"""

    def __init__(self, filename=join(SRC_DIR, 'constants.yml')):
        super(_ConstantDefinitions, self).__init__(filename)

        # Convert month indices to more machine friendly 0-indexed non-negative
        # values
        for key, val in self.data['avg_interval'].items():
            avg_indices = val['elem_to_avg']
            nelem_in_yr = val['nelem_in_yr']

            avg_indices = get_averaging_period(avg_indices, nelem_in_yr)

            avg_indices = np.array(avg_indices, dtype=np.int16)

            val['elem_to_avg'] = avg_indices


# Load dataset information on configuration import
_DataInfo = _DatasetDescriptors()
_GridDef = _GridDefinitions()

# TODO: these should probably be attached to config objects that will need them
Constants = _ConstantDefinitions()


class wrapper(ConfigGroup):
    """
    Parameters for reconstruction realization manager LMR_wrapper.

    Attributes
    ----------
    multi_seed: list(int), None
        List of RNG seeds to be used during a reconstruction for each
        realization.  This overrides the 'core.seed' parameter.
    iter_range: tuple(int)
        Range of Monte-Carlo iterations to perform
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

    iter_range = (0, 0)
    param_search = None
    multi_seed = None

    ##** END User Parameters **##

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        if self.multi_seed is not None:
            self.multi_seed = list(self.multi_seed)
        self.iter_range = self.iter_range
        self.param_search = deepcopy(self.param_search)

class core(ConfigGroup):
    """
    High-level parameters of LMR_driver_callable.

    Notes
    -----
    curr_iter attribute is created during initialization

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
    use_precalc_ye: bool
        Use pre-existing files for the psm Ye values.  If the file does not
        exist and the required state variables are missing the reconstruction
        will quit.
    recon_period: tuple(int)
        Time period for reconstruction
    nens: int
        Ensemble size
    loc_rad: float
        Localization radius for DA (in km)
    inflation_fact : float
        Covariance inflation factor
    seed: int, None
        RNG seed.  Passed to all random function calls. (e.g. prior and proxy
        record sampling)  Overridden by wrapper.multi_seed.
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
    write_posterior_Ye: bool
        Flag to indicate whether the analysis_Ye.pckl is to be generated 
        or not (large file containing full information on the posterior 
        proxy estimates (assimilated proxy records).

    """

    ##** BEGIN User Parameters **##

    nexp = 'testdev_persistence'
    lmr_path = '/home/disk/katabatic2/wperkins/cp_lim_archive/LMR_slim'
    datadir_output = '/home/katabatic2/wperkins/LMR_output/working'
    archive_dir = '/home/katabatic2/wperkins/LMR_output/testing'

    # Whether or not to produce the analysis_Ye.pckl file
    write_posterior_Ye = False

    online_reconstruction = False
    # TODO: Move this to forecaster config
    persistence_forecast = False
    clean_start = True
    use_precalc_ye = False
    ignore_pre_avg_file = False
    save_pre_avg_file = True
    recon_period = [1950, 1960]
    nens = 10
    recon_timescale = 1  # annual
    seed = None

    loc_rad = None

    # Forecasting Hybrid Update
    hybrid_update = True
    hybrid_update &= online_reconstruction
    hybrid_a = 0.85
    blend_prior = True

    # Adaptive Covariance Inflation
    adaptive_inflate = False
    reg_inflate = False
    inflation_factor = 1.1

    # Ensemble archiving options: ens_full, ens_variance, ens_percentiles, ens_subsample
    save_archive = 'ens_variance'
    # if save_archive = 'ens_percentiles', which percentiles to caclulate and save
    save_archive_percentiles = (5,95)
    # if save_archive = 'ens_subsample', number of members to archive
    save_archive_ens_subsample = 10

    # Possibly regrid the generated reanalysis fields to archive files.
    # Options: None (no regridding) or 'esmpy'
    archive_regrid_method = None
    archive_esmpy_interp_method = 'bilinear'
    archive_esmpy_regrid_to = 't42'

    
    ##** END User Parameters **##

    def __init__(self, curr_iter=None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.recon_timescale = int(self.recon_timescale)

        self.nexp = self.nexp
        self.lmr_path = self.lmr_path
        self.datadir_output = self.datadir_output
        self.archive_dir = self.archive_dir
        self.write_posterior_Ye = self.write_posterior_Ye
        self.save_full_field = self.save_full_field
        self.online_reconstruction = self.online_reconstruction
        self.persistence_forecast = self.persistence_forecast
        self.clean_start = self.clean_start
        self.use_precalc_ye = self.use_precalc_ye
        self.ignore_pre_avg_file = self.ignore_pre_avg_file
        self.save_pre_avg_file = self.save_pre_avg_file
        self.recon_period = tuple(self.recon_period)
        self.nens = self.nens
        self.recon_timescale = self.recon_timescale
        self.seed = self.seed
        self.loc_rad = self.loc_rad
        self.inflation_factor = self.inflation_factor
        self.hybrid_update = self.hybrid_update
        self.hybrid_a = self.hybrid_a
        self.blend_prior = self.blend_prior
        self.adaptive_inflate = self.adaptive_inflate
        self.reg_inflate = self.reg_inflate
        self.inflation_factor = self.inflation_factor
        self.datadir_output = self.datadir_output
        self.archive_dir = self.archive_dir
        self.write_posterior_Ye = self.write_posterior_Ye
        self.anom_reference_period = self.anom_reference_period


        self.save_archive = self.save_archive
        self.save_archive_percentiles = self.save_archive_percentiles
        self.save_archive_ens_subsample = self.save_archive_ens_subsample


        if self.archive_regrid_method is not None:
            if self.archive_regrid_method == 'esmpy':
                self.archive_esmpy_interp_method = self.archive_esmpy_interp_method
                self.archive_esmpy_grid_def = _GridDef.get_info(self.archive_esmpy_regrid_to)
            else:
                raise ValueError('Unrecognized option for regridding to archive files!'
                                 ' Only None or esmpy are allowed.')

        if curr_iter is None:
            self.curr_iter = wrapper.iter_range[0]
        else:
            self.curr_iter = curr_iter


class proxies(ConfigGroup):
    """
    Parameters for proxy data

    Attributes
    ----------
    use_from: list(str)
        A list of keys for proxy classes to load from.  Keys available are
        stored in LMR_proxy2.
    proxy_frac: float
        Fraction of available proxy data (sites) to assimilate
    proxy_timeseries_kind: string
        Type of proxy timeseries to use. 'anom' for anomalies or 'asis'
        to keep records as included in the database.
    load_psm_with_proxy: bool
        Flag to indicate whether PSMs should be loaded with proxy objects.
        If False the psm_obj will be None until explicitly loaded.
    proxy_availability_filter: boolean
        True/False flag indicating whether filtering of proxy records
        according to data availability over reconstruction period is
        to be performed. If True, only proxies with data covering the
        reconstruction period are retained for assimilation. 
        Condition on record completeness is controlled with
        proxy_availability_fraction.
    proxy_availability_fraction: float
        Minimum threshold on the fraction of available proxy annual data 
        over the reconstruction period. i.e. control on the fraction of 
        available data that a recors must have in order to be assimilated. 
    """

    ##** BEGIN User Parameters **##

    # =============================
    # Which proxy database to use ?
    # =============================
    use_from = 'PAGES2kv1'
    #use_from = 'LMRdb'
    #use_from = 'NCDCdtda'

    proxy_frac = 1.0
    #proxy_frac = 0.75

    # type of proxy timeseries to return: 'anom' for anomalies
    # (temporal mean removed) or asis' to keep unchanged
    proxy_timeseries_kind = 'asis'

    load_psm_with_proxies = True

    # Filtering proxy records on conditions of data availability during
    # the reconstruction period.
    # - Filtrering disabled if proxy_availability_filter = False.
    # - If proxy_availability_filter = True, only records with
    #   oldest and youngest data outside or at edges of the recon. period
    #   are considered for assimilation.
    # - Testing for record completeness through the application of a threshold
    #   on data availability fraction (proxy_availability_fraction parameter).
    #   Records with a fraction of available data (ratio of valid data over
    #   the maximum data expected within the reconstruction period) below the
    #   user-defined threshold are omitted.
    #   Set this threshold to 0.0 if you do not want this threshold applied.
    #   Set this threshold to 1.0 to prevent assimilation of records with
    #   any missing data within the reconstruction period.
    proxy_availability_filter = False
    proxy_availability_fraction = 1.0

    ##** END User Parameters **##

    # -------------------
    # Generic Proxy Config Group
    # -------------------
    class ProxyConfigGroup(ConfigGroup):
        """
        Parameters for a ProxyGroup Class

        Notes
        -----
        proxy_type_mappings and simple_filters are creating during instance
        creation.

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data *or* None if using default lmr_path
        datafile_proxy: str
            proxy records filename
        metafile_proxy: str
            proxy metadata filename
        dataformat_proxy: str
            File format of the proxy data files
        regions: list(str)
            List of proxy data regions (data keys) to use.
        proxy_resolution: list(float)
            List of proxy time resolutions to use
        proxy_order: list(str):
            Proxy types to be assimilated and 
            order of assimilation.
        proxy_psm_type: dict{str:str}
            Association between proxy type and psm type.
        proxy_assim2: dict{ str: list(str)}
            Maps proxy type and measurement to our proxy type keys.
            Uses dictionary with structure {<<proxy type>>: [.. list of measurement
            tags ..] where "proxy type" is written as
            "<<archive type>>_<<measurement type>>"
        simple_filters: dict{'str': Iterable}
            List mapping Pages2k metadata sheet columns to a list of values
            to filter by.
        proxy_blacklist: list(str)
            A list of proxy ids to prevent from being used in the reconstruction
        proxy_type_mapping: dict{(str,str): str}
            Maps proxy type and measurement to our proxy type keys.
            (e.g. {('Tree ring', 'TRW'): 'Tree ring_Width'} )
        """

        # This class can't be instantiated.  Just used to share docstring and
        # initialization of the proxy group configurations.

        def __init__(self, lmr_path=None, **kwargs):
            super(proxies.ProxyConfigGroup, self).__init__(**kwargs)
            if lmr_path is None:
                lmr_path = core.lmr_path

            if self.datadir_proxy is None:
                self.datadir_proxy = join(lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = self.datadir_proxy

            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = list(self.proxy_resolution)
            self.proxy_timeseries_kind = proxies.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_psm_type = deepcopy(self.proxy_psm_type)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = proxies.proxy_availability_filter
            self.proxy_availability_fraction = proxies.proxy_availability_fraction

            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.items():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

    # -----------------
    # PAGES2kv1 proxies
    # -----------------
    class PAGES2kv1(ProxyConfigGroup):
        """
        Parameters for a PAGES2kv1 Class

        See ProxyConfigGroup for a description of common Proxy configuration
        attributes.

        Attributes
        ----------
        simple_filters: dict{'str': Iterable}
            List mapping proxy metadata sheet columns to a list of values
            to filter by.

        """

        ##** BEGIN User Parameters **##

        datadir_proxy = None
        datafile_proxy = 'Pages2kv1_Proxies.df.pckl'
        metafile_proxy = 'Pages2kv1_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']
        proxy_resolution = [1.0]

        # DO NOT CHANGE *FORMAT* BELOW

        proxy_order = [
            'Tree ring_Width',
            'Tree ring_Density',
            'Ice core_d18O',
            'Ice core_d2H',
            'Ice core_Accumulation',
            'Coral_d18O',
            'Coral_Luminescence',
            'Lake sediment_All',
            'Marine sediment_All',
            'Speleothem_All'
            ]

        # Assignment of psm type per proxy type
        # Choices are: 'linear', 'linear_TorP', 'bilinear', 'h_interp'
        #  The linear PSM can be used on *all* proxies.
        #  The linear_TorP and bilinear w.r.t. temperature or/and moisture
        #  PSMs are aimed at *tree ring* proxies in particular
        #  The h_interp forward model is to be used for isotope proxies when
        #  the prior is taken from an isotope-enabled GCM model output.
        proxy_psm_type = {
            'Tree ring_Width'      : 'linear',
            'Tree ring_Density'    : 'linear',
            'Ice core_d18O'        : 'linear',
            'Ice core_d2H'         : 'linear',
            'Ice core_Accumulation': 'linear',
            'Coral_d18O'           : 'linear',
            'Coral_Luminescence'   : 'linear',
            'Lake sediment_All'    : 'linear',
            'Marine sediment_All'  : 'linear',
            'Speleothem_All'       : 'linear',
            }

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

        # A blacklist on proxy records, to prevent assimilation of chronologies
        # known to be duplicates
        proxy_blacklist = []

        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(proxies.PAGES2kv1, self).__init__(lmr_path=lmr_path, **kwargs)

            self.simple_filters = {'PAGES 2k Region': self.regions,
                                   'Resolution (yr)': self.proxy_resolution}

    # -------------
    # LMRdb proxies
    # -------------
    class LMRdb(ProxyConfigGroup):
        """
        Parameters for LMRdb proxy class

        See ProxyConfigGroup for a description of common Proxy configuration
        attributes.

        Attributes
        ----------
        dbversion: str
            Version string of the database file to use.
        database_filter: list(str)
            List of databases from which to limit the selection of proxies.
            Use [] (empty list) if no restriction, or ['db_name1', db_name2'] to
            limit to proxies contained in "db_name1" OR "db_name2".
            Possible choices are: 'PAGES1', 'PAGES2', 'LMR_FM'
        simple_filters: dict{'str': Iterable}
            List mapping proxy metadata sheet columns to a list of values
            to filter by.
        """

        ##** BEGIN User Parameters **##

        dbversion = 'v1.0.0'
        
        datadir_proxy = None
        datafile_proxy = 'LMRdb_{}_Proxies.df.pckl'
        metafile_proxy = 'LMRdb_{}_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        # This is not activated with LMRdb data yet...
        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']

        proxy_resolution = [1.0]

        # Limit proxies to those included in the following list of databases
        # Note: Empty list = no restriction
        #       If list has more than one element, only records contained in ALL
        #       databases listed will be retained. Possibilities are:
        #       database_filter = ['PAGES2']                   # for db v0.1.0
        #       database_filter = ['LMR','Breits','PAGES2']    # for db v0.1.0
        #       database_filter = ['LMR','Breits','PAGES2kv2'] # for db v0.2.0
        database_filter = []
        
        # DO NOT CHANGE *FORMAT* BELOW
        proxy_order = [
#old            'Tree Rings_WidthPages',
            'Tree Rings_WidthPages2',
            'Tree Rings_WidthBreit',
            'Tree Rings_WoodDensity',
            'Tree Rings_Isotopes',
            'Corals and Sclerosponges_d18O',
            'Corals and Sclerosponges_SrCa',
            'Corals and Sclerosponges_Rates',
            'Ice Cores_d18O',
            'Ice Cores_dD',
            'Ice Cores_Accumulation',
            'Ice Cores_MeltFeature',
            'Lake Cores_Varve',
            'Lake Cores_BioMarkers',
            'Lake Cores_GeoChem',
            'Lake Cores_Misc',
            'Marine Cores_d18O',
            'Bivalve_d18O',
            'Speleothems_d18O',
            ]

        # Assignment of psm type per proxy type
        # Choices are: 'linear', 'linear_TorP', 'bilinear', 'h_interp'
        #  The linear PSM can be used on *all* proxies.
        #  The linear_TorP and bilinear w.r.t. temperature or/and moisture
        #  PSMs are aimed at *tree ring* proxies in particular
        #  The h_interp forward model is to be used for isotope proxies when
        #  the prior is taken from an isotope-enabled GCM output. 
        proxy_psm_type = {
            'Bivalve_d18O'                  : 'linear',
            'Corals and Sclerosponges_d18O' : 'linear',
            'Corals and Sclerosponges_SrCa' : 'linear',
            'Corals and Sclerosponges_Rates': 'linear',
            'Ice Cores_d18O'                : 'linear',
            'Ice Cores_dD'                  : 'linear',
            'Ice Cores_Accumulation'        : 'linear',
            'Ice Cores_MeltFeature'         : 'linear',
            'Lake Cores_Varve'              : 'linear',
            'Lake Cores_BioMarkers'         : 'linear',
            'Lake Cores_GeoChem'            : 'linear',
            'Lake Cores_Misc'               : 'linear',
            'Marine Cores_d18O'             : 'linear',
            'Tree Rings_WidthBreit'         : 'linear',
            'Tree Rings_WidthPages2'        : 'linear',
#old            'Tree Rings_WidthPages'         : 'linear',
            'Tree Rings_WoodDensity'        : 'linear',
            'Tree Rings_Isotopes'           : 'linear',
            'Speleothems_d18O'              : 'linear',
        }
         
        proxy_assim2 = {
            'Bivalve_d18O'                  : ['d18O'],
            'Corals and Sclerosponges_d18O' : ['d18O', 'delta18O', 'd18o',
                                               'd18O_stk', 'd18O_int',
                                               'd18O_norm', 'd18o_avg',
                                               'd18o_ave', 'dO18',
                                               'd18O_4'],
            'Corals and Sclerosponges_SrCa' : ['Sr/Ca', 'Sr_Ca', 'Sr/Ca_norm',
                                               'Sr/Ca_anom', 'Sr/Ca_int'],
            'Corals and Sclerosponges_Rates': ['ext','calc','calcification','calcification rate',
                                               'composite'],
            'Ice Cores_d18O'                : ['d18O', 'delta18O', 'delta18o',
                                               'd18o', 'd18o_int', 'd18O_int',
                                               'd18O_norm', 'd18o_norm', 'dO18',
                                               'd18O_anom'],
            'Ice Cores_dD'                  : ['deltaD', 'delD', 'dD'],
            'Ice Cores_Accumulation'        : ['accum', 'accumu'],
            'Ice Cores_MeltFeature'         : ['MFP','melt'],
            'Lake Cores_Varve'              : ['varve', 'varve_thickness',
                                               'varve thickness', 'thickness'],
            'Lake Cores_BioMarkers'         : ['Uk37', 'TEX86','tex86'],
            'Lake Cores_GeoChem'            : ['Sr/Ca', 'Mg/Ca', 'Cl_cont'],
            'Lake Cores_Misc'               : ['RABD660_670','X_radiograph_dark_layer','massacum'],
            'Marine Cores_d18O'             : ['d18O'],
            'Tree Rings_WidthBreit'         : ['trsgi_breit'],
            'Tree Rings_WidthPages2'        : ['trsgi'],
#old            'Tree Rings_WidthPages'         : ['TRW',
#old                                              'ERW',
#old                                              'LRW'],
            'Tree Rings_WoodDensity'        : ['max_d',
                                               'min_d',
                                               'early_d',
                                               'earl_d',
                                               'density',
                                               'late_d',
                                               'MXD'],
            'Tree Rings_Isotopes'           : ['d18O'],
            'Speleothems_d18O'              : ['d18O'],
        }

        # A blacklist on proxy records, to prevent assimilation of specific
        # chronologies known to be duplicates or to have errors.
        #proxy_blacklist = ['00aust01a', '06cook02a', '06cook03a', '08vene01a',
        #                   '09japa01a', '10guad01a', '99aust01a', '99fpol01a',
        #                   '72Devo01',  '72Devo05'] # for db v0.1.0
        proxy_blacklist = []
        
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(proxies.LMRdb, self).__init__(lmr_path=lmr_path, **kwargs)
            self.metafile_proxy = self.metafile_proxy.format(self.dbversion)
            self.datafile_proxy = self.datafile_proxy.format(self.dbversion)
            self.database_filter = list(self.database_filter)
            self.simple_filters = {'Resolution (yr)': self.proxy_resolution}

            

    # --------------------------------------------------------
    # proxies specific to Deep Times Data Assimilation project
    # --------------------------------------------------------
    class NCDCdtda(ConfigGroup):
        """
        Parameters for NCDCdtda proxy class

        See ProxyConfigGroup for a description of common Proxy configuration
        attributes.

        Attributes
        ----------
        simple_filters: dict{'str': Iterable}
            List mapping proxy metadata sheet columns to a list of values
            to filter by.
        """

        ##** BEGIN User Parameters **##

        #dbversion = 'v0.0.0'
        dbversion = 'v0.0.1'
        
        datadir_proxy = None
        datafile_proxy = 'DTDA_{}_Proxies.df.pckl'
        metafile_proxy = 'DTDA_{}_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        # This is not activated with yet...
        regions = []

        # Restrict uploaded proxy records to those within specified
        # range of temporal resolutions. 
        proxy_resolution = [(0.,5000.)]

        # Limit proxies to those included in the following list of databases
        # Note: Empty list = no restriction
        #       If list has more than one element, only records contained in ALL
        #       databases listed will be retained.
        database_filter = []

        # DO NOT CHANGE *FORMAT* BELOW

        proxy_order = [
        #    'Marine Cores_uk37',
            'Marine sediments_uk37',
            ]

        # Assignment of psm type per proxy type
        # Choices are: 'linear', 'h_interp', 'bayesreg_uk37', 'bayesreg_tex86'
        #  The linear PSM can be used on *all* proxies.
        #  The h_interp forward model is to be used for isotope proxies when
        #  the prior is taken from an isotope-enabled GCM output. 
        proxy_psm_type = {
            'Marine Cores_uk37'             : 'bayesreg_uk37',
            'Marine sediments_uk37'         : 'bayesreg_uk37',
        }
        
        proxy_assim2 = {
            'Marine Cores_uk37'             : ['uk37', 'UK37'],
            'Marine sediments_uk37'         : ['UK37'],
        }

        # A blacklist on proxy records, to prevent assimilation of specific
        # chronologies known to be duplicates.
        proxy_blacklist = []
        
        
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(proxies.NCDCdtda, self).__init__(lmr_path=lmr_path, **kwargs)

            self.database_filter = list(self.database_filter)
            self.simple_filters = {'Resolution (yr)': self.proxy_resolution}

    # Initialize subclasses with all attributes
    def __init__(self, lmr_path=None, seed=None, **kwargs):
        self.PAGES2kv1 = self.PAGES2kv1(lmr_path=lmr_path,
                                        **kwargs.pop('PAGES2kv1', {}))
        self.LMRdb = self.LMRdb(lmr_path=lmr_path, **kwargs.pop('LMRdb', {}))
        self.NCDCdtda = self.NCDCdtda(lmr_path=lmr_path, **kwargs.pop('NCDCdtda', {}))

        super(self.__class__, self).__init__(**kwargs)
        
        self.use_from = self.use_from
        self.proxy_frac = self.proxy_frac
        self.proxy_timeseries_kind = self.proxy_timeseries_kind
        self.load_psm_with_proxies = self.load_psm_with_proxies
        self.proxy_availability_filter = self.proxy_availability_filter
        self.proxy_availability_fraction = self.proxy_availability_fraction

        if seed is None:
            seed = core.seed
        self.seed = seed


class psm(ConfigGroup):
    """
    Parameters for PSM classes
    """

    class linear(ConfigGroup):
        """
        Parameters for the linear fit PSM.

        Attributes
        ----------
        datatag: str
            Source key of calibration data for PSM
        datadir: str
            Absolute path to calibration data *or* None if using default
            lmr_path
        datafile: str
            Filename for calibration data
        dataformat: str
            Data storage type for calibration data
        pre_datafile: str
            Absolute path to precalibrated Linear PSM data *or* None if using
            default LMR path
        varname: str
            Variable name to use from the calibration dataset
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """

        ##** BEGIN User Parameters **##

        datatag = 'GISTEMP'

        pre_calib_datafile = None

        avg_interval = 'annual'
        # avg_interval = 'season

        season_source = 'proxy_metadata'
        # season_source = 'psm_calib'

        psm_r_crit = 0.0

        # Period over which data is used to establish a statistical relationship
        # between proxy and instrumental data (statistical PSMs only)
        calib_period = (1850, 2015)

        min_data_req_frac = 1.0  # 0.0 no data required, 1.0 all data required
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.datatag = self.datatag

            dataset_descr = _DataInfo.get_info(self.datatag)
            self.datainfo = dataset_descr['info']
            self.datadir = dataset_descr['datadir']
            self.datafile = dataset_descr['datafile']
            self.dataformat = dataset_descr['dataformat']

            self.psm_r_crit = self.psm_r_crit
            self.min_data_req_frac = self.min_data_req_frac

            self.avg_interval = self.avg_interval
            self.season_source = self.season_source

            if 'PAGES2kv1' in proxies.use_from and 'season' in self.avg_interval:
                raise ValueError('No seasonality information in PAGES2kv1 '
                                 'database.  Change avg_period to "annual" in '
                                 'your configuration')

            if lmr_path is None:
                lmr_path = core.lmr_path

            if self.datadir is None:
                self.datadir = join(lmr_path, 'data', 'analyses')
            else:
                self.datadir = self.datadir

            if self.pre_calib_datafile is None:
                if 'LMRdb' in proxies.use_from:
                    dbversion = proxies.LMRdb.dbversion
                    filename = ('PSMs_' + proxies.use_from +
                                '_' + dbversion +
                                '_' + self.avg_interval+
                                '_' + self.datatag_calib +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                else:
                    filename = ('PSMs_' + '-'.join(proxies.use_from) +
                                '_' + self.datatag_calib +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                self.pre_calib_datafile = join(lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile

            if self.datainfo['multiple_vars']:
                raise ValueError('Ambiguous calibration variable source '
                                 'detected in configuration.  If '
                                 'multiple_vars is true for dataset, '
                                 'calibration does not currently have ability t'
                                 'o choose specific variable. Switch '
                                 'datatag_calib in the config.')
            self.psm_required_variables = self.datainfo['psm_vartype']
                
    class linear_TorP(ConfigGroup):                
        """
        Parameters for the linear fit PSM, calibrated against 
        temperature OR moisture.

        Attributes
        ----------
        datatag_T: str
            Source of temperature calibration data for linear PSM
        datadir_T: str
            Absolute path to temperature calibration data *or* None if using
            default lmr_path
        datafile_T: str
            Filename for temperature calibration data
        datatag_P: str
            Source of precipitation calibration data for linear PSM
        datadir_P: str
            Absolute path to precipitation calibration data *or* None if using
            default lmr_path
        datafile_P: str
            Filename for precipitation calibration data
        dataformat: str
            Data storage type for calibration data
        pre_calib_datafile_T: str
            Absolute path to precalibrated Linear temperature PSM data
        pre_calib_datafile_P: str
            Absolute path to precalibrated Linear precipitation PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """

        ##** BEGIN User Parameters **##

        
        # linear PSM w.r.t. temperature
        # -----------------------------
        # Choice between:
        # ---------------
        datatag_T = 'GISTEMP'
        
        # linear PSM w.r.t. precipitation/moisture
        # ----------------------------------------
        datatag_P = 'GPCC'

        pre_calib_datafile_T = None
        pre_calib_datafile_P = None

        psm_r_crit = 0.0

        avg_interval = 'annual'
        # avg_interval = 'season

        season_source = 'proxy_metadata'
        # season_source = 'psm_calib'

        metric = 'corr'
        # metric = 'mse'


        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            temp_kwarg = {'datatag': self.datatag_T,
                          'pre_calib_datafile': self.pre_calib_datafile_T,
                          'avg_interval': self.avg_interval,
                          'season_source': self.season_source}
            mois_kwarg = {'datatag': self.datatag_P,
                          'pre_calib_datafile': self.pre_calib_datafile_P,
                          'avg_interval': self.avg_interval,
                          'season_source': self.season_source}

            # Configuration for temperature and moisture psms
            self.temperature = psm.linear(lmr_path=lmr_path, **temp_kwarg)
            self.moisture = psm.linear(lmr_path=lmr_path, **mois_kwarg)

            self.psm_r_crit = self.psm_r_crit
            self.metric = self.metric
                
    class bilinear(ConfigGroup):
        """
        Parameters for the bilinear fit PSM.

        Attributes
        ----------
        datatag_T: str
            Source of calibration temperature data for PSM
        datadir_T: str
            Absolute path to calibration temperature data
        datafile_T: str
            Filename for calibration temperature data
        dataformat_T: str
            Data storage type for calibration temperature data
        datatag_P: str
            Source of calibration precipitation/moisture data for PSM
        datadir_P: str
            Absolute path to calibration precipitation/moisture data
        datafile_P: str
            Filename for calibration precipitation/moisture data
        dataformat_P: str
            Data storage type for calibration precipitation/moisture data
        pre_calib_datafile: str
            Absolute path to precalibrated Linear PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """

        ##** BEGIN User Parameters **##

        # calibration source for  temperature
        # -----------------------------------
        datatag_T = 'GISTEMP'
        
        # calibration source for precipitation/moisture
        # ---------------------------------------------
        datatag_P = 'GPCC'

        avg_interval = 'annual'
        # avg_interval = 'season

        season_source = 'proxy_metadata'
        # season_source = 'psm_calib'

        pre_calib_datafile = None
        psm_r_crit = 0.0

        
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            temp_kwarg = {'datatag': self.datatag_T,
                          'avg_interval': self.avg_interval,
                          'season_source': self.season_source}
            mois_kwarg = {'datatag': self.datatag_P,
                          'avg_interval': self.avg_interval,
                          'season_source': self.season_source}

            # Configuration for temperature and moisture psms
            self.temperature = psm.linear(lmr_path=lmr_path, **temp_kwarg)
            self.moisture = psm.linear(lmr_path=lmr_path, **mois_kwarg)

            self.psm_r_crit = self.psm_r_crit
                
            if self.pre_calib_datafile is None:
                if proxies.use_from == 'LMRdb':
                    if self.avg_interval == 'season':
                        if self.season_source == 'proxy_metadata':
                            source = 'META'
                        elif self.season_source == 'psm_calib':
                            source = 'PSM'
                    else:
                        source = ''
                    dbversion = proxies.LMRdb.dbversion
                    filename = ('PSMs_'+proxies.use_from +
                                '_' + dbversion +
                                '_' + self.avgPeriod +
                                '_' + self.datatag_calib_T +
                                '_' + self.datatag_calib_P +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                else:
                    filename = ('PSMs_'+'-'.join(proxies.use_from) +
                                '_' + self.datatag_calib_T +
                                '_' + self.datatag_calib_P +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                self.pre_calib_datafile = join(lmr_path,
                                                 'PSM',
                                                 filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile
        
    class h_interp(ConfigGroup):
        """
        Parameters for the horizontal interpolator PSM.

        Attributes
        ----------
        radius_influence : real
            Distance-scale used the calculation of exponentially-decaying
            weights in interpolator (in km)
        datadir_obsError: str
            Absolute path to obs. error variance data
        filename_obsError: str
            Filename for obs. error variance data
        dataformat_obsError: str
            String indicating the format of the file containing obs. error
            variance data
            Note: note currently used by code. For info purpose only.
        datafile_obsError: str
            Absolute path/filename of obs. error variance data
        """

        ##** BEGIN User Parameters **##

        # Interpolation parameter:
        # Set to 'None' if want Ye = value at nearest grid point to proxy
        # location
        # Set to a non-zero float if want Ye = weighted-average of surrounding
        # gridpoints
        # radius_influence = None
        radius_influence = 50. # distance-scale in km
        
        datadir_obsError = './'
        filename_obsError = 'R.txt'
        dataformat_obsError = 'TXT'

        datafile_obsError = None

        ##** END User Parameters **##

        def __init__(self, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.radius_influence = self.radius_influence
            self.datadir_obsError = self.datadir_obsError
            self.filename_obsError = self.filename_obsError
            self.dataformat_obsError = self.dataformat_obsError 

            if self.datafile_obsError is None:
                self.datafile_obsError = join(self.datadir_obsError,
                                              self.filename_obsError)
            else:
                self.datafile_obsError = self.datafile_obsError

            # define state variable needed to calculate Ye's
            # only d18O for now ...

            # psm requirements depend on settings in proxies class 
            proxy_kind = proxies.proxy_timeseries_kind
            if proxies.proxy_timeseries_kind == 'asis':
                psm_var_kind = 'full'
            elif proxies.proxy_timeseries_kind == 'anom':
                psm_var_kind = 'anom'
            else:
                raise ValueError('Unrecognized proxy_timeseries_kind value in proxies class.'
                                 ' Unable to assign kind to psm_required_variables'
                                 ' in h_interp psm class.')
            self.psm_required_variables = {'d18O_sfc_Amon': psm_var_kind}


    class bayesreg_uk37(ConfigGroup):
        """
        Parameters for the Bayesian regression PSM for uk37 proxies.

        Attributes
        ----------
        radius_influence : real
            Distance-scale used the calculation of exponentially-decaying
            weights in interpolator (in km)
        datadir_obsError: str
            Absolute path to obs. error variance data
        filename_obsError: str
            Filename for obs. error variance data
        dataformat_obsError: str
            String indicating the format of the file containing obs. error
            variance data
            Note: note currently used by code. For info purpose only.
        datafile_obsError: str
            Absolute path/filename of obs. error variance data
        """

        ##** BEGIN User Parameters **##
        
        datadir_BayesRegressionData = None
        filename_BayesRegressionData = None
        
        dataformat_BayesRegressionData = 'MAT' # a .mat Matlab file

        
        ##** END User Parameters **##

        def __init__(self, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            if self.datadir_BayesRegressionData is None:
                self.datadir_BayesRegressionData = join(core.lmr_path, 'PSM')
            else:
                self.datadir_BayesRegressionData = self.datadir_BayesRegressionData
            
            if self.filename_BayesRegressionData is None:
                self.filename_BayesRegressionData = 'PSM_bayes_posterior_UK37.mat'
            else:
                self.filename_BayesRegressionData = self.filename_BayesRegressionData

            self.datafile_BayesRegressionData = join(self.datadir_BayesRegressionData,
                                              self.filename_BayesRegressionData)
    
            # define state variable needed to calculate Ye's
            self.psm_required_variables = {'tos_sfc_Odec': 'full'}

            # matlab engine as python object, needed for calculations
            # of the forward model.
            #self.MatlabEng = matlab.engine.start_matlab('-nojvm')

            self.datadir_BayesRegressionData = self.datadir_BayesRegressionData
            self.datafile_BayesRegressionData = self.datafile_BayesRegressionData
            self.dataformat_BayesRegressionData = self.dataformat_BayesRegressionData


    # Initialize subclasses with all attributes 
    def __init__(self, lmr_path=None, save_pre_avg_file=None,
                 ignore_pre_avg_file=None, **kwargs):

        self.linear = self.linear(lmr_path=lmr_path, **kwargs.pop('linear', {}))
        self.linear_TorP = self.linear_TorP(lmr_path=lmr_path,
                                            **kwargs.pop('linear_TorP', {}))
        self.bilinear = self.bilinear(lmr_path=lmr_path,
                                      **kwargs.pop('bilinear', {}))
        self.h_interp = self.h_interp(**kwargs.pop('h_interp', {}))
        self.bayesreg_uk37 = self.bayesreg_uk37(**kwargs.pop('bayesreg_uk37', {}))

        super(self.__class__, self).__init__(**kwargs)

        # TODO: Find out if this is the right level to have the save
        # TODO:   info (should it be in PSM-specific config
        if ignore_pre_avg_file is None:
            ignore_pre_avg_file = core.ignore_pre_avg_file
        self.ignore_pre_avg_file = ignore_pre_avg_file

        if save_pre_avg_file is None:
            save_pre_avg_file = core.save_pre_avg_file
        self.save_pre_avg_file = save_pre_avg_file

        # Add constants instance for averaging periods
        self._avg_def_constants = Constants.get_info('avg_interval')

    def add_avg_interval_kwargs(self, name, elem_to_avg, nelem_in_yr, nyears):
        """Add an averaging definition to the current psm_config list of
        definitions"""
        if name in self._avg_def_constants:
            raise KeyError('Average definition already exists.')

        self._avg_def_constants[name] = {'nelem_in_yr': nelem_in_yr,
                                         'elem_to_avg': elem_to_avg,
                                         'nyears': nyears}

    def find_key_for_elem_list(self, elem_to_avg):
        """Find an average definition key from the list of elements in that
        defition"""
        for avg_def_name, avg_def_kwargs in self._avg_def_constants.items():
            if elem_to_avg == avg_def_kwargs['elem_to_avg']:
                return avg_def_name
        else:
            return None

    def create_avg_def_name(self, elem_to_avg):
        """Create an average definition name from a list of elements being
        averaged"""
        avg_def_tmpl = 'proxy_seasonal_{}'
        elem_to_avg_str = '-'.join([str(elem) for elem in elem_to_avg])
        avg_def_name = avg_def_tmpl.format(elem_to_avg_str)

        return avg_def_name


class prior(ConfigGroup):
    """
    Parameters for the ensemble DA prior

    Attributes
    ----------
    prior_source: str
        Source of prior data
    datadir: str
        Absolute path to prior data *or* None if using default LMR path
    datafile: str
        Name of prior file to use
    dataformat: str
        Datatype of prior container ('NCD' for netCDF, 'TXT' for ascii files).
        Note: Currently not used. 
    backend_type: str
        Which backend to use for storing prior data during updates with
        shifted assimilation resolution.  Allowed flags are 'NPY' for numpy
        and 'H5' for HDF5 backends.
    state_variables: dict
       Dict. of the form {'var1': 'kind1', 'var2':'kind2', etc.} where 'var1',
       'var2', etc. (keys of the dict) are the names of the state variables to
       be included in the state vector and 'kind1', 'kind2' etc. are the
       associated "kind" for each state variable indicating whether anomalies
       ('anom') or full field ('full') are desired.
    detrend: bool
        Indicates whether to detrend the prior or not. Applies to ALL state
        variables.
    avg_interval: dict OR list(int)
        dict of the form {'type':value} where 'type' indicates the type of
        averaging ('annual' or 'multiyear').
        If type = 'annual', the corresponding value is a 
        list of integers indficsting the months of the year over which the
        averaging is the be performed (ex. [6,7,8] for JJA).
        If type = 'multiyear', the list is composed of a single integer
        indicating the length of the averaging period, in number of years
        (ex. [100] for prior returned as 100-yr averages).
        -OR-
        List of integers indicating the months over which to average the annual
        prior. (as 'annual' above).
    output: dict
        Output designations for fields and scalars during runtime.
    regrid_method: str
        String indicating the method used to regrid the prior to lower spatial
        resolution.
        Allowed options are: 
        1) None : Regridding NOT performed. 
        2) 'spherical_harmonics' : Original regridding using pyspharm library.
        3) 'simple': Regridding through simple inverse distance averaging of
            surrounding grid points.
        4) 'esmpy': Regridding using the ESMpy package. Includes bilinear and
           higher-order patch fit regridding.
    regrid_resolution: int
        Integer representing the triangular truncation of the lower resolution
        grid (e.g. 42 for T42). Not used for 'esmpy' regrid_method.
    esmpy_interp_method: str or dict of str
        Which ESMpy regridding method to use. A single string will be apply
        this method to all fields, or a dict of mappings from prior variables
        to interpolation methods allows for specification.
        Currently supports:
        'bilinear':  Bilinear interpolation, good for smooth fields
        'patch': Higher order patch interpolation
        'conserve': Conservative regridding, only use on fields where
            conservation makes sense.  E.g. Mass or energy fields
    esmpy_regrid_to: str
        A grid defined in grid_def.yml to use as the regridding target.  
        Currently supports 't42' and 'reg_4x5deg'.
    state_variables_info: dict
        Defines which variables represent temperature or moisture.
        Should be modified only if a new temperature or moisture state variable
        is added.
    """

    ##** BEGIN User Parameters **##

    # Prior data directory & model source
    prior_source = 'ccsm4_last_millenium'

    # dict defining variables to be included in state vector (keys)
    # and associated "kind", i.e. as anomalies ('anom') or full field ('full')
    state_variables = {
        'tas_sfc_Amon'              : 'anom',
        }

    # Averaging interval for data defined in constants.yml
    avg_interval = 'annual_std'

    # Designate outputs
    outputs = {
        'prior': [],
        'posterior': ['ens_var', 'ens_mean'],
        'field_ens_output': None,
        'analysis_Ye': False,
        'scalar_ens': {'tas_sfc_Amon': ['glob_mean']}
    }




    ## The reference period (in year CE) for calculation of anomalies
    ## ** Valid for prior ccsm3_trace21ka only for now. Use None for all others **
    ## Options: None or tuple indicating the reference period
    #anom_reference = None



    
    # boolean : detrend prior?
    # by default, considers the entire length of the simulation
    detrend = False

    # In memory ('NPY') or out of memory ('H5') state handling
    backend_type = 'NPY'

    # Method for regridding to lower resolution grid.
    # Possible methods:
    # 1) None : No regridding performed. 
    # 2) 'spherical_harmonics': through spherical harmonics using pyspharm
    #    library. Note: Does *NOT* handle fields with missing/masked values (
    #    e.g. ocean variables)
    # 3) 'simple': through simple interpolation using distance-weighted
    #    averaging. Note: fields with missing/masked values (e.g. ocean
    #    variables) allowed.
    # 4) 'esmpy': regridding facilitated by ESMpy package, includes
    #    bilinear, and higher order patch interpolation
    regrid_method = 'simple'
    # resolution of truncated grid, based on triangular truncation (e.g.,
    # use 42 for T42))
    # 2) 'spherical_harmonics': through spherical harmonics using pyspharm library.
    #    Note: Does *NOT* handle fields with missing/masked values (e.g. ocean variables)
    # 3) 'simple': through simple interpolation using distance-weighted averaging.
    #    Note: fields with missing/masked values (e.g. ocean variables) allowed.
    # 4) 'esmpy': regridding facilitated by ESMpy package, includes blinear,
    #    and higher order patch interpolation
    regrid_method = 'esmpy'
    # resolution of truncated grid, based on triangular truncation (e.g., use 42 for T42))
    # note: this option applies only to 'simple' or 'spherical_harmonics' options
    regrid_resolution = 42

    # ESMpy regridding options
    esmpy_interp_method = 'bilinear'
    esmpy_regrid_to = 't42'

    ##** END User Parameters **##

    def __init__(self, lmr_path=None, seed=None, nens=None,
                 save_pre_avg_file=None, ignore_pre_avg_file=None,
                 **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.prior_source = self.prior_source

        dataset_descr = _DataInfo.get_info(self.prior_source)
        self.datainfo = dataset_descr['info']
        self.datadir = dataset_descr['datadir']
        self.datafile = dataset_descr['datafile']
        self.dataformat = dataset_descr['dataformat']
        self.psm_var_map = self.datainfo['psm_var_map']

        self.state_variables = deepcopy(self.state_variables)
        self.detrend = self.detrend
        self.regrid_method = self.regrid_method

        if nens is None:
            nens = core.nens
        self.nens = nens

        if seed is None:
            seed = core.seed
        self.seed = seed

        if lmr_path is None:
            lmr_path = core.lmr_path

        if save_pre_avg_file is None:
            save_pre_avg_file = core.save_pre_avg_file
        self.save_pre_avg_file = save_pre_avg_file

        if ignore_pre_avg_file is None:
            ignore_pre_avg_file = core.ignore_pre_avg_file
        self.ignore_pre_avg_file = ignore_pre_avg_file

        if self.datadir is None:
            self.datadir = join(lmr_path, 'data', 'model',
                                      self.prior_source)

        # TODO: Move these into visible configuration param, fix multiyear DADT
        # if core.recon_timescale == 1:
        #     self.avg_interval = {'annual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}  # annual (calendar) as default
        # elif core.recon_timescale > 1:
        #     # new format for CCSM3 TraCE21ka:
        #     self.avg_interval = {'multiyear': [core.recon_timescale]}
        # else:
        #     raise ValueError('ERROR in config.: unrecognized '
        #                      'core.recon_timescale!')

        self.avg_interval = self.avg_interval
        avg_interval_defs = Constants.get_info('avg_interval')
        self.avg_interval_kwargs = avg_interval_defs[self.avg_interval]

        self.outputs = deepcopy(self.outputs)

        if self.regrid_method != 'esmpy':
            self.regrid_resolution = int(self.regrid_resolution)
            self.regrid_grid = self.regrid_resolution
            self.esmpy_interp_method = None
            self.esmpy_grid_def = None
        elif self.regrid_method == 'esmpy':
            self.regrid_resolution = None
            self.esmpy_interp_method = self._validate_interp_method()
            self.esmpy_grid_def = _GridDef.get_info(self.esmpy_regrid_to)
            self.regrid_grid = self.esmpy_regrid_to

        # Is variable requested in list of those specified as available?
        var_mismat = [varname for varname in self.state_variables
                      if varname not in self.datainfo['available_vars']]
        if var_mismat:
            raise SystemExit(('Could not find requested variable(s) {} in the '
                              'list of available variables for the {} '
                              'dataset').format(var_mismat, self.prior_source))

    def _validate_interp_method(self):

        interp_method = self.esmpy_interp_method

        if isinstance(interp_method, dict):
            state_keys = self.state_variables.keys()
            for key in interp_method.keys():
                if key not in state_keys:
                    raise KeyError('ESMPy interpolation method mapping key '
                                   'not found in state variable list: '
                                   '{} ... Please fix '
                                   'configuration'.format(key))
        else:
            if not isinstance(interp_method, str):
                raise ValueError('Incorrect ESMPy interpolation method '
                                 'specified in the configuration.')

        return interp_method


class forecaster(ConfigGroup):
    """
    Parameters for the online DA forecasting method.

    Attributes
    ----------
    use_forecaster: str
        Key of forecasting class to use for the current reconstruction.
    """

    # Which forecaster class to use
    use_forecaster = 'lim'

    class lim(ConfigGroup):
        """
        datatag: str
            Source key of calibration data for LIM
        datainfo: dict
            Information about the calibration data. From the info section of
            datasets.yml
        datadir: str
            Absolute path to the source data file for the LIM
        datafile: str
            Filename of source data for LIM
        dataformat: str
            Dataformat key of source data for LIM
        avg_interval: str
            Average interval key as defined in constants.yml for averaging
            the calibration data.
        fcast_type: str
            Type of LIM forecast operation to perform.
            'perfect': noiseless projection from time t to forecast at t + fcast
                       lead. This should be used with hybrid covariance
                       blending in core.
            'ens_mean_perfect': Forecasts the ensemble mean value using a
                                perfect forecast.  Still reduces the ensemble
                                spread but presumabaly less so.  EXPERIMENTAL
            'noise_integrate': takes each ensemble member and forecasts using
                               noise integration method.
        fcast_varnames: list(str)
            List of variables to calibrate and forecast
        prior_mapping: dict{str: str}
            Mapping of forecast variables to prior variables.
        forecast_lead: int
            Forecast time period in units of the avg_interval
        fcast_num_pcs: int
            Number of principle components to retain during LIM forecast
            calibration.
        dobj_num_pcs: int
            Number of principle components to reduce each variable to before
            creation of multi-variate state vector. (Parameter reduction that
            lowers the memory cost of multi-variate EOF truncation.)
        detrend: bool
            Flag to detrend source data prior to calibration step.
        ignore_precalib: bool
            Ignore pre-calibrated LIM files
        use_ens_mean_fcast: bool
            Perform a forecast on the ensemble mean rather than each ensemble
            member
        """
        #TODO: Make this general to prior or analysis var, potentiall can remove
        #  the distinction in LMR_gridded
        datatag = 'ccsm4_last_millenium'

        avg_interval = 'annual_std'

        fcast_varnames = []
        prior_mapping = {}
        fcast_type = 'perfect'

        fcast_lead = 1
        fcast_num_pcs = 20
        dobj_num_pcs = 400
        detrend = True
        ignore_precalib_lim = False

        def __init__(self, lmr_path=None, save_pre_avg_file=None,
                     ignore_pre_avg_file=None, regrid_method=None,
                     regrid_grid=None, esmpy_interp_method=None, **kwargs):

            super(ConfigGroup, self).__init__(**kwargs)

            self.datatag = self.datatag

            dataset_descr = _DataInfo.get_info(self.datatag)
            self.datainfo = dataset_descr['info']
            self.datadir = dataset_descr['datadir']
            self.datafile = dataset_descr['datafile']
            self.dataformat = dataset_descr['dataformat']

            self.fcast_varnames = list(self.fcast_varnames)
            self.prior_mapping = deepcopy(self.prior_mapping)

            self.fcast_lead = self.fcast_lead
            self.fcast_num_pcs = self.fcast_num_pcs
            self.dobj_num_pcs = self.dobj_num_pcs
            self.detrend = self.detrend
            self.ignore_precalib_lim = self.ignore_precalib_lim

            self.avg_interval = self.avg_interval
            avg_interval_defs = Constants.get_info('avg_interval')
            self.avg_interval_kwargs = avg_interval_defs[self.avg_interval]

            self.fcast_type = self.fcast_type

            self.regrid_method = regrid_method
            self.regrid_grid = regrid_grid
            self.esmpy_interp_method = esmpy_interp_method
            if self.regrid_method != 'esmpy':
                self.regrid_resolution = self.regrid_grid
                self.esmpy_grid_def = None
            elif self.regrid_method == 'esmpy':
                self.regrid_resolution = None
                self.esmpy_grid_def = _GridDef.get_info(self.regrid_grid)

            if lmr_path is None:
                lmr_path = core.lmr_path

            if save_pre_avg_file is None:
                save_pre_avg_file = core.save_pre_avg_file
            self.save_pre_avg_file = save_pre_avg_file

            if ignore_pre_avg_file is None:
                ignore_pre_avg_file = core.ignore_pre_avg_file
            self.ignore_pre_avg_file = ignore_pre_avg_file

            if self.datadir is None:
                model_dir = join(lmr_path, 'data', 'model', self.datatag)
                analysis_dir = join(lmr_path, 'data', 'analyses',
                                    self.datatag)

                # TODO: Currently only allowing single source calibration
                self.datadir = model_dir

                # if exists(join(model_dir, self.datafile)):
                #     self.datadir = model_dir
                # elif exists(join(analysis_dir, self.datafile)):
                #     self.datadir = analysis_dir
                # else:
                #     raise ValueError('Could not find calibration datafile in '
                #                      'default model or analyses directory.')

    def __init__(self, lmr_path=None, save_pre_avg_file=None,
                 ignore_pre_avg_file=None, regrid_method=None,
                 regrid_grid=None, esmpy_interp_method=None, **kwargs):
        self.lim = self.lim(lmr_path=lmr_path,
                            save_pre_avg_file=save_pre_avg_file,
                            ignore_pre_avg_file=ignore_pre_avg_file,
                            regrid_method=regrid_method,
                            regrid_grid=regrid_grid,
                            esmpy_interp_method=esmpy_interp_method,
                            **kwargs.pop('lim', {}))

        super(ConfigGroup, self).__init__(**kwargs)
        self.use_forecaster = self.use_forecaster


class Config(ConfigGroup):
    """
    An instanceable container for all the configuration objects.
    """

    def __init__(self, **kwargs):
        self.LEGACY_CONFIG = LEGACY_CONFIG
        self.SRC_DIR = SRC_DIR
        self.LOG_LEVEL = LOG_LEVEL

        self.wrapper = wrapper(**kwargs.pop('wrapper', {}))
        self.core = core(**kwargs.pop('core', {}))
        lmr_path = self.core.lmr_path
        seed = self.core.seed
        save_preavg = self.core.save_pre_avg_file
        ignore_preavg = self.core.ignore_pre_avg_file
        self.proxies = proxies(lmr_path=lmr_path,
                               seed=seed,
                               **kwargs.pop('proxies', {}))
        self.psm = psm(lmr_path=lmr_path, **kwargs.pop('psm', {}))
        self.prior = prior(lmr_path=lmr_path,
                           seed=seed, save_pre_avg_file=save_preavg,
                           ignore_pre_avg_file=ignore_preavg,
                           **kwargs.pop('prior', {}))

        regrid_method = self.prior.regrid_method
        regrid_grid = self.prior.regrid_grid
        esmpy_interp_method = self.prior.esmpy_interp_method
        self.forecaster = forecaster(lmr_path=lmr_path,
                                     save_pre_avg_file=save_preavg,
                                     ignore_pre_avg_file=ignore_preavg,
                                     regrid_method=regrid_method,
                                     regrid_grid=regrid_grid,
                                     esmpy_interp_method=esmpy_interp_method,
                                     **kwargs.pop('forecaster', {}))


def is_config_class(obj):
    """
    Tests whether the input object is an instance of ConfigGroup

    Parameters
    ----------
    obj: object
        Object for testing
    """
    try:
        if isinstance(obj, ConfigGroup):
            return True
        else:
            return issubclass(obj, ConfigGroup)
    except TypeError:
        return False


def update_config_attrs_yaml(yaml_dict, cfg_module):
    """
    Updates a configuration object using a dictionary (typically from a yaml
    file) that follows the naming convention and nesting of these
    configuration classes.

    Parameters
    ----------
    yaml_dict: dict
        The dictionary of values to update in the current configuration input
    cfg_module: ConfigGroup like
        The configuration object to be updated by yaml_dict

    Returns
    -------
    dict
        Returns a dictionary of all unused parameters from the update process

    Examples
    --------
    If cfg_module is an imported LMR_config as cfg then the following
    dictionary could be used to update a core and linear psm attribute.
    yaml_dict = {'core': {'lmr_path': '/new/path/to/LMR_files'},
                 'psm': {'linear': {'datatag': 'GISTEMP'}}}

    These are the same types of dictionaries that result from a yaml.load
    function.

    Warnings
    --------
    This function is meant to be run on imported configuration classes not
    their instances.  If you'd only like to update the attributes of an
    instance then please use keyword arguments during initialization.
    """

    for attr_name in list(yaml_dict.keys()):
        try:
            curr_cfg_obj = getattr(cfg_module, attr_name)
            cfg_attr = yaml_dict.pop(attr_name)

            if is_config_class(curr_cfg_obj):
                result = update_config_attrs_yaml(cfg_attr, curr_cfg_obj)

                if result:
                    yaml_dict[attr_name] = result

            else:
                setattr(cfg_module, attr_name, cfg_attr)
        except (AttributeError, KeyError) as e:
            print(e)

    return yaml_dict


def initialize_config_yaml(cfg_module, yaml_file=None):

    if yaml_file is None:
        yaml_file = join(cfg_module.SRC_DIR, 'config.yml')

    try:
        print('Loading configuration: {}'.format(yaml_file))
        f = open(yaml_file, 'r')
        yml_dict = yaml.load(f)
        update_result = cfg_module.update_config_attrs_yaml(yml_dict,
                                                            cfg_module)

        # Check that all yml params match value in LMR_config
        if update_result:
            raise SystemExit(
                'Extra or mismatching values found in the configuration yaml'
                ' file.  Please fix or remove them.\n  Residual parameters:\n '
                '{}'.format(update_result))

    except IOError as e:
        raise SystemExit(
            ('Could not locate {}.  If use of legacy LMR_config usage is '
             'desired then please change LEGACY_CONFIG to True'
             'in LMR_wrapper.py.').format(yaml_file))



if __name__ == "__main__":
    # kwargs = {'wrapper': {'multi_seed': [1, 2, 3]},
    #           'psm': {'linear': {'datatag': 'BE'}}}
    tmp = Config()
    pass
