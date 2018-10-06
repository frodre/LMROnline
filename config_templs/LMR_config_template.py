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

from LMR_utils import get_averaging_period

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
            avg_indices = tuple(avg_indices)

            val['elem_to_avg'] = avg_indices


# Load dataset information on configuration import
_DataInfo = _DatasetDescriptors()
_GridDef = _GridDefinitions()
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
    datadir_output: str
        Absolute path to working directory output for LMR
    archive_dir: str
        Absolute path to LMR reconstruction archive directory.

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

    datadir_output = '/home/katabatic2/wperkins/LMR_output/working'
    archive_dir = '/home/katabatic2/wperkins/LMR_output/testing'

    nexp = 'testdev_persistence'

    ##** END User Parameters **##

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.iter_range = self.iter_range

        num_iters = self.iter_range[1] - self.iter_range[0] + 1
        if self.multi_seed is not None:
            if isinstance(self.multi_seed, int):
                np.random.seed(self.multi_seed)
                self.multi_seed = np.random.randint(0, high=100000,
                                                    size=num_iters)
            else:
                self.multi_seed = list(self.multi_seed)

        self.iter_range = self.iter_range
        self.param_search = deepcopy(self.param_search)

        self.datadir_output = self.datadir_output
        self.archive_dir = self.archive_dir

        self.nexp = self.nexp

class core(ConfigGroup):
    """
    High-level parameters of LMR_driver_callable.

    Notes
    -----
    curr_iter attribute is created during initialization

    Attributes
    ----------
    nexp: str
        Name of reconstruction experiment. None defaults to wrapper.nexp
    lmr_path: str
        Absolute path for the experiment
    datadir_output: str
        Absolute path to working directory output for LMR. None defaults to
        wrapper.datadir_output
    online_reconstruction: bool
        Perform reconstruction with (True) or without (False) cycling
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
    seed: int, None
        RNG seed.  Passed to all random function calls. (e.g. prior and proxy
        record sampling)  Overridden by wrapper.multi_seed.
    loc_rad: float
        Localization radius for DA (in km)
    assimilation_solver: str
        Which solver to use for data assimilation. 'Serial' uses the serial
        EnSRF solver which allows localization (slower), and 'optimal' uses the
        optimal transorm EnKF method to update fields all at once.
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
    inflation_factor: float
        Variance inflation factor to use when ``reg_inflate`` is True

    """

    ##** BEGIN User Parameters **##

    lmr_path = '/home/disk/katabatic2/wperkins/cp_lim_archive/LMR_slim'

    nexp = None
    datadir_output = None

    # Whether or not to produce the analysis_Ye.pckl file
    write_posterior_Ye = False

    online_reconstruction = False
    clean_start = True
    use_precalc_ye = False
    recon_period = [1950, 1960]
    nens = 10
    recon_timescale = 1  # annual
    seed = None
    loc_rad = None
    assimilation_solver = 'optimal'

    # Forecasting Hybrid Update
    hybrid_update = True
    hybrid_update &= online_reconstruction
    hybrid_a = 0.85
    blend_prior = True

    # Adaptive Covariance Inflation
    adaptive_inflate = False
    reg_inflate = False
    inflation_factor = 1.1
    
    ##** END User Parameters **##

    def __init__(self, curr_iter=None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.recon_timescale = int(self.recon_timescale)

        self.nexp = self.nexp
        self.lmr_path = self.lmr_path
        self.write_posterior_Ye = self.write_posterior_Ye
        self.online_reconstruction = self.online_reconstruction
        self.clean_start = self.clean_start
        self.use_precalc_ye = self.use_precalc_ye
        self.recon_period = tuple(self.recon_period)
        self.nens = self.nens
        self.recon_timescale = self.recon_timescale
        self.seed = self.seed
        self.loc_rad = self.loc_rad
        self.assimilation_solver = self.assimilation_solver
        self.hybrid_update = self.hybrid_update
        self.hybrid_a = self.hybrid_a
        self.blend_prior = self.blend_prior
        self.adaptive_inflate = self.adaptive_inflate
        self.reg_inflate = self.reg_inflate
        self.inflation_factor = self.inflation_factor

        if curr_iter is None:
            self.curr_iter = wrapper.iter_range[0]
        else:
            self.curr_iter = curr_iter

        if self.datadir_output is None:
            self.datadir_output = wrapper.datadir_output

        if self.nexp is None:
            self.nexp = wrapper.nexp


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
    on_the_fly_calib = False

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
            Absolute path to proxy data directory. None defaults to proxy
            directory in lmr_path
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
        self.on_the_fly_calib = self.on_the_fly_calib
        self.proxy_availability_filter = self.proxy_availability_filter
        self.proxy_availability_fraction = self.proxy_availability_fraction

        if seed is None:
            seed = core.seed
        self.seed = seed


class psm(ConfigGroup):
    """
    Parameters for PSM classes

    Attributes
    ----------
    anom_reference_period: tuple of int
        The period to use as the reference for anomalie centering when
        calibrating PSMs. Edges are inclusive.  All reconstruction output
        will be an anomaly relative to this time
    calib_period: tuple of int
        Year range to use for calibrating the PSM.  Edges are inclusive.
    """

    # Period to use to center the calibration data, sets the reference period
    #  for the reconstruction when using statistical PSMs
    anom_reference_period = (1951, 1980)

    # Period over which data is used to establish a statistical relationship
    # between proxy and instrumental data (statistical PSMs only)
    calib_period = (1850, 2015)

    class linear(ConfigGroup):
        """
        Parameters for the linear fit PSM.

        Attributes
        ----------
        datatag: str
            Source key of calibration data for PSM
        pre_datafile: str, Optional
            Absolute path to precalibrated Linear PSM data
        avg_type: str
            Whether to use 'annual' or 'seasonal' time average definitions
            for calibration and use of proxy system models
        season_source: str
            Use seasonal information determeined from metadata (
            'proxy_metadata') or objective PSM calibration ('psm_calib')
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        min_data_req_frac: float
            The fraction of data required in a given year to count as an average
            and not NaN. TODO: Find out if this is still relevant
        """

        ##** BEGIN User Parameters **##

        datatag = 'GISTEMP'

        pre_calib_datafile = None

        avg_type = 'annual'
        # avg_type = 'season

        season_source = 'proxy_metadata'
        # season_source = 'psm_calib'

        detrend = False

        psm_r_crit = 0.0

        min_data_req_frac = 1.0  # 0.0 no data required, 1.0 all data required
        ##** END User Parameters **##

        def __init__(self, regrid_cfg, lmr_path=None,
                     anom_reference_period=None,
                     proxy_use_from=None, calib_period=None,
                     **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.datatag = self.datatag
            self.regrid_cfg = regrid_cfg

            dataset_descr = _DataInfo.get_info(self.datatag)
            self.datainfo = dataset_descr['info']
            self.datadir = dataset_descr['datadir']
            self.datafile = dataset_descr['datafile']
            self.dataformat = dataset_descr['dataformat']

            self.psm_r_crit = self.psm_r_crit
            self.min_data_req_frac = self.min_data_req_frac
            if self.min_data_req_frac < 0 or self.min_data_req_frac > 1:
                raise ValueError('Minimum fraction of data required must be '
                                 'between 0.0 and 1.0.')

            self.detrend = self.detrend

            self.avg_type = self.avg_type
            self.season_source = self.season_source

            if proxy_use_from is None:
                proxy_use_from = proxies.use_from
            self.proxy_use_from = proxy_use_from

            if anom_reference_period is None:
                anom_reference_period = psm.anom_reference_period
            self.anom_reference_period = anom_reference_period

            if calib_period is None:
                calib_period = psm.calib_period
            self.calib_period = calib_period


            if 'PAGES2kv1' in self.proxy_use_from and 'season' in self.avg_type:
                raise ValueError('No seasonality information in PAGES2kv1 '
                                 'database.  Change avg_period to "annual" in '
                                 'your configuration')

            # Avg_interval values will be set on a per proxy basis when
            # calibrating
            self.avg_interval = None
            self.avg_interval_kwargs = None

            if lmr_path is None:
                lmr_path = core.lmr_path

            if self.datadir is None:
                self.datadir = join(lmr_path, 'data', 'analyses', self.datatag)
            else:
                self.datadir = self.datadir

            if self.pre_calib_datafile is None:

                dbversion = proxies.LMRdb.dbversion
                cstart, cend = self.calib_period
                anom_start, anom_end = anom_reference_period
                regrid_grid = self.regrid_cfg.regrid_grid

                if self.avg_type == 'annual':
                    season_tag = _avg_type_pre_calib_tag['annual']
                else:
                    season_tag = _avg_type_pre_calib_tag[(self.avg_type,
                                                          self.season_source)]

                if 'LMRdb' == proxy_use_from:

                    filename = (f'PSMs_{proxy_use_from}_{dbversion}_'
                                f'{self.avg_type}{season_tag}_{self.datatag}_'
                                f'{regrid_grid}_ref{anom_start}-{anom_end}_'
                                f'cal{cstart}-{cend}.pckl')
                else:
                    filename = (f'PSMs_{proxy_use_from}_{self.datatag}_'
                                f'{regrid_grid}_ref{anom_start}-{anom_end}_'
                                f'cal{cstart}-{cend}.pckl')

                self.pre_calib_datafile = join(lmr_path, 'PSM', filename)
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

        def update_avg_interval(self, avg_interval, avg_interval_kwargs):
            self.avg_interval = avg_interval
            self.avg_interval_kwargs = avg_interval_kwargs
                
    class linear_TorP(ConfigGroup):                
        """
        Parameters for the linear fit PSM, calibrated against 
        temperature OR moisture.

        Attributes
        ----------
        datatag_T: str
            Source of temperature calibration data for linear PSM
        datatag_P: str
            Source of precipitation calibration data for linear PSM
        pre_calib_datafile_T: str, Optional
            Absolute path to precalibrated Linear temperature PSM data
        pre_calib_datafile_P: str, Optional
            Absolute path to precalibrated Linear precipitation PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        avg_type: str
            Whether to use 'annual' or 'seasonal' time average definitions
            for calibration and use of proxy system models
        season_source: str
            Use seasonal information determeined from metadata (
            'proxy_metadata') or objective PSM calibration ('psm_calib')
        metric: str
            Metric to use in determination of whether to use temperature or
            moisture PSM. 'corr' for correlation or 'mse' for mean squared
            error.
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

        avg_type = 'annual'
        # avg_type = 'season

        season_source = 'proxy_metadata'
        # season_source = 'psm_calib'

        metric = 'corr'
        # metric = 'mse'

        min_data_req_frac = 1.0


        ##** END User Parameters **##

        def __init__(self, regrid_cfg, lmr_path=None, proxy_use_from=None,
                     calib_period=None, anom_reference_period=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.min_data_req_frac = self.min_data_req_frac

            temp_kwarg = {'datatag': self.datatag_T,
                          'pre_calib_datafile': self.pre_calib_datafile_T,
                          'avg_type': self.avg_type,
                          'season_source': self.season_source,
                          'min_data_req_frac': self.min_data_req_frac}
            mois_kwarg = {'datatag': self.datatag_P,
                          'pre_calib_datafile': self.pre_calib_datafile_P,
                          'avg_type': self.avg_type,
                          'season_source': self.season_source,
                          'min_data_req_frac': self.min_data_req_frac}

            if proxy_use_from is None:
                proxy_use_from = proxies.use_from
            self.proxy_use_from = proxy_use_from

            if anom_reference_period is None:
                anom_reference_period = psm.anom_reference_period
            self.anom_reference_period = anom_reference_period

            if calib_period is None:
                calib_period = psm.calib_period
            self.calib_period = calib_period

            # Configuration for temperature and moisture psms
            self.temperature = psm.linear(regrid_cfg, lmr_path=lmr_path,
                                          proxy_use_from=proxy_use_from,
                                          calib_period=calib_period,
                                          anom_reference_period=anom_reference_period,
                                          **temp_kwarg)
            self.moisture = psm.linear(regrid_cfg, lmr_path=lmr_path,
                                       proxy_use_from=proxy_use_from,
                                       calib_period=calib_period,
                                       anom_reference_period=anom_reference_period,
                                       **mois_kwarg)

            self.psm_r_crit = self.psm_r_crit
            self.metric = self.metric

        def update_avg_interval(self, avg_interval, avg_interval_kwargs):
            self.temperature.update_avg_interval(avg_interval,
                                                 avg_interval_kwargs)
            self.moisture.update_avg_interval(avg_interval,
                                              avg_interval_kwargs)
                
    class bilinear(ConfigGroup):
        """
        Parameters for the bilinear fit PSM.

        Attributes
        ----------
        datatag_T: str
            Source of calibration temperature data for PSM
        datatag_P: str
            Source of calibration precipitation/moisture data for PSM
        pre_calib_datafile: str, Optional
            Absolute path to precalibrated Linear PSM data
        avg_type: str
            Whether to use 'annual' or 'seasonal' time average definitions
            for calibration and use of proxy system models
        season_source: str
            Use seasonal information determeined from metadata (
            'proxy_metadata') or objective PSM calibration ('psm_calib')
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

        avg_type = 'annual'
        # avg_type = 'season

        season_source = 'proxy_metadata'
        # season_source = 'psm_calib'

        pre_calib_datafile = None
        psm_r_crit = 0.0

        min_data_req_frac = 1.0

        ##** END User Parameters **##

        def __init__(self, regrid_cfg, lmr_path=None, proxy_use_from=None,
                     calib_period=None, anom_reference_period=None, **kwargs):

            super(self.__class__, self).__init__(**kwargs)

            self.min_data_req_frac = self.min_data_req_frac

            temp_kwarg = {'datatag': self.datatag_T,
                          'avg_type': self.avg_type,
                          'season_source': self.season_source,
                          'min_data_req_frac': self.min_data_req_frac}
            mois_kwarg = {'datatag': self.datatag_P,
                          'avg_type': self.avg_type,
                          'season_source': self.season_source,
                          'min_data_req_frac': self.min_data_req_frac}

            if proxy_use_from is None:
                proxy_use_from = proxies.use_from
            self.proxy_use_from = proxy_use_from

            if anom_reference_period is None:
                anom_reference_period = psm.anom_reference_period
            self.anom_reference_period = anom_reference_period

            if calib_period is None:
                calib_period = psm.calib_period
            self.calib_period = calib_period

            # Configuration for temperature and moisture psms
            self.temperature = psm.linear(regrid_cfg, lmr_path=lmr_path,
                                          proxy_use_from=proxy_use_from,
                                          calib_period=calib_period,
                                          anom_reference_period=anom_reference_period,
                                          **temp_kwarg)
            self.moisture = psm.linear(regrid_cfg, lmr_path=lmr_path,
                                       proxy_use_from=proxy_use_from,
                                       calib_period=calib_period,
                                       anom_reference_period=anom_reference_period,
                                       **mois_kwarg)

            self.psm_r_crit = self.psm_r_crit
                
            if self.pre_calib_datafile is None:

                dbversion = proxies.LMRdb.dbversion
                cstart, cend = self.calib_period
                anom_start, anom_end = anom_reference_period
                regrid_grid = regrid_cfg.regrid_grid

                if self.avg_type == 'annual':
                    season_tag = _avg_type_pre_calib_tag['annual']
                else:
                    season_tag = _avg_type_pre_calib_tag[(self.avg_type,
                                                          self.season_source)]

                if self.proxy_use_from == 'LMRdb':

                    filename = (f'PSMs_{proxy_use_from}_{dbversion}_'
                                f'{self.avg_type}{season_tag}_'
                                f'{self.datatag_T}_{self.datatag_P}_'
                                f'{regrid_grid}_ref{anom_start}-{anom_end}_'
                                f'cal{cstart}-{cend}.pckl')
                else:

                    filename = (f'PSMs_{proxy_use_from}_{dbversion}_'
                                f'{self.datatag_T}_{self.datatag_P}_'
                                f'{regrid_grid}_ref{anom_start}-{anom_end}_'
                                f'cal{cstart}-{cend}.pckl')

                self.pre_calib_datafile = join(lmr_path, 'PSM', filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile

        def update_avg_interval(self, avg_interval, avg_interval_kwargs):
            self.temperature.update_avg_interval(avg_interval,
                                                 avg_interval_kwargs)
            self.moisture.update_avg_interval(avg_interval,
                                              avg_interval_kwargs)
        
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
    def __init__(self, regrid_cfg, lmr_path=None, proxy_use_from=None,
                 **kwargs):

        self.anom_reference_period = self.anom_reference_period
        self.calib_period = self.calib_period

        self.linear = self.linear(regrid_cfg,
                                  lmr_path=lmr_path,
                                  proxy_use_from=proxy_use_from,
                                  anom_reference_period=self.anom_reference_period,
                                  calib_period=self.calib_period,
                                  **kwargs.pop('linear', {}))

        self.linear_TorP = self.linear_TorP(regrid_cfg,
                                            lmr_path=lmr_path,
                                            proxy_use_from=proxy_use_from,
                                            anom_reference_period=self.anom_reference_period,
                                            calib_period=self.calib_period,
                                            **kwargs.pop('linear_TorP', {}))

        self.bilinear = self.bilinear(regrid_cfg,
                                      lmr_path=lmr_path,
                                      proxy_use_from=proxy_use_from,
                                      anom_reference_period=self.anom_reference_period,
                                      calib_period=self.calib_period,
                                      **kwargs.pop('bilinear', {}))

        self.h_interp = self.h_interp(**kwargs.pop('h_interp', {}))
        self.bayesreg_uk37 = self.bayesreg_uk37(**kwargs.pop('bayesreg_uk37', {}))

        super(self.__class__, self).__init__(**kwargs)

        # Add constants instance for averaging periods (NOT A COPY)
        self._avg_def_constants = Constants.data['avg_interval']

    def _add_avg_interval_kwargs(self, name, elem_to_avg, nelem_in_yr, nyears):
        """Add an averaging definition to the current psm_config list of
        definitions"""
        if name in self._avg_def_constants:
            raise KeyError('Average definition already exists.')

        self._avg_def_constants[name] = {'nelem_in_yr': nelem_in_yr,
                                         'elem_to_avg': elem_to_avg,
                                         'nyears': nyears}

    def _find_key_for_elem_list(self, elem_to_avg):
        """Find an average definition key from the list of elements in that
        defition"""
        for avg_def_name, avg_def_kwargs in self._avg_def_constants.items():
            if elem_to_avg == avg_def_kwargs['elem_to_avg']:
                return avg_def_name
        else:
            return None

    @staticmethod
    def _create_avg_def_name(elem_to_avg):
        """Create an average definition name from a list of elements being
        averaged"""
        avg_def_tmpl = 'proxy_seasonal_{}'
        elem_to_avg_str = '-'.join([str(elem) for elem in elem_to_avg])
        avg_def_name = avg_def_tmpl.format(elem_to_avg_str)

        return avg_def_name

    @staticmethod
    def decode_avg_def_name(avg_interval_name):

        if 'proxy_seasonal' not in avg_interval_name:
            raise ValueError('Input average interval name does not appear to '
                             'be a generated seasonal proxy definition.')

        elem_str = avg_interval_name.split('_')[-1]
        elements = elem_str.split('-')
        return elements

    def get_avg_def(self, key):
        """Get a subannual element averaging definition from the attached
        constant definition"""
        return self._avg_def_constants[key]

    def handle_proxy_elem_list(self, elem_to_avg, nelem_in_yr=12, nyears=1):
        """Look for the appropriate avg_interval key for the element list and
        add it to the definitions if not found."""
        avg_interval = self._find_key_for_elem_list(elem_to_avg)

        if avg_interval is None:
            # No matching definition found, create a new one
            avg_interval = self._create_avg_def_name(elem_to_avg)

            self._add_avg_interval_kwargs(avg_interval, elem_to_avg,
                                          nelem_in_yr, nyears)

        avg_interval_kwargs = self.get_avg_def(avg_interval)

        return avg_interval, avg_interval_kwargs


class prior(ConfigGroup):
    """
    Parameters for the ensemble DA prior

    Attributes
    ----------
    prior_source: str
        Source of prior data
    state_variables: dict of str
       State variables to be reconstructed. Formated as
       {variable name: field type} where field type denotes whether anomaly
       ('anom') or full field ('full') results are desired.
    detrend: bool
        Whether to detrend the prior data after averaging is performed.
    avg_interval: str
        The averaging interval as defined in constants.yml to average the prior
        data to and the the average interval of the output.
    output: dict
        Output designations for fields and scalars during runtime.
    """

    ##** BEGIN User Parameters **##

    # Prior data directory & model source
    prior_source = 'ccsm4_last_millenium'

    # dict defining variables to be included in state vector (keys)
    # and associated "kind", i.e. as anomalies ('anom') or full field ('full')
    state_variables = {'tas_sfc_Amon': 'anom'}

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

    detrend = False

    # In memory ('NPY') or out of memory ('H5') state handling
    backend_type = 'NPY'

    ##** END User Parameters **##

    def __init__(self, regrid_cfg, lmr_path=None, seed=None, nens=None,
                 **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.prior_source = self.prior_source
        self.regrid_cfg = regrid_cfg

        dataset_descr = _DataInfo.get_info(self.prior_source)
        self.datainfo = dataset_descr['info']
        self.datadir = dataset_descr['datadir']
        self.datafile = dataset_descr['datafile']
        self.dataformat = dataset_descr['dataformat']
        self.psm_var_map = self.datainfo['psm_var_map']

        self.state_variables = deepcopy(self.state_variables)
        self.detrend = self.detrend

        if nens is None:
            nens = core.nens
        self.nens = nens

        if seed is None:
            seed = core.seed
        self.seed = seed

        if lmr_path is None:
            lmr_path = core.lmr_path

        if self.datadir is None:
            self.datadir = join(lmr_path, 'data', 'model', self.prior_source)

        self.avg_interval = self.avg_interval
        self._avg_interval_defs = Constants.data['avg_interval']
        self.avg_interval_kwargs = self._avg_interval_defs[self.avg_interval]

        self.outputs = deepcopy(self.outputs)

        # Is variable requested in list of those specified as available?
        var_mismat = [varname for varname in self.state_variables
                      if varname not in self.datainfo['available_vars']]
        if var_mismat:
            raise SystemExit(('Could not find requested variable(s) {} in the '
                              'list of available variables for the {} '
                              'dataset').format(var_mismat, self.prior_source))

    def update_avg_interval(self, avg_interval):
        self.avg_interval = avg_interval
        avg_interval_kwargs = self._avg_interval_defs[avg_interval]
        self.avg_interval_kwargs = avg_interval_kwargs


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
        match_prior: bool
            Designates whether to forecast all variables present in the state.
            If true, fcast_varnames and avg_interval is desregarded.
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
            Forecast time period.  Units of annual or multi-annual time scale
            depending on the averaging interval.
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
            Currently not in use... Ignore pre-calibrated LIM files.
        """

        match_prior = True
        datatag = 'ccsm4_last_millenium'

        avg_interval = 'annual_std'

        fcast_varnames = []
        prior_mapping = {}
        fcast_type = 'perfect'

        var_to_std_before_eof = None

        fcast_lead = 1
        fcast_num_pcs = 20
        dobj_num_pcs = 400
        detrend = True
        ignore_precalib_lim = False

        def __init__(self, regrid_cfg, lmr_path=None, prior_config=None,
                     **kwargs):

            super(ConfigGroup, self).__init__(**kwargs)

            self.regrid_cfg = regrid_cfg

            self.match_prior = self.match_prior

            self.var_to_std_before_eof = deepcopy(self.var_to_std_before_eof)

            if self.match_prior:
                self.datatag = prior_config.prior_source
                self.fcast_varnames = list(prior.state_variables.keys())
                self.prior_mapping = {state_var: state_var
                                      for state_var in self.fcast_varnames}
                self.avg_interval = prior.avg_interval
            else:
                self.datatag = self.datatag
                self.fcast_varnames = list(self.fcast_varnames)
                self.prior_mapping = deepcopy(self.prior_mapping)
                self.avg_interval = self.avg_interval

            dataset_descr = _DataInfo.get_info(self.datatag)
            self.datainfo = dataset_descr['info']
            self.datadir = dataset_descr['datadir']
            self.datafile = dataset_descr['datafile']
            self.dataformat = dataset_descr['dataformat']

            self.fcast_lead = self.fcast_lead
            self.fcast_num_pcs = self.fcast_num_pcs
            self.dobj_num_pcs = self.dobj_num_pcs
            self.detrend = self.detrend
            self.ignore_precalib_lim = self.ignore_precalib_lim

            self._avg_interval_defs = Constants.data['avg_interval']
            self.avg_interval_kwargs = self._avg_interval_defs[self.avg_interval]

            self.fcast_type = self.fcast_type

            if lmr_path is None:
                lmr_path = core.lmr_path

            self.lim_precalib_dir = join(lmr_path, 'lim_precalib_files')

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

        def update_avg_interval(self, avg_interval):
            self.avg_interval = avg_interval
            avg_kwargs = self._avg_interval_defs[avg_interval]
            self.avg_interval_kwargs = avg_kwargs

    def __init__(self, regrid_cfg, lmr_path=None,
                 prior_config=None, **kwargs):
        self.lim = self.lim(regrid_cfg,
                            lmr_path=lmr_path,
                            prior_config=prior_config,
                            **kwargs.pop('lim', {}))

        super(ConfigGroup, self).__init__(**kwargs)
        self.use_forecaster = self.use_forecaster


class regrid(ConfigGroup):
    """
    Parameters for regridding of gridded fields

    Attributes
    ----------
    ignore_pre_avg_file: bool
        Ignore pre-processed files for prior and anlaysis gridded data. This
        slows down the initial stages of reconstruction, but is useful for
        resetting the pre-processed data when used with save_pre_avg_file.
    save_pre_avg_file: bool
        Save intermediate files for loaded gridded datasets. Speeds up load
        times during reconstructions.
    regrid_method: str
        String indicating the method used to regrid the prior to lower spatial
        resolution.
        Allowed options are:
        1) None : Regridding NOT performed.
        2) 'spherical_harmonics' : Original regridding using pyspharm
           library. Cannot handle fields with masked/NaN data
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
    esmpy_regrid_to: str
        A grid defined in grid_def.yml to use as the regridding target.
        Currently supports 't42' and 'reg_4x5deg'.
    """

    ignore_pre_avg_file = False
    save_pre_avg_file = True

    regrid_method = 'esmpy'
    regrid_resolution = 42

    # ESMpy regridding options
    esmpy_interp_method = 'bilinear'
    esmpy_regrid_to = 't42'

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.ignore_pre_avg_file = self.ignore_pre_avg_file
        self.save_pre_avg_file = self.save_pre_avg_file

        self.regrid_method = self.regrid_method

        if self.regrid_method != 'esmpy':
            self.regrid_resolution = int(self.regrid_resolution)
            self.regrid_grid = self.regrid_resolution
            self.esmpy_interp_method = None
            self.esmpy_grid_def = None
        elif self.regrid_method == 'esmpy':
            self.regrid_resolution = None
            self.esmpy_interp_method = self.esmpy_interp_method
            self.esmpy_grid_def = _GridDef.get_info(self.esmpy_regrid_to)
            self.regrid_grid = self.esmpy_regrid_to

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

        self.regrid = regrid(**kwargs.pop('regrid', {}))
        self.proxies = proxies(lmr_path=lmr_path,
                               seed=seed,
                               **kwargs.pop('proxies', {}))
        self.psm = psm(self.regrid, lmr_path=lmr_path,
                       proxy_use_from=self.proxies.use_from,
                       **kwargs.pop('psm', {}))
        self.prior = prior(self.regrid, lmr_path=lmr_path,
                           seed=seed,
                           **kwargs.pop('prior', {}))

        self.forecaster = forecaster(self.regrid, lmr_path=lmr_path,
                                     prior_config=self.prior,
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


_avg_type_pre_calib_tag = {'annual': '',
                           ('seasonal', 'proxy_metadata'): 'META',
                           ('seasonal', 'psm_calib'): 'PSM'}


if __name__ == "__main__":
    # kwargs = {'wrapper': {'multi_seed': [1, 2, 3]},
    #           'psm': {'linear': {'datatag': 'BE'}}}
    tmp = Config()
    pass
