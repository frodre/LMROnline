"""
Module: LMR_proxy_pandas_rework.py

Purpose: Module containing various classes associated with proxy types to be
         assimilated in the LMR, as well as numerous functionalities for
         selection of proxy types/sites to be included in the reanalysis.

         Rewritten by AndreP to incorporate features from the original
         LMR_proxy code using OOP and Pandas. Is used by the driver but not by
         verification scripts.

Originator: Andre Perkins, U. of Washington.

Revisions:
         - Added capability to filter uploaded *NCDC proxies* according to the
           database they are included in (PAGES1, PAGES2, or LMR_FM). This
           information is found in the metadata, as extracted from the
           NCDC-templated text files.
           [ R. Tardif, U. Washington, March 2016 ]
         - Added capability to filter out *NCDC proxies* listed in a blacklist.
           This is mainly used to prevent the assimilation of chronologies known
           to be duplicates.
           [ R. Tardif, U. Washington, March 2016 ]
         - Added capability to select proxies according to data availability over
           the reconstruction period.
           [ R. Tardif, U. Washington, October 2016 ]
         - Added class for low-resolution marine proxies used for LGM & Holocene
           reconstructions (NCDCdtda class). 
           [ R. Tardif, U. Washington, January 2017 ]
         - Renamed the proxy databases to less-confusing convention. 
           'pages' renamed as 'PAGES2kv1' and 'NCDC' renamed as 'LMRdb'
           [ R. Tardif, Univ. of Washington, Sept 2017 ]
"""

import LMR_psms
from LMR_utils2 import augment_docstr, class_docs_fixer, fix_lon

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
import random
import pandas as pd
import numpy as np

class ProxyManager:
    """
    High-level manager to handle loading proxies from multiple sources and
    randomly sampling the proxies.

    Attributes
    ----------
    all_proxies: list(BaseProxyObject like)
        A list of all proxy objects loaded for current reconstruction
    all_ids_by_group: dict{str: list(str)}
        A dictionary holding list of proxy site ids for each proxy type loaded.
    ind_assim: list(int)
        List of indices (pretaining to all_proxies)to be assimilated during the
        reconstruction.
    ind_eval: list(int)
        List of indices of proxies withheld for verification purposes.

    Parameters
    ----------
    config: LMR_config
        Configuration module for current LMR run
    data_range: list(int)
        A two int list defining the beginning and ending time of the
        reconstruction
    """

    def __init__(self, proxy_config, data_range, psm_config):

        self.all_ids_by_group = defaultdict(list)

        pclass = get_proxy_class(proxy_config.use_from)

        # Load proxies for current class
        ids_by_grp, proxies = pclass.load_all(proxy_config,
                                              data_range,
                                              psm_config)

        self.all_proxies = proxies
        self.all_ids_by_group = ids_by_grp
        proxy_frac = proxy_config.proxy_frac
        nsites = len(self.all_proxies)

        # Sample subset from all proxies if specified
        if proxy_frac < 1.0:
            nsites_assim = int(nsites * proxy_frac)

            # Set seed if set in configuration
            random.seed(proxy_config.seed)
            self.ind_assim = random.sample(list(range(nsites)), nsites_assim)
            self.ind_assim.sort()
            self.ind_eval = list(set(range(nsites)) - set(self.ind_assim))
            self.ind_eval.sort()

            # Make list of assimilated proxies by group
            self.assim_ids_by_group = deepcopy(self.all_ids_by_group)
            for idx in self.ind_eval:
                pobj = self.all_proxies[idx]
                grp = self.assim_ids_by_group[pobj.type]
                if pobj.id in grp:
                    grp.remove(pobj.id)

        else:
            self.ind_assim = list(range(nsites))
            self.ind_eval = None
            self.assim_ids_by_group = self.all_ids_by_group

    def proxy_obj_generator(self, indexes):
        """
        Generator to iterate over proxy objects in list at specified indexes

        Parameters
        ----------
        indexes: list(int)
            List of indices pertaining to self.all_proxies

        Returns
        -------
        generator
            A generator over all_proxies for each specified index
        """

        for idx in indexes:
            yield self.all_proxies[idx]

    def sites_assim_proxy_objs(self):
        """
        Generator over ind_assim indices.

        Yields
        ------
        BaseProxyObject like
            Proxy object from the all_proxies list.
        """

        return self.proxy_obj_generator(self.ind_assim)

    def sites_eval_proxy_objs(self):
        """
        Generator over ind_eval indices.

        Yields
        ------
        BaseProxyObject like
            Proxy object from the all_proxies list.
        """
        if self.ind_eval:
            return self.proxy_obj_generator(self.ind_eval)
        else:
            return []


class BaseProxyObject(metaclass=ABCMeta):
    """
    Class defining attributes and methods for descendant proxy objects.

    Attributes
    ----------
    id: str
        Proxy site id
    type: str
        Proxy type
    start_yr: int
        Earliest year that data exists for this site
    end_yr: int
        Latest year that data exists for this site
    values: pandas.DataFrame
        Proxy record with time (in years) as the index
    lat: float
        Latitude of proxy site
    lon: float
        Longitude of proxy site
    elev: float
        Elevation/depth of proxy site
    time: ndarray
        List of times for which proxy contains valid data
    psm_obj: BasePSM like
        PSM for this proxy
    psm: function
        Exposed psm mapping function from the psm_obj

    Parameters
    ----------
    config: LMR_config
        Configuration module for current LMR run
    pid -> id
    prox_type -> type
    start_yr
    end_yr
    lat
    lon
    elev
    values
    time

    Notes
    -----
    All proxy object classes should descend from the BaseProxyObject abstract
    class.
    """

    def __init__(self, psm_config, psm_type, pid, prox_type, start_yr, end_yr,
                 lat, lon, elev, seasonality, values, time, load_psm_obj=True):

        if (values is None) or len(values) == 0:
            raise ValueError('No proxy data given for object initialization')

        assert len(values) == len(time), 'Time and value dimensions must match'

        self.id = pid
        self.type = prox_type
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.values = values
        self.lat = lat
        self.lon = fix_lon(lon)
        self.elev = elev
        self.time = time
        self.seasonality = seasonality

        # Retrieve appropriate PSM function
        self.psm_config = psm_config
        if load_psm_obj:
            psm_obj = LMR_psms.get_psm_class(psm_type)
            self.psm_obj = psm_obj(psm_config, self)
            self.psm = self.psm_obj.psm

    def _constrain_to_date_range(self, date_range):
        start, end = date_range

        site_data = self.values
        site_data = site_data[(site_data.index >= start) &
                              (site_data.index <= end)]
        self.values = site_data
        self.time = site_data.index.values

    @staticmethod
    @abstractmethod
    def get_psm_obj(psm_config, proxy_obj):
        """ Retrieves PSM object class to be attached to this proxy"""
        pass

    @classmethod
    @abstractmethod
    def load_site(cls,  config, site, psm_config,
                  data_range=None, meta_src=None,
                  data_src=None):
        """
        Load proxy object from single site.

        Parameters
        ----------
        config: LMR_config
            Configuration for current LMR run
        site: str
            Key to identify which site to load from source data
        meta_src: optional
            Source for proxy metadata
        data_src: optional
            Source for proxy record data (might be same as meta_src)
        data_range: iterable
            Two-item container holding beginning and end date of reconstruction
        psm_kwargs: dict(unpacked)
            Keyword arguments to be passed to the psm.

        Returns
        -------
        BaseProxyObject like
            Proxy object instance at specified site

        Notes
        -----
        If source data not specified, it should attempt to load data using
        config file information.
        """
        pass

    @classmethod
    @abstractmethod
    def load_all(cls, proxy_config, data_range, psm_config):
        """
        Load proxy objects from all sites matching filter criterion.

        Parameters
        ----------
        proxy_config: LMR_config
            Configuration for current LMR run
        meta_src: optional
            Source for proxy metadata
        data_src: optional
            Source for proxy record data (might be same as meta_src)
        data_range: iterable
            Two-item container holding beginning and end date of reconstruction
        psm_kwargs: dict(unpacked)
            Keyword arguments to be passed to the psm.

        Returns
        -------
        dict
            Dictionary of proxy types (keys) with associated site ids (values)
        list
            List of all proxy objects created

        Notes
        -----
        If source data not specified, it should attempt to load data using
        config file information.
        """
        pass

    @abstractmethod
    def error(self):
        """
        error model for proxy data
        """
        pass


@class_docs_fixer
class ProxyPAGES2kv1(BaseProxyObject):

    @staticmethod
    def get_psm_obj(pages2kv1_cfg, proxy_type):
        psm_key = pages2kv1_cfg.proxy_psm_type[proxy_type]
        return LMR_psms.get_psm_class(psm_key)

    @classmethod
    @augment_docstr
    def load_site(cls, pages2kv1_cfg, site, psm_config,
                  data_range=None,
                  meta_src=None,
                  data_src=None, load_psm=True):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

        if meta_src is None:
            meta_src = pd.read_pickle(pages2kv1_cfg.metafile_proxy)
        if data_src is None:
            data_src = pd.read_pickle(pages2kv1_cfg.datafile_proxy)

        site_meta = meta_src[meta_src['Proxy ID'] == site]
        pid = site_meta['Proxy ID'].iloc[0]
        pmeasure = site_meta['Proxy measurement'].iloc[0]
        pages2kv1_type = site_meta['Archive type'].iloc[0]
        try:
            proxy_type = pages2kv1_cfg.proxy_type_mapping[(pages2kv1_type,
                                                           pmeasure)]
            psm_type = pages2kv1_cfg.proxy_psm_type[proxy_type]
        except (KeyError, ValueError) as e:
            print('Proxy type/measurement not found in mapping: {}'.format(e))
            raise ValueError(e)

        start_yr = site_meta['Youngest (C.E.)'].iloc[0]
        end_yr = site_meta['Oldest (C.E.)'].iloc[0]
        lat = site_meta['Lat (N)'].iloc[0]
        lon = site_meta['Lon (E)'].iloc[0]
        elev = 0.0  # elev not info available in PAGES2kS1 data
        seasonality = None  # not defined in PAGES2kS1 metadata
        site_data = data_src[site]

        if data_range is not None:
            start, finish = data_range
            values = site_data[(site_data.index >= start) &
                               (site_data.index <= finish)]
        else:
            values = site_data

        # Might need to remove following line
        values = values[values.notnull()]
        times = values.index.values

        # transform in "anomalies" (time-mean removed) if option activated
        if pages2kv1_cfg.proxy_timeseries_kind == 'anom':
            values = values - values.mean()

        # Send full proxy timeseries in case calibration is necessary
        proxy_obj = cls(psm_config, psm_type, pid, proxy_type, start_yr,
                        end_yr, lat, lon, elev, seasonality, values, times,
                        load_psm_obj=load_psm)

        proxy_obj._constrain_to_date_range(data_range)

        if len(proxy_obj.values) == 0:
            raise ValueError('No observations in specified time range.')

        return proxy_obj

    @classmethod
    @augment_docstr
    def load_all(cls, proxy_config, data_range, psm_config,
                 meta_src=None, data_src=None):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

        pages2kv1_cfg = proxy_config.PAGES2kv1
        load_psm = proxy_config.load_psm_with_proxies

        # Load source data files
        if meta_src is None:
            meta_src = pd.read_pickle(pages2kv1_cfg.metafile_proxy)
        if data_src is None:
            data_src = pd.read_pickle(pages2kv1_cfg.datafile_proxy)

        filters = pages2kv1_cfg.simple_filters
        proxy_order = pages2kv1_cfg.proxy_order
        ptype_filters = pages2kv1_cfg.proxy_assim2
        availability_filter = pages2kv1_cfg.proxy_availability_filter
        availability_fraction = pages2kv1_cfg.proxy_availability_fraction

        # initial masks all true before filtering
        useable = meta_src[meta_src.columns[0]] == 0
        useable |= True
        availability_mask = meta_src[meta_src.columns[0]] == 0
        availability_mask |= True

        # Find indices matching simple filter specifications
        for colname, filt_list in filters.items():
            simple_mask = meta_src[colname] == 0
            simple_mask &= False

            for value in filt_list:
                simple_mask |= meta_src[colname] == value

            useable &= simple_mask

        # Filtering proxy records on conditions of availability during
        # the reconstruction period (recon_period in configuration, or
        # data_range here).
        if availability_filter:
            start, finish = data_range
            # Checking proxy metadata's period of availability against
            # reconstruction period.
            availability_mask = ((meta_src['Oldest (C.E.)'] <= start) &
                                 (meta_src['Youngest (C.E.)'] >= finish))
            # Checking level of completeness of record within the reconstruction
            # period (ignore record if fraction of available data is below
            # user-defined threshold (proxy_availability_fraction in config).
            maxnb = (finish - start) + 1
            proxies_to_test = meta_src['Proxy ID'][availability_mask & useable].values
            for prx in proxies_to_test.tolist():
                values = data_src[prx][(data_src[:].index >= start) & (data_src[:].index <= finish)]
                values = values[values.notnull()]
                frac_available = float(len(values))/float(maxnb)
                if frac_available < availability_fraction:
                    availability_mask[meta_src[meta_src['Proxy ID'] == prx].index] = False

        # Create proxy id lists
        proxy_id_by_type = {}
        all_proxy_ids = []

        type_col = 'Archive type'
        measure_col = 'Proxy measurement'
        for name in proxy_order:

            type_mask = meta_src[type_col] == 0
            type_mask |= True

            # Filter to proxies of a certain type
            ptype = name.split('_', 1)[0]
            type_mask &= meta_src[type_col] == ptype

            # Reduce to listed measures
            measure_mask = meta_src[measure_col] == 0
            measure_mask &= False

            for measure in ptype_filters[name]:
                measure_mask |= meta_src[measure_col] == measure

            # Extract proxy ids using mask and append to lists
            proxies = meta_src['Proxy ID'][measure_mask & type_mask &
                                           availability_mask & useable].values

            # If we have ids after filtering add them to the type list
            if len(proxies) > 0:
                proxy_id_by_type[name] = proxies.tolist()

            all_proxy_ids += proxies.tolist()

        # Create proxy objects list
        all_proxies = []
        for site in all_proxy_ids:
            try:
                pobj = cls.load_site(pages2kv1_cfg, site, psm_config,
                                     data_range=data_range,
                                     meta_src=meta_src, data_src=data_src,
                                     load_psm=load_psm)
                all_proxies.append(pobj)
            except ValueError as e:
                # Proxy had no obs or didn't meet psm r crit
                for group in list(proxy_id_by_type.values()):
                    if site in group:
                        group.remove(site)
                        break  # Should only be one instance

        return proxy_id_by_type, all_proxies

    @classmethod
    def load_all_annual_no_filtering(cls, proxy_config, meta_src=None,
                                     data_src=None):
        """
        Method created to facilitate the loading of all possible proxy records
        that can be calibrated with annual resolution.

        Note: This is still subject to constraints from the PSM calibration (
        i.e. if there is an r_crit or not enough calibration data the proxy
        will not be loaded)

        Returns
        -------
        proxy_objs: list(BaseProxyObject like)
        """

        pages2kv1_cfg = proxy_config.PAGES2kv1
        # TODO: What should psm loading do in this case?
        load_psm = proxy_config.load_psm_with_proxies

        # Load source data files
        if meta_src is None:
            meta_src = pd.read_pickle(pages2kv1_cfg.metafile_proxy)
        if data_src is None:
            data_src = pd.read_pickle(pages2kv1_cfg.datafile_proxy)

        # TODO: For now hard coded to annual resolution - AP
        useable = meta_src['Resolution (yr)'] == 1.0

        proxy_ids = meta_src['Proxy ID'][useable].values

        proxy_objs = []
        for site in proxy_ids:
            try:
                pobj = cls.load_site(proxy_config, site,
                                     meta_src=meta_src, data_src=data_src,
                                     load_psm=False)
                proxy_objs.append(pobj)
            except ValueError as e:
                print(e)

        return proxy_objs

    def error(self):
        # Constant error for now
        return 0.1


class ProxyLMRdb(BaseProxyObject):
    @staticmethod
    def get_psm_obj(proxy_config, proxy_type):
        psm_key = proxy_config.LMRdb.proxy_psm_type[proxy_type]
        return LMR_psms.get_psm_class(psm_key)

    @classmethod
    @augment_docstr
    def load_site(cls, lmr_db_cfg, site, psm_config, data_range=None,
                  meta_src=None, data_src=None, load_psm=True, anomly=False):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

        if meta_src is None:
            meta_src = pd.read_pickle(lmr_db_cfg.metafile_proxy)
        if data_src is None:
            data_src = pd.read_pickle(lmr_db_cfg.datafile_proxy)

        site_meta = meta_src[meta_src['Proxy ID'] == site]
        pid = site_meta['Proxy ID'].iloc[0]
        pmeasure = site_meta['Proxy measurement'].iloc[0]
        LMRdb_type = site_meta['Archive type'].iloc[0]
        try:
            proxy_type = lmr_db_cfg.proxy_type_mapping[(LMRdb_type, pmeasure)]
            psm_type = lmr_db_cfg.proxy_psm_type[proxy_type]
        except (KeyError, ValueError) as e:
            print('Proxy type/measurement not found in mapping: {}'.format(e))
            raise ValueError(e)

        start_yr = site_meta['Youngest (C.E.)'].iloc[0]
        end_yr = site_meta['Oldest (C.E.)'].iloc[0]
        lat = site_meta['Lat (N)'].iloc[0]
        lon = site_meta['Lon (E)'].iloc[0]
        elev = site_meta['Elev'].iloc[0]
        site_data = data_src[site]
        seasonality = site_meta['Seasonality'].iloc[0]
        # make sure a list is returned
        if type(seasonality) is not list: seasonality = ast.literal_eval(
            seasonality)

        if data_range is not None:
            start, finish = data_range
            values = site_data[(site_data.index >= start) &
                               (site_data.index <= finish)]
        else:
            values = site_data

        # Might need to remove following line
        values = values[values.notnull()]
        times = values.index.values

        # transform in "anomalies" (time-mean removed) if option activated
        if anomly:
            values = values - values.mean()

        if len(values) == 0:
            raise ValueError('No observations in specified time range.')

        return cls(psm_config, psm_type,
                   pid, proxy_type, start_yr, end_yr, lat, lon,
                   elev, seasonality, values, times, load_psm_obj=load_psm)

    @classmethod
    @augment_docstr
    def load_all(cls, proxy_config, data_range, psm_config, meta_src=None,
                 data_src=None):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

        lmr_db_cfg = proxy_config.LMRdb
        load_psm = proxy_config.load_psm_with_proxies
        if proxy_config.proxy_timeseries_kind == 'anom':
            convert_anomly = True
        else:
            convert_anomly = False


        # Load source data files
        if meta_src is None:
            meta_src = pd.read_pickle(lmr_db_cfg.metafile_proxy)
        if data_src is None:
            data_src = pd.read_pickle(lmr_db_cfg.datafile_proxy)

        filters = lmr_db_cfg.simple_filters
        proxy_order = lmr_db_cfg.proxy_order
        ptype_filters = lmr_db_cfg.proxy_assim2
        dbase_filters = lmr_db_cfg.database_filter
        proxy_blacklist = lmr_db_cfg.proxy_blacklist
        availability_filter = lmr_db_cfg.proxy_availability_filter
        availability_fraction = lmr_db_cfg.proxy_availability_fraction

        # initial mask all true before filtering
        useable = meta_src[meta_src.columns[0]] == 0
        useable |= True
        availability_mask = meta_src[meta_src.columns[0]] == 0
        availability_mask |= True

        # Find indices matching simple filter specifications
        for colname, filt_list in filters.items():
            simple_mask = meta_src[colname] == 0
            simple_mask &= False

            for value in filt_list:
                simple_mask |= meta_src[colname] == value

            useable &= simple_mask

        # Filtering proxy records on conditions of availability during
        # the reconstruction period (recon_period in configuration, or
        # data_range here).
        if availability_filter:  # if not None
            start, finish = data_range
            # Checking proxy metadata's period of availability against
            # reconstruction period.
            availability_mask = ((meta_src['Oldest (C.E.)'] <= start) &
                                 (meta_src['Youngest (C.E.)'] >= finish))
            # Checking level of completeness of record within the reconstruction
            # period (ignore record if fraction of available data is below user-defined
            # threshold (proxy_availability_fraction in config).
            maxnb = (finish - start) + 1
            proxies_to_test = meta_src['Proxy ID'][
                availability_mask & useable].values
            for prx in proxies_to_test.tolist():
                values = data_src[prx][(data_src[:].index >= start) & (
                data_src[:].index <= finish)]
                values = values[values.notnull()]
                frac_available = float(len(values)) / float(maxnb)
                if frac_available < availability_fraction:
                    availability_mask[
                        meta_src[meta_src['Proxy ID'] == prx].index] = False

        # Find indices matching **database filter** specifications
        database_col = 'Databases'

        # dbase_filters not "None" or empty list (some selection on db has been activated)
        if dbase_filters:
            # define boolean array with right dimension & set all to False
            dbase_mask = meta_src[database_col] == 0
            # set mask to True for proxies matching all databases found in dbase_filters
            for i in range(len(meta_src[database_col])):
                if meta_src[database_col][i]:
                    # dbase_mask[i] = set(meta_src[database_col][i]).isdisjoint(dbase_filters) # oldold code
                    # dbase_mask[i] = set(dbase_filters).issubset(meta_src[database_col][i]) # old code
                    dbase_mask[i] = bool(
                        set(meta_src[database_col][i]).intersection(
                            set(dbase_filters)))
                else:
                    dbase_mask[i] = False
        else:
            # selection on db has NOT been activated:
            # define boolean array with right dimension & set all to True
            dbase_mask = meta_src[database_col] != 0

        # Define mask of proxies listed in a user-defined "blacklist"
        # (see LMR_config).
        # boolean array set with right dimension & all set to True
        blacklist_mask = meta_src['Proxy ID'] != ' '
        if proxy_blacklist:
            # If site listed in blacklist, modify corresponding elements of
            # boolean array to False
            for pbl in proxy_blacklist:
                tmp = meta_src['Proxy ID'].map(lambda x: x.startswith(pbl))
                inds = meta_src['Proxy ID'][tmp].index
                blacklist_mask[inds] = False

        # Create proxy id lists
        proxy_id_by_type = {}
        all_proxy_ids = []

        type_col = 'Archive type'
        measure_col = 'Proxy measurement'
        for name in proxy_order:

            type_mask = meta_src[type_col] == 0
            type_mask |= True

            # Filter to proxies of a certain type
            ptype = name.split('_', 1)[0]
            type_mask &= meta_src[type_col] == ptype

            # Reduce to listed measures
            measure_mask = meta_src[measure_col] == 0
            measure_mask &= False

            for measure in ptype_filters[name]:
                measure_mask |= meta_src[measure_col] == measure

            # Extract proxy ids using mask and append to lists
            proxies = meta_src['Proxy ID'][measure_mask & type_mask &
                                           dbase_mask & blacklist_mask &
                                           availability_mask & useable].values
            # If we have ids after filtering add them to the type list
            if len(proxies) > 0:
                proxy_id_by_type[name] = proxies.tolist()

            all_proxy_ids += proxies.tolist()

        # Create proxy objects list
        all_proxies = []
        for site in all_proxy_ids:
            try:
                pobj = cls.load_site(lmr_db_cfg, site, psm_config,
                                     data_range=data_range,
                                     meta_src=meta_src, data_src=data_src,
                                     load_psm=load_psm, anomly=convert_anomly)
                all_proxies.append(pobj)
            except ValueError as e:
                # Proxy had no obs or didn't meet psm r crit
                for group in list(proxy_id_by_type.values()):
                    if site in group:
                        group.remove(site)
                        break  # Should only be one instance

        return proxy_id_by_type, all_proxies

    @classmethod
    def load_all_annual_no_filtering(cls, proxy_config, meta_src=None,
                                     data_src=None):
        """
        Method created to facilitate the loading of all possible proxy records
        that can be calibrated with annual resolution.

        Note: This is still subject to constraints from the PSM calibration (
        i.e. if there is an r_crit or not enough calibration data the proxy
        will not be loaded)

        Returns
        -------
        proxy_objs: list(BaseProxyObject like)
        """

        lmr_db_cfg = proxy_config.LMRdb

        # Load source data files
        if meta_src is None:
            meta_src = pd.read_pickle(lmr_db_cfg.metafile_proxy)
        if data_src is None:
            data_src = pd.read_pickle(lmr_db_cfg.datafile_proxy)

        # TODO: For now hard coded to annual resolution - AP
        useable = meta_src['Resolution (yr)'] == 1.0

        proxy_ids = meta_src['Proxy ID'][useable].values

        proxy_objs = []
        for site in proxy_ids:
            try:
                pobj = cls.load_site(proxy_config, site,
                                     meta_src=meta_src, data_src=data_src)
                proxy_objs.append(pobj)
            except ValueError as e:
                print(e)

        return proxy_objs

    def error(self):
        # Constant error for now
        return 0.1


class ProxyNCDCdtda(BaseProxyObject):

    @staticmethod
    def get_psm_obj(config,proxy_type):
        psm_key = config.proxies.ncdcdtda.proxy_psm_type[proxy_type]
        return LMR_psms.get_psm_class(psm_key)

    @classmethod
    @augment_docstr
    def load_site(cls, config, site, psm_config,
                  data_range=None, meta_src=None, data_src=None):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

        ncdcdtda_cfg = config.proxies.ncdcdtda
        if meta_src is None:
            meta_src = load_data_frame(ncdcdtda_cfg.metafile_proxy)
        if data_src is None:
            data_src = load_data_frame(ncdcdtda_cfg.datafile_proxy)

        site_meta = meta_src[meta_src['Proxy ID'] == site]
        pid = site_meta['Proxy ID'].iloc[0]
        pmeasure = site_meta['Proxy measurement'].iloc[0]
        NCDCdtda_type = site_meta['Archive type'].iloc[0]
        try:
            proxy_type = ncdcdtda_cfg.proxy_type_mapping[(NCDCdtda_type,pmeasure)]
        except (KeyError, ValueError) as e:
            print('Proxy type/measurement not found in mapping: {}'.format(e))
            raise ValueError(e)

        start_yr = site_meta['Youngest (C.E.)'].iloc[0]
        end_yr = site_meta['Oldest (C.E.)'].iloc[0]
        lat = site_meta['Lat (N)'].iloc[0]
        lon = site_meta['Lon (E)'].iloc[0]
        elev = site_meta['Elev'].iloc[0]
        site_data = data_src[site]
        seasonality = site_meta['Seasonality'].iloc[0]

        # if field exists, make sure a list is returned for seasonality
        if seasonality:
            if type(seasonality) is not list: seasonality = ast.literal_eval(seasonality)
        else:
            seasonality = None

        if data_range is not None:
            start, finish = data_range
            values = site_data[(site_data.index >= start) &
                               (site_data.index <= finish)]
        else:
            values = site_data

        # Might need to remove following line
        values = values[values.notnull()]
        times = values.index.values

        # transform in "anomalies" (time-mean removed) if option activated
        if config.proxies.ncdcdtda.proxy_timeseries_kind == 'anom':
            values = values - values.mean()

        if len(values) == 0:
            raise ValueError('No observations in specified time range.')

        return cls(config, pid, proxy_type, start_yr, end_yr, lat, lon, elev,
                   seasonality, values, times)

    @classmethod
    @augment_docstr
    def load_all(cls, config, data_range, meta_src=None,
                 data_src=None):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

        # Load source data files
        if meta_src is None:
            meta_src = load_data_frame(config.proxies.ncdcdtda.metafile_proxy)
        if data_src is None:
            data_src = load_data_frame(config.proxies.ncdcdtda.datafile_proxy)

        filters = config.proxies.ncdcdtda.simple_filters
        proxy_order = config.proxies.ncdcdtda.proxy_order
        ptype_filters = config.proxies.ncdcdtda.proxy_assim2
        dbase_filters = config.proxies.ncdcdtda.database_filter
        proxy_blacklist = config.proxies.ncdcdtda.proxy_blacklist
        availability_filter = config.proxies.ncdcdtda.proxy_availability_filter
        availability_fraction = config.proxies.ncdcdtda.proxy_availability_fraction

        # initial mask all true before filtering
        useable = meta_src[meta_src.columns[0]] == 0
        useable |= True
        availability_mask = meta_src[meta_src.columns[0]] == 0
        availability_mask |= True

        # Find indices matching simple filter specifications
        for colname, filt_list in filters.items():

            simple_mask = meta_src[colname] == 0
            simple_mask &= False

            for value in filt_list:
                if colname == 'Resolution (yr)' and type(value) is tuple:
                    for i in range(len(meta_src[colname].index)):
                        simple_mask[i] |= (value[0] <= meta_src[colname][i] <= value[1])
                else:
                    simple_mask |= meta_src[colname] == value

            useable &= simple_mask


        # Filtering proxy records on conditions of availability during
        # the reconstruction period (recon_period in configuration, or
        # data_range here).
        if availability_filter: # if not None
            start, finish = data_range
            # Checking proxy metadata's period of availability against
            # reconstruction period.
            availability_mask = ((meta_src['Oldest (C.E.)'] <= start) &
                                 (meta_src['Youngest (C.E.)'] >= finish))
            # Checking level of completeness of record within the reconstruction
            # period (ignore record if fraction of available data is below user-defined
            # threshold (proxy_availability_fraction in config).
            maxnb = (finish - start) + 1
            proxies_to_test = meta_src['Proxy ID'][availability_mask & useable].values
            for prx in proxies_to_test.tolist():
                values = data_src[prx][(data_src[:].index >= start) & (data_src[:].index <= finish)]
                values = values[values.notnull()]
                frac_available = float(len(values))/float(maxnb)
                if frac_available < availability_fraction:
                    availability_mask[meta_src[meta_src['Proxy ID'] == prx].index] = False

        # Find indices matching **database filter** specifications
        database_col = 'Databases'

        # dbase_filters not "None" or empty list (some selection on db has been activated)
        if dbase_filters:
            # define boolean array with right dimension & set all to False
            dbase_mask = meta_src[database_col] == 0
            # set mask to True for proxies matching all databases found in dbase_filters
            for i in range(len(meta_src[database_col])):
                if meta_src[database_col][i]:
                    #dbase_mask[i] = set(meta_src[database_col][i]).isdisjoint(dbase_filters) # oldold code
                    #dbase_mask[i] = set(dbase_filters).issubset(meta_src[database_col][i]) # old code
                    dbase_mask[i] = bool(set(meta_src[database_col][i]).intersection(set(dbase_filters)))
                else:
                    dbase_mask[i] = False
        else:
            # selection on db has NOT been activated:
            # define boolean array with right dimension & set all to True
            dbase_mask = meta_src[database_col] != 0


        # Define mask of proxies listed in a user-defined "blacklist"
        # (see LMR_config).
        # boolean array set with right dimension & all set to True
        blacklist_mask = meta_src['Proxy ID'] != ' '
        if proxy_blacklist:
            # If site listed in blacklist, modify corresponding elements of
            # boolean array to False
            for pbl in proxy_blacklist:
                tmp = meta_src['Proxy ID'].map(lambda x: x.startswith(pbl))
                inds = meta_src['Proxy ID'][tmp].index
                blacklist_mask[inds] = False

        # Create proxy id lists
        proxy_id_by_type = {}
        all_proxy_ids = []

        type_col = 'Archive type'
        measure_col = 'Proxy measurement'
        for name in proxy_order:

            type_mask = meta_src[type_col] == 0
            type_mask |= True

            # Filter to proxies of a certain type
            ptype = name.split('_', 1)[0]
            type_mask &= meta_src[type_col] == ptype

            # Reduce to listed measures
            measure_mask = meta_src[measure_col] == 0
            measure_mask &= False

            for measure in ptype_filters[name]:
                measure_mask |= meta_src[measure_col] == measure

            # Extract proxy ids using mask and append to lists
            proxies = meta_src['Proxy ID'][measure_mask & type_mask &
                                          dbase_mask & blacklist_mask &
                                          availability_mask & useable].values

            # If we have ids after filtering add them to the type list
            if len(proxies) > 0:
                proxy_id_by_type[name] = proxies.tolist()

            all_proxy_ids += proxies.tolist()

        # Create proxy objects list
        all_proxies = []
        for site in all_proxy_ids:
            try:
                pobj = cls.load_site(config, site, data_range,
                                     meta_src=meta_src, data_src=data_src)
                all_proxies.append(pobj)
            except ValueError as e:
                # Proxy had no obs or didn't meet psm r crit
                for group in list(proxy_id_by_type.values()):
                    if site in group:
                        group.remove(site)
                        break  # Should only be one instance

        return proxy_id_by_type, all_proxies

    @classmethod
    def load_all_annual_no_filtering(cls, config, meta_src=None,
                                     data_src=None):
        """
        Method created to facilitate the loading of all possible proxy records
        that can be calibrated with annual resolution.

        Note: This is still subject to constraints from the PSM calibration (
        i.e. if there is an r_crit or not enough calibration data the proxy
        will not be loaded)

        Returns
        -------
        proxy_objs: list(BaseProxyObject like)
        """

        # Load source data files
        if meta_src is None:
            meta_src = load_data_frame(config.proxies.ncdcdtda.metafile_proxy)
        if data_src is None:
            data_src = load_data_frame(config.proxies.ncdcdtda.datafile_proxy)

        # set for any resolution
        useable = meta_src['Resolution (yr)'] > 0.0

        proxy_ids = meta_src['Proxy ID'][useable].values

        proxy_objs = []
        for site in proxy_ids:
            try:
                pobj = cls.load_site(config, site,
                                     meta_src=meta_src, data_src=data_src)
                proxy_objs.append(pobj)
            except ValueError as e:
                print(e)

        return proxy_objs

    def error(self):
        # Constant error for now
        return 0.1




def fix_lon(lon):
    """
    Fixes negative longitude values.

    Parameters
    ----------
    lon: ndarray like or value
        Input longitude array or single value
    """
    if lon < 0:
        lon += 360.
    return lon


_proxy_classes = {'PAGES2kv1': ProxyPAGES2kv1,
                  'LMRdb': ProxyLMRdb,
                  'NCDCdtda': ProxyNCDCdtda}

def get_proxy_class(proxy_key):
    """
    Retrieve proxy class type to be instantiated.

    Parameters
    ----------
    proxy_key: str
        Dict key to retrieve correct PSM class type.

    Returns
    -------
    BaseProxyObject like:
        Class type to be instantiated and attached to a proxy.
    """
    return _proxy_classes[proxy_key]
