import sys
sys.path.append('../')

import pytest
import yaml
import LMR_config as cfg
import os
import numpy as np

from copy import deepcopy


@pytest.fixture(scope='function')
def data_descr(request):
    with open(os.path.join(cfg.SRC_DIR, 'datasets.yml'), 'r') as f:
        tmp = yaml.load(f)

    return tmp

@pytest.fixture(scope='module')
def constant_def(request):
    filepath = os.path.join(cfg.SRC_DIR, 'tests', 'test_constants.yml')
    constants = cfg._ConstantDefinitions(filename=filepath)

    return constants


# Test that Config instance has all configuration objects attached and that
# core has the correct instance only attributes
def test_default_configuration_core():
    cfg_object = cfg.Config()

    attrs = ['wrapper', 'core', 'proxies', 'psm', 'prior', 'forecaster']
    for attr in attrs:
        assert hasattr(cfg_object, attr)

    assert cfg.core.lmr_path == cfg_object.core.lmr_path
    assert hasattr(cfg_object.core, 'curr_iter')

    assert hasattr(cfg_object.prior, 'datadir')

# Test that the seed is propagated to the correct configuration modules.
def test_default_configuration_seed():
    new_seed = 1234
    update_dict = {'core': {'seed': new_seed}}
    cfg_object = cfg.Config(**update_dict)

    assert hasattr(cfg_object.core, 'seed')
    assert hasattr(cfg_object.proxies, 'seed')
    assert hasattr(cfg_object.prior, 'seed')

    assert cfg_object.proxies.seed == new_seed
    assert cfg_object.prior.seed == new_seed

# Test that the instance only attributes in pages are set
def test_default_configuration_proxies_pages():
    cfg_object = cfg.Config()

    assert hasattr(cfg_object.proxies, 'PAGES2kv1')
    assert hasattr(cfg_object.proxies.PAGES2kv1, 'datadir_proxy')
    assert hasattr(cfg_object.proxies.PAGES2kv1, 'datafile_proxy')
    assert hasattr(cfg_object.proxies.PAGES2kv1, 'metafile_proxy')
    assert hasattr(cfg_object.proxies.PAGES2kv1, 'proxy_type_mapping')
    assert hasattr(cfg_object.proxies.PAGES2kv1, 'simple_filters')

    assert cfg_object.core.lmr_path in cfg_object.proxies.PAGES2kv1.datadir_proxy


# Test that the instance only attributes of the linear psm are set
def test_default_configuration_psm_linear():
    cfg_object = cfg.Config()

    assert hasattr(cfg_object.psm, 'linear')
    assert hasattr(cfg_object.psm.linear, 'datadir')
    assert hasattr(cfg_object.psm.linear, 'datainfo')
    assert hasattr(cfg_object.psm.linear, 'datafile')
    assert hasattr(cfg_object.psm.linear, 'dataformat')
    assert hasattr(cfg_object.psm.linear, 'season_source')
    assert hasattr(cfg_object.psm.linear, 'avg_interval')
    assert hasattr(cfg_object.psm.linear, 'psm_required_variables')

    assert cfg_object.core.lmr_path in cfg_object.psm.linear.datadir


def test_default_configuration_psm_linear_t_or_p():
    cfg_object = cfg.Config()

    assert hasattr(cfg_object.psm, 'linear_TorP')
    assert hasattr(cfg_object.psm.linear_TorP, 'moisture')
    assert hasattr(cfg_object.psm.linear_TorP, 'temperature')


# test default and then changed default
def test_default_configuration_change_default_path():
    orig_path = cfg.core.lmr_path
    cfg1 = cfg.Config()
    new_path = 'new_path/is/here'
    update_dict = {'core': {'lmr_path': new_path}}
    cfg2 = cfg.Config(**update_dict)

    assert cfg1.core.lmr_path != cfg2.core.lmr_path
    assert new_path in cfg2.prior.datadir
    assert new_path in cfg2.psm.linear.datadir
    assert new_path in cfg2.proxies.PAGES2kv1.datadir_proxy
    assert orig_path not in cfg2.prior.datadir
    assert orig_path not in cfg2.psm.linear.datadir
    assert orig_path not in cfg2.proxies.PAGES2kv1.datadir_proxy

    cfg.core.lmr_path = orig_path


# Test class, subclass, and instances are recognized by is_config_class
def test_is_config_class_check():
    CfgClass = cfg.ConfigGroup

    class CfgSubclass(CfgClass):
        pass

    assert cfg.is_config_class(CfgClass)
    assert cfg.is_config_class(CfgSubclass)
    assert cfg.is_config_class(CfgClass())
    assert not cfg.is_config_class(1)


# Test that instances are not sharing attribute referencess
def test_class_instance_separation():
    # Only tests non-hashable attributes for reference exclusivity
    cfg_obj = cfg.Config()

    def test_references(cfg_class, cfg_object):

        for attr in dir(cfg_class):
            class_attr = getattr(cfg_class, attr)

            try:
                obj_attr = getattr(cfg_object, attr)

                # If True it's a ConfigGroup attribute
                if callable(class_attr) and cfg.is_config_class(class_attr):
                    test_references(class_attr, obj_attr)
                # Else see if it's a mutable attribute using hash() and test
                elif not callable(class_attr) and not attr.startswith('__'):
                    try:
                        hash(class_attr)
                    except TypeError:
                        assert class_attr is not obj_attr
            except AttributeError as e:
                pass

    test_references(cfg, cfg_obj)


# Use a yaml file to update the configuration class attributes
def test_config_update_with_yaml():

    with open('test_config.yml', 'r') as f:
        yaml_dict = yaml.load(f)

    ref_yaml_dict = deepcopy(yaml_dict)
    cfg.update_config_attrs_yaml(yaml_dict, cfg)

    # Recursive comparison function between instance and yaml dict
    def compare_obj_to_refdict(cfg_obj, ref_dict):
        for attr in dir(cfg_obj):
            if not attr.startswith('__') and attr in ref_dict:
                obj_attr = getattr(cfg_obj, attr)
                if callable(obj_attr):
                    compare_obj_to_refdict(obj_attr, ref_dict[attr])
                else:
                    assert obj_attr == ref_dict[attr]

    compare_obj_to_refdict(cfg, ref_yaml_dict)
    reload(cfg)


# Test that unused attributes in the yaml update are returned
def test_config_update_with_yaml_unused_attrs():

    yaml_dict = {'core': {'unused1': 1},
                 'psm': {'linear': {'unused2': 2}}}

    result = cfg.update_config_attrs_yaml(yaml_dict, cfg)

    assert result['core']['unused1'] == 1
    assert result['psm']['linear']['unused2'] == 2


# Test alteration of attributes using keyword arguments during object init
def test_config_update_with_kwarg():

    kwargs = {'wrapper': {'multi_seed': [1, 2, 3]},
              'psm': {'linear': {'datatag': 'BerkeleyEarth'}}}

    tmp = cfg.Config(**kwargs)

    # Was instance updated?
    assert tmp.wrapper.multi_seed == [1, 2, 3]
    assert tmp.psm.linear.datatag == 'BerkeleyEarth'
    # Was the class left unaltered?
    assert cfg.wrapper.multi_seed != [1, 2, 3]
    assert cfg.psm.linear.datatag != 'BerkeleyEarth'


# DatasetDescriptor Tests #
def test_datadescr_initialize():
    tmp = cfg._DatasetDescriptors().data
    assert isinstance(tmp, dict)


def test_datadescr_file_not_found():

    root = os.path.abspath(os.sep)
    wrong_filepath = os.path.join(root, 'incorrect_dir', 'datasets.yml')
    with pytest.raises(SystemExit):
        cfg._DatasetDescriptors(filename=wrong_filepath)


def test_datadescr_config_init():

    assert hasattr(cfg, '_DataInfo')
    res = cfg._DataInfo.get_info('GISTEMP')

    assert 'info' in res.keys()
    assert 'datadir' in res.keys()
    assert 'datafile' in res.keys()
    assert 'dataformat' in res.keys()


def test_datadescr_null_datadir():

    cfg._DataInfo.data['GISTEMP']['datadir'] = None
    cfg_update = {'psm': {'linear': {'datatag': 'GISTEMP'}}}
    cfg_obj = cfg.Config(**cfg_update)

    path = os.path.join(cfg_obj.core.lmr_path, 'data', 'analyses')
    assert cfg_obj.psm.linear.datadir == path


def test_datadescr_non_default_datadir():

    cfg._DataInfo.data['GISTEMP']['datadir'] = '/new/data/dir/path'
    cfg_update = {'psm': {'linear': {'datatag': 'GISTEMP'}}}
    cfg_obj = cfg.Config(**cfg_update)

    assert cfg_obj.psm.linear.datadir == '/new/data/dir/path'


def test_constant_avg_period(constant_def):

    annual = constant_def.get_info('avg_interval')['annual_std']

    assert 'nelem_in_yr' in annual
    assert 'elem_to_avg' in annual
    assert 'nyears' in annual

    np.testing.assert_array_equal(annual['elem_to_avg'],
                                  range(annual['nelem_in_yr']))


def test_constant_avg_period_negative_indices(constant_def):

    negs = constant_def.get_info('avg_interval')['negative_vals']

    np.testing.assert_array_equal(negs['elem_to_avg'],
                                  range(9, 15))


def test_constants_non_contiguous_avg_period():

    with pytest.raises(ValueError):
        path = os.path.join(cfg.SRC_DIR, 'tests', 'test_constants_fail.yml')
        constants = cfg._ConstantDefinitions(filename=path)
