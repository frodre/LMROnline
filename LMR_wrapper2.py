"""
Module: LMR_wrapper.py

Purpose: Wrapper around the callable version of LMR_driver
         prototype for Monte Carlo iterations

Options: None. 
         Experiment parameters defined through namelist, 
         passed through object called "state"

Originator: Greg Hakim   | Dept. of Atmospheric Sciences, Univ. of Washington
                         | April 2015
 
Revisions: 
          - Adapted to OOP coding of proxy- and prior-related classes [A. Perkins, U. of Washington]
          - Includes new parameter space search into iterations [A. Perkins, UW, Feb 2017]
          - Added flag to control whether or not the analysis_Ye.pckl is generated. [G. Hakim, UW, Aug 2017]
          - Added flag to control whether or not the full ensemble is output. [M. Erb, G. Hakim port, Aug 2017]
"""
import os
import numpy as np
import sys
import yaml
import itertools
import datetime
import LMR_driver_callable2 as LMR
import LMR_config

import LMR_utils2 as Util2
from LMR_utils2 import ensemble_stats, validate_config

print('\n' + str(datetime.datetime.now()) + '\n')

if not LMR_config.LEGACY_CONFIG:
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = os.path.join(LMR_config.SRC_DIR, 'config.yml')

    LMR_config.initialize_config_yaml(LMR_config, yaml_file)

# Define main experiment output directory
iter_range = LMR_config.wrapper.iter_range
expdir = os.path.join(LMR_config.core.datadir_output, LMR_config.core.nexp)
arc_dir = os.path.join(LMR_config.core.archive_dir, LMR_config.core.nexp)

# Check if it exists, if not, create it
if not os.path.isdir(expdir):
    os.system('mkdir {}'.format(expdir))

# Monte-Carlo approach: loop over iterations (range of iterations defined in
# namelist)
MCiters = range(iter_range[0], iter_range[1]+1)
param_iterables = [MCiters]

# get other parameters to sweep over in the reconstruction
param_search = LMR_config.wrapper.param_search
if param_search is not None:
    # sort them by parameter name and combine into a list of iterables
    sort_params = list(param_search.keys())
    sort_params.sort(key=lambda x: x.split('.')[-1])
    param_values = [param_search[key] for key in sort_params]
    param_iterables = param_values + [MCiters]

for iter_and_params in itertools.product(*param_iterables):

    iter_num = iter_and_params[-1]
    cfg_dict = Util2.param_cfg_update('core.curr_iter', iter_num)

    if LMR_config.wrapper.multi_seed is not None:
        curr_seed = LMR_config.wrapper.multi_seed[iter_num]
        cfg_dict = Util2.param_cfg_update('core.seed', curr_seed,
                                          cfg_dict=cfg_dict)
        print(('Setting current iteration seed: {}'.format(curr_seed)))

    itr_str = 'r{:d}'.format(iter_num)
    # If parameter space search is being performed then set the current
    # search space values and create a special sub-directory
    if param_search is not None:
        curr_param_values = iter_and_params[:-1]
        cfg_dict, psearch_dir = Util2.psearch_list_cfg_update(sort_params,
                                                              curr_param_values,
                                                              cfg_dict=cfg_dict)

        working_dir = os.path.join(expdir, psearch_dir, itr_str)
        mc_arc_dir = os.path.join(arc_dir, psearch_dir, itr_str)
    else:
        working_dir = os.path.join(expdir, itr_str)
        mc_arc_dir = os.path.join(arc_dir, itr_str)

    cfg_params = Util2.param_cfg_update('core.datadir_output', working_dir,
                                        cfg_dict=cfg_dict)

    cfg = LMR_config.Config(**cfg_params)

    # proceed = validate_config(cfg)
    # if not proceed:
    #     raise SystemExit()
    # else:
    #     print 'OK!'
    core = cfg.core

    # Check if it exists, if not create it
    if not os.path.isdir(core.datadir_output):
        os.makedirs(core.datadir_output)
    elif os.path.isdir(core.datadir_output) and core.clean_start:
        print (' **** clean start --- removing existing files in iteration'
               ' output directory')
        os.system('rm -rf {}'.format(core.datadir_output + '/*'))

    # Call the driver
    try:
        LMR.LMR_driver_callable(cfg)
    except LMR.FilterDivergenceError as e:
        print(e)

        # removing the work output directory
        cmd = 'rm -f -r ' + working_dir
        print(cmd)
        os.system(cmd)
        continue

    # start: DO NOT DELETE
    # move files from local disk to an archive location

    # scrub the monte carlo subdirectory if this is a clean start
    if os.path.isdir(mc_arc_dir):
        if core.clean_start:
            print (' **** clean start --- removing existing files in'
                   ' iteration output directory')
            os.system('rm -f -r {}'.format(mc_arc_dir + '/*'))
    else:
        os.makedirs(mc_arc_dir)

    # or just move select files and delete the rest

    cmd = 'mv -f ' + working_dir + '/*.npz' + ' ' + mc_arc_dir + '/'
    print(cmd)
    os.system(cmd)
    cmd = 'mv -f ' + working_dir + '/*.h5' + ' ' + mc_arc_dir + '/'
    print(cmd)
    os.system(cmd)
    cmd = 'mv -f ' + working_dir + '/*.pkl' + ' ' + mc_arc_dir + '/'
    print(cmd)
    os.system(cmd)
    cmd = 'mv -f ' + working_dir + '/*.zarr' + ' ' + mc_arc_dir + '/'
    print(cmd)
    os.system(cmd)

    # removing the work output directory once selected files have been moved
    cmd = 'rm -f -r ' + working_dir
    print(cmd)
    os.system(cmd)

    # copy the configuration file to archive directory
    if LMR_config.LEGACY_CONFIG:
        cmd = 'cp ./LMR_config.py ' + mc_arc_dir + '/'
    else:
        cmd = 'cp ' + yaml_file + ' ' + mc_arc_dir + '/'
    print(cmd)
    os.system(cmd)

    print('\n' + str(datetime.datetime.now()) + '\n')

    #   end: DO NOT DELETE
    
# ==============================================================================
