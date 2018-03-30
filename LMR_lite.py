"""
This is a 'lite' version of the LMR driver for Python2. Consider it a wrapper on the basic LMR functionality so that the user is shielded from details, and can work at a higher level. It assumes that you have done the following:

* set up the configuration file config_lite.py (a regular config.yml file)
* set up the PSMs and pre-built the Ye values that correspond to the chosen prior.

There are two examples:
(1) reconstruct a single year using two different solvers for the Kalman filter
(2) reconstruct a time period, and plot the result with instrumental analyses

The code can be easy extended and used in Jupyter notebooks. Note that there are other convenience functions in LMR_lite_utils.py not used here.

Originator:

Greg Hakim
University of Washington
26 February 2018

Modifications:
27 February 2018: added second example; bug fix in Ye call (GJH)

"""
import LMR_lite_utils as LMRlite
import LMR_utils
import numpy as np

#-----------------------------------------------------------------
# load components
#-----------------------------------------------------------------
print('loading configuration in your config_lite.yml file...')
cfg = LMRlite.load_config()
print('loading proxies...')
prox_manager = LMRlite.load_proxies(cfg)
print('loading prior...')
X, Xb_one = LMRlite.load_prior(cfg)
print('loading Ye...')
Ye_assim, Ye_assim_coords = LMR_utils.load_precalculated_ye_vals_psm_per_proxy(cfg, prox_manager,'assim',X.prior_sample_indices)

#-----------------------------------------------------------------
# example reconstruction for one year
#-----------------------------------------------------------------
target_year=1950
print('performing a reconstruction for year:' + str(target_year))
vY,vR,vP,vYe,vT,vYe_coords = LMRlite.get_valid_proxies(cfg,prox_manager,target_year,Ye_assim,Ye_assim_coords)
xam,Xap,_ = LMRlite.Kalman_optimal(vY,vR,vYe,Xb_one,verbose=False)
xam2,Xap2 = LMRlite.Kalman_ESRF(cfg,vY,vR,vYe,Xb_one,verbose=False)
print('ens mean max difference from different solvers...', str(np.max(np.abs((xam2-xam)/xam))))

#-----------------------------------------------------------------
# example reconstruction over 1800-2000, computing GMT on the way
#-----------------------------------------------------------------

# set years to reconstruct
years = list(range(1800,2000))

# make a grid object based on grid info in prior object
grid = LMRlite.make_grid(X)

gmt_save = np.zeros(len(years))
gmt_ens_save = np.zeros([len(years),grid.Nens])
yk = -1
for target_year in years:
    yk = yk + 1
    vY,vR,vP,vYe,vT,vYe_coords = LMRlite.get_valid_proxies(cfg,prox_manager,target_year,Ye_assim,Ye_assim_coords,verbose=False)
    xam,Xap,_ = LMRlite.Kalman_optimal(vY,vR,vYe,Xb_one)
    xam_lalo = np.reshape(xam,[grid.nlat,grid.nlon])
    # GMT for the ensemble mean
    gmt, nhmt, shmt = LMR_utils.global_hemispheric_means(xam_lalo,grid.lat[:, 0])
    print(str(target_year)+': '+str(gmt)+', '+str(nhmt)+', '+str(shmt))
    gmt_save[yk] = gmt
    # GMT for all ensemble members
    for k in range(grid.Nens):
        xam_lalo = np.reshape(Xap[:,k],[grid.nlat,grid.nlon])
        gmt_ens_save[yk,k],_,_ = LMR_utils.global_hemispheric_means(xam_lalo, grid.lat[:, 0])

# numpy array for the years list
lmr_years = np.array(years)

# load GMT fields into dictionaries, add LMR ensemble mean, with specified reference and verification time periods
analysis_data,analysis_time,_,_ = LMRlite.load_analyses(cfg,full_field=False,lmr_gm=gmt_save,lmr_time=lmr_years,satime=1900,eatime=2000,svtime=1880,evtime=2000)

# make a plot and save a figure
LMRlite.make_gmt_figure(analysis_data,analysis_time,fsave='lite_testing')
