

#
# verify statistics related to the global-mean 2m air temperature
#
# started from LMR_plots.py r-86
#
# CRU data too coarse to include in the analysis
#

import matplotlib
# need to do this when running remotely
matplotlib.use('Agg')
# use this line when running a notebook
#get_ipython().magic(u'matplotlib inline')
#%matplotlib

# generic imports
import numpy as np
import glob, os
from datetime import datetime, timedelta
from netCDF4 import Dataset
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from matplotlib import ticker
from spharm import Spharmt, getspecindx, regrid
# LMR specific imports
from LMR_utils import global_mean, assimilated_proxies
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from LMR_plot_support import *
from LMR_exp_NAMELIST import *
from LMR_plot_support import *

# change default value of latlon kwarg to True.
bm.latlon_default = True




##################################
# START:  set user parameters here
##################################

# option to suppress figures
#iplot = False
iplot = True

# centered time mean (nya must be odd! 3 = 3 yr mean; 5 = 5 year mean; etc 0 = none)
nya = 0

# option to print figures
fsave = True
#fsave = False

# set paths, the filename for plots, and global plotting preferences

# file specification
#
# current datasets
#
nexp = 'testing_1000_75pct_ens_size_Nens_10'
#nexp = 'testdev_check_1000_75pct'
#nexp = 'ReconDevTest_1000_testing_coral'
#nexp = 'ReconDevTest_1000_testing_icecore'
#nexp = 'testdev_1000_100pct_icecoreonly'
#nexp = 'testdev_1000_100pct_mxdonly'
#nexp = 'testdev_1000_100pct_sedimentonly'

# OLD:

#nexp = 'testdev_500_allproxies'
#exp = 'testdev_500_trunctest'
#
#nexp = 'testdev_1000_75pct'
#nexp = 'testdev_1000_75pct_noTRW'
#nexp = 'testdev_1000_75pct_treesonly'
#nexp = 'testdev_1000_75pct_icecoreonly'
#nexp = 'testdev_1000_75pct_coralonly'
#nexp = 'testdev_1000_100pct_coralonly'
#nexp = 'testdev_1000_100pct_icecoreonly'
#nexp = 'testdev_1000_100pct_mxdonly'
#nexp = 'testdev_1000_100pct_sedimentonly'
# new
#nexp = 'testdev_1000_75pct_BE'
#nexp = 'testdev_1000_75pct_BE_noTRW'

# override datadir
datadir_output = '/home/disk/kalman3/hakim/LMR/'
#datadir_output = './data/'

# number of contours for plots
nlevs = 30

# plot alpha transparency
alpha = 0.5

# time range for verification (in years CE)
#trange = [1960,1962]
trange = [1880,2000] #works for nya = 0
#trange = [1885,1995] #works for nya = 5
#trange = [1890,1990] #works for nya = 10

# set the default size of the figure in inches. ['figure.figsize'] = width, height;  
# aspect ratio appears preserved on smallest of the two
plt.rcParams['figure.figsize'] = 10, 10  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
#plt.rc('text', usetex=True)
plt.rc('text', usetex=False)

##################################
# END:  set user parameters here
##################################



workdir = datadir_output + '/' + nexp
print 'working directory = ' + workdir

print '\n getting file system information...\n'

# get number of mc realizations from directory count
tmp = os.listdir(workdir)
print tmp
# since some files may not be iteration subdirectories; count those that are
niters = 0
mcdir = []
for subdir in tmp:
    if os.path.isdir(workdir+'/'+subdir):
        niters = niters + 1
        mcdir.append(subdir)
        
print 'mcdir:' + str(mcdir)
print 'niters = ' + str(niters)

# get time period from the GMT file...
gmtpfile =  workdir + '/r0/gmt.npz'
npzfile = np.load(gmtpfile)
npzfile.files
LMR_time = npzfile['recon_times']

# get grid information from the prior file...
prior_filn = workdir + '/r0/Xb_one.npz'
npzfile = np.load(prior_filn)
npzfile.files
lat = npzfile['lat']
lon = npzfile['lon']
nlat = npzfile['nlat']
nlon = npzfile['nlon']
lat2 = np.reshape(lat,(nlat,nlon))
lon2 = np.reshape(lon,(nlat,nlon))

#print lat[7],lat[34]

# read ensemble mean data
print '\n reading LMR ensemble-mean data...\n'

first = True
k = -1
for dir in mcdir:
    k = k + 1
    ensfiln = workdir + '/' + dir + '/ensemble_mean.npz'
    print ensfiln
    npzfile = np.load(ensfiln)
    #npzfile.files
    print  npzfile.files
    years = npzfile['years']
    nyrs =  len(years)
    tmp = npzfile['xam']
    print 'shape of tmp: ' + str(np.shape(tmp))
    if first:
        first = False
        xam = np.zeros([nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
        xam_all = np.zeros([niters,nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
    xam = xam + tmp
    xam_all[k,:,:,:] = tmp
    
# this is the sample mean computed with low-memory accumulation
xam = xam/len(mcdir)
# this is the sample mean computed with numpy on all data
xam_check = xam_all.mean(0)
# check..
max_err = np.max(np.max(np.max(xam_check - xam)))
print 'max error = ' + str(max_err)

# sample variance
xam_var = xam_all.var(0)
print np.shape(xam_var)

print '\n shape of the ensemble array: ' + str(np.shape(xam_all)) +'\n'
print '\n shape of the ensemble-mean array: ' + str(np.shape(xam)) +'\n'




#################################################################
# BEGIN: load verification data (GISTEMP, HadCRU, BE, and 20CR) #
#################################################################
print '\nloading verification data...\n'

datadir_calib = '../data/'

# load GISTEMP
datafile_calib   = 'gistemp1200_ERSST.nc'
calib_vars = ['Tsfc']
[GIS_time,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars)
nlat_GIS = len(GIS_lat)
nlon_GIS = len(GIS_lon)
lon2_GIS, lat2_GIS = np.meshgrid(GIS_lon, GIS_lat)

# load HadCRU
datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
calib_vars = ['Tsfc']
[CRU_time,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars)

# load BerkeleyEarth
datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
calib_vars = ['Tsfc']
[BE_time,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars)
nlat_BE = len(BE_lat)
nlon_BE = len(BE_lon)
lon2_BE, lat2_BE = np.meshgrid(BE_lon, BE_lat)

# load 20th century reanalysis (this is copied from R. Tardif's load_gridded_data.py routine)

infile = '/home/disk/ice4/hakim/data/20th_century_reanalysis_v2/T_0.995/air.sig995.mon.mean.nc'
#infile = './data/500_allproxies_0/air.sig995.mon.mean.nc'

data = Dataset(infile,'r')
lat_20CR   = data.variables['lat'][:]
lon_20CR   = data.variables['lon'][:]
nlat_20CR = len(lat_20CR)
nlon_20CR = len(lon_20CR)
lon2_TCR, lat2_TCR = np.meshgrid(lon_20CR, lat_20CR)
 
dateref = datetime(1800,1,1,0)
time_yrs = []
# absolute time from the reference
for i in xrange(0,len(data.variables['time'][:])):
    time_yrs.append(dateref + timedelta(hours=int(data.variables['time'][i])))

years_all = []
for i in xrange(0,len(time_yrs)):
    isotime = time_yrs[i].isoformat()
    years_all.append(int(isotime.split("-")[0]))

TCR_time = np.array(list(set(years_all))) # 'set' is used to get unique values in list
TCR_time.sort # sort the list

time_yrs  = np.empty(len(TCR_time), dtype=int)
TCR = np.empty([len(TCR_time), len(lat_20CR), len(lon_20CR)], dtype=float)
tcr_gm = np.zeros([len(TCR_time)])

# Loop over years in dataset
for i in xrange(0,len(TCR_time)):        
    # find indices in time array where "years[i]" appear
    ind = [j for j, k in enumerate(years_all) if k == TCR_time[i]]
    time_yrs[i] = TCR_time[i]
    # ---------------------------------------
    # Calculate annual mean from monthly data
    # Note: data has dims [time,lat,lon]
    # ---------------------------------------
    TCR[i,:,:] = np.nanmean(data.variables['air'][ind],axis=0)
    # compute the global mean temperature
    tcr_gm[i] = global_mean(TCR[i,:,:],lat_20CR,lon_20CR)
    
# Remove the temporal mean 
TCR = TCR - np.mean(TCR,axis=0)
print 'TCR shape = ' + str(np.shape(TCR))

# compute and remove the 20th century mean
stime = 1900
etime = 1999
smatch, ematch = find_date_indices(TCR_time,stime,etime)
tcr_gm = tcr_gm - np.mean(tcr_gm[smatch:ematch])

###############################################################
# END: load verification data (GISTEMP, HadCRU, BE, and 20CR) #
###############################################################



print '\n regridding data to a common T42 grid...\n'

iplot= False
#iplot= True

# create instance of the spherical harmonics object for each grid
specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')
specob_tcr = Spharmt(nlon_20CR,nlat_20CR,gridtype='regular',legfunc='computed')
specob_gis = Spharmt(nlon_GIS,nlat_GIS,gridtype='regular',legfunc='computed')
specob_be = Spharmt(nlon_BE,nlat_BE,gridtype='regular',legfunc='computed')

# truncate to a lower resolution grid (common:21, 42, 62, 63, 85, 106, 255, 382, 799)
ntrunc_new = 42 # T21
ifix = np.remainder(ntrunc_new,2.0).astype(int)
nlat_new = ntrunc_new + ifix
nlon_new = int(nlat_new*1.5)
# lat, lon grid in the truncated space
dlat = 90./((nlat_new-1)/2.)
dlon = 360./nlon_new
veclat = np.arange(-90.,90.+dlat,dlat)
veclon = np.arange(0.,360.,dlon)
blank = np.zeros([nlat_new,nlon_new])
lat2_new = (veclat + blank.T).T  
lon2_new = (veclon + blank)  

# create instance of the spherical harmonics object for the new grid
specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')


lmr_trunc = np.zeros([nyrs,nlat_new,nlon_new])
print 'lmr_trunc shape: ' + str(np.shape(lmr_trunc))

# loop over years of interest and transform...specify trange at top of file

iw = 0
if nya > 0:
    iw = (nya-1)/2

cyears = range(trange[0],trange[1])
lt_csave = np.zeros([len(cyears)])
lg_csave = np.zeros([len(cyears)])
tg_csave = np.zeros([len(cyears)])
lb_csave = np.zeros([len(cyears)])
bg_csave = np.zeros([len(cyears)])
lmr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
tcr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
gis_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
be_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
k = -1
for yr in cyears:
    k = k + 1
    LMR_smatch, LMR_ematch = find_date_indices(LMR_time,yr-iw,yr+iw+1)
    TCR_smatch, TCR_ematch = find_date_indices(TCR_time,yr-iw,yr+iw+1)
    GIS_smatch, GIS_ematch = find_date_indices(GIS_time,yr-iw,yr+iw+1)
    BE_smatch, BE_ematch = find_date_indices(BE_time,yr-iw,yr+iw+1)
    print '------------------------------------------------------------------------'
    print 'working on year...' + str(yr)
    print 'working on year...' + str(yr) + ' LMR index = ' + str(LMR_smatch) + ' = LMR year ' + str(LMR_time[LMR_smatch])
    #print 'working on year...' + str(yr) + ' TCR index = ' + str(TCR_smatch) + ' = TCR year ' + str(TCR_time[TCR_smatch])
    #print 'working on year...' + str(yr) + ' GIS index = ' + str(GIS_smatch) + ' = GIS year ' + str(GIS_time[GIS_smatch])
    #print 'working on year...' + str(yr) + ' BE index = ' + str(BE_smatch) + ' = BE year ' + str(BE_time[BE_smatch])

    # LMR
    pdata_lmr = np.mean(xam[LMR_smatch:LMR_ematch,:,:],0)    
    #pdata_lmr = np.squeeze(xam[LMR_smatch,:,:])
    lmr_trunc = regrid(specob_lmr, specob_new, pdata_lmr, ntrunc=nlat_new-1, smooth=None)
    #print 'shape of old LMR data array:' + str(np.shape(pdata_lmr))
    #print 'shape of new LMR data array:' + str(np.shape(lmr_trunc))
    #LMR_plotter(lmr_trunc,lat2_new,lon2_new,'bwr',nlevs)
    #LMR_plotter(lmr_trunc,lat2_new,lon2_new,'bwr',nlevs)
    #LMR_plotter(pdata_lmr,lat2,lon2,'bwr',nlevs)
    #plt.show()

    # TCR
    pdata_tcr = np.mean(TCR[TCR_smatch:TCR_ematch,:,:],0)    
    #pdata_tcr = np.squeeze(TCR[TCR_smatch,:,:])
    tcr_trunc = regrid(specob_tcr, specob_new, pdata_tcr, ntrunc=nlat_new-1, smooth=None)
    # TCR latitudes upside down
    tcr_trunc = np.flipud(tcr_trunc)
    #print 'shape of old TCR data array:' + str(np.shape(pdata_tcr))
    #print 'shape of new TCR data array:' + str(np.shape(tcr_trunc))

    # GIS
    pdata_gis = np.mean(GIS_anomaly[GIS_smatch:GIS_ematch,:,:],0)    
    #pdata_gis = np.squeeze(np.nan_to_num(GIS_anomaly[GIS_smatch,:,:]))
    gis_trunc = regrid(specob_gis, specob_new, np.nan_to_num(pdata_gis), ntrunc=nlat_new-1, smooth=None)
    # GIS logitudes are off by 180 degrees
    gis_trunc = np.roll(gis_trunc,shift=nlon_new/2,axis=1)
    #print 'shape of old GIS data array:' + str(np.shape(pdata_gis))
    #print 'shape of new GIS data array:' + str(np.shape(gis_trunc))

    # BE
    pdata_be = np.mean(BE_anomaly[BE_smatch:BE_ematch,:,:],0)    
    #pdata_be = np.squeeze(np.nan_to_num(BE_anomaly[BE_smatch,:,:]))
    be_trunc = regrid(specob_be, specob_new, np.nan_to_num(pdata_be), ntrunc=nlat_new-1, smooth=None)
    # BE logitudes are off by 180 degrees
    be_trunc = np.roll(be_trunc,shift=nlon_new/2,axis=1)
    #print 'shape of old BE data array:' + str(np.shape(pdata_be))
    #print 'shape of new BE data array:' + str(np.shape(be_trunc))

    # save the full grids
    lmr_allyears[k,:,:] = lmr_trunc
    tcr_allyears[k,:,:] = tcr_trunc
    gis_allyears[k,:,:] = gis_trunc
    be_allyears[k,:,:] = be_trunc

    if iplot:
        ncints = 30
        cmap = 'bwr'
        nticks = 6 # number of ticks on the colorbar
        #set contours based on Berkeley Earth
        maxabs = np.nanmax(np.abs(be_trunc))
        # round the contour interval, and then set limits to fit
        dc = np.round(maxabs*2/ncints,2)
        cl = dc*ncints/2.
        cints = np.linspace(-cl,cl,ncints,endpoint=True)
        
        # compare LMR and TCR and GIS and BE
        fig = plt.figure()
        
        ax = fig.add_subplot(2,2,1)
        m1 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(lmr_trunc))
        cs = m1.contourf(lon2_new,lat2_new,lmr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m1.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('LMR T'+str(ntrunc_new) + ' ' + str(yr))
        
        ax = fig.add_subplot(2,2,2)
        m2 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(tcr_trunc))
        cs = m2.contourf(lon2_new,lat2_new,tcr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m2.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('TCR T'+str(ntrunc_new)+ ' ' + str(yr))
        
        ax = fig.add_subplot(2,2,3)
        m3 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(gis_trunc))
        cs = m3.contourf(lon2_new,lat2_new,gis_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m3.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('GIS T'+str(ntrunc_new)+ ' ' + str(yr))
        
        ax = fig.add_subplot(2,2,4)
        m4 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(be_trunc))
        cs = m2.contourf(lon2_new,lat2_new,be_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m4.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('BE T'+str(ntrunc_new)+ ' ' + str(yr))
        plt.clim(-maxabs,maxabs)
        
        # get these numbers by adjusting the figure interactively!!!
        plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.95, wspace=0.1, hspace=0.0)
        # plt.tight_layout(pad=0.3)
        fig.suptitle('2m air temperature for ' +str(nya) +' year centered average')

    
    # anomaly correlation
    lmrvec = np.reshape(lmr_trunc,(1,nlat_new*nlon_new))
    tcrvec = np.reshape(tcr_trunc,(1,nlat_new*nlon_new))
    gisvec = np.reshape(gis_trunc,(1,nlat_new*nlon_new))
    bevec = np.reshape(be_trunc,(1,nlat_new*nlon_new))
    lmr_tcr_corr = np.corrcoef(lmrvec,tcrvec)

    #print 'lmr-tcr correlation: '+str(lmr_tcr_corr[0,1])
    lmr_gis_corr = np.corrcoef(lmrvec,gisvec)
    #print 'lmr-gis correlation: '+ str(lmr_gis_corr[0,1])
    lmr_be_corr = np.corrcoef(lmrvec,bevec)
    #print 'lmr-be correlation: '+ str(lmr_be_corr[0,1])
    tcr_gis_corr = np.corrcoef(tcrvec,gisvec)
    #print 'gis-tcr correlation: '+ str(tcr_gis_corr[0,1])
    be_gis_corr = np.corrcoef(bevec,gisvec)
    #print 'gis-be correlation: '+ str(be_gis_corr[0,1])

    # save the correlation values
    lt_csave[k] = lmr_tcr_corr[0,1]
    lg_csave[k] = lmr_gis_corr[0,1]
    lb_csave[k] = lmr_be_corr[0,1]
    tg_csave[k] = tcr_gis_corr[0,1]
    bg_csave[k] = be_gis_corr[0,1]

    

# plots for anomaly correlation statistics

# number of bins in the histograms
nbins = 10
#nbins = 5

# LMR compared to GIS and TCR
fig = plt.figure()
ax = fig.add_subplot(3,2,1)
ax.plot(cyears,lt_csave)
ax.set_title('LMR-TCR')
ax = fig.add_subplot(3,2,2)
ax.hist(lt_csave,bins=nbins)
ax.set_title('LMR-TCR')
ax = fig.add_subplot(3,2,3)
ax.plot(cyears,lg_csave)
ax.set_title('LMR-GIS')
ax = fig.add_subplot(3,2,4)
ax.hist(lg_csave,bins=nbins)
ax.set_title('LMR-GIS')
ax = fig.add_subplot(3,2,5)
ax.plot(cyears,lb_csave)
ax.set_title('LMR-BE')
ax = fig.add_subplot(3,2,6)
ax.hist(lb_csave,bins=nbins)
ax.set_title('LMR-BE')
fig.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.9, wspace=0.5, hspace=0.5)
fig.suptitle('2m air temperature anomaly correlation') 
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1]))

# GIS compared to TCR
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.plot(cyears,tg_csave)
ax.set_title('GIS-TCR')
ax = fig.add_subplot(2,2,2)
ax.hist(tg_csave,bins=nbins)
ax.set_title('GIS-TCR')
ax = fig.add_subplot(2,2,3)
ax.plot(cyears,bg_csave)
ax.set_title('GIS-BE')
ax = fig.add_subplot(2,2,4)
ax.hist(bg_csave,bins=nbins)
ax.set_title('GIS-BE')
#fig.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.9, wspace=0.5, hspace=0.5)
fig.suptitle('2m air temperature anomaly correlation') 
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_reference_'+str(trange[0])+'-'+str(trange[1]))


#
# BEGIN r and CE calculations
#

# correlation and CE at each (lat,lon) point

lt_err = lmr_allyears - tcr_allyears
lg_err = lmr_allyears - gis_allyears
lb_err = lmr_allyears - be_allyears
tg_err = tcr_allyears - gis_allyears

r_lt = np.zeros([nlat_new,nlon_new])
ce_lt = np.zeros([nlat_new,nlon_new])
r_lg = np.zeros([nlat_new,nlon_new])
ce_lg = np.zeros([nlat_new,nlon_new])
r_lb = np.zeros([nlat_new,nlon_new])
ce_lb = np.zeros([nlat_new,nlon_new])
r_tg = np.zeros([nlat_new,nlon_new])
ce_tg = np.zeros([nlat_new,nlon_new])
for la in range(nlat_new):
    for lo in range(nlon_new):
        # LMR-TCR
        tstmp = np.corrcoef(lmr_allyears[:,la,lo],tcr_allyears[:,la,lo])
        evar = np.var(lt_err[:,la,lo],ddof=1)
        tvar = np.var(tcr_allyears[:,la,lo],ddof=1)
        r_lt[la,lo] = tstmp[0,1]
        ce_lt[la,lo] = 1. - (evar/tvar)
        # LMR-GIS
        tstmp = np.corrcoef(lmr_allyears[:,la,lo],gis_allyears[:,la,lo])
        evar = np.var(lg_err[:,la,lo],ddof=1)
        tvar = np.var(gis_allyears[:,la,lo],ddof=1)
        r_lg[la,lo] = tstmp[0,1]
        ce_lg[la,lo] = 1. - (evar/tvar)
        # LMR-BE
        tstmp = np.corrcoef(lmr_allyears[:,la,lo],be_allyears[:,la,lo])
        evar = np.var(lb_err[:,la,lo],ddof=1)
        tvar = np.var(be_allyears[:,la,lo],ddof=1)
        r_lb[la,lo] = tstmp[0,1]
        ce_lb[la,lo] = 1. - (evar/tvar)
        # TCR-GIS
        tstmp = np.corrcoef(tcr_allyears[:,la,lo],gis_allyears[:,la,lo])
        evar = np.var(tg_err[:,la,lo],ddof=1)
        tvar = np.var(gis_allyears[:,la,lo],ddof=1)
        r_tg[la,lo] = tstmp[0,1]
        ce_tg[la,lo] = 1. - (evar/tvar)
   

lt_rmean = str(float('%.2g' % np.median(np.median(r_lt)) ))
print 'lmr-tcr all-grid median r: ' + str(lt_rmean)
lt_rmean60 = str(float('%.2g' % np.median(np.median(r_lt[7:34,:])) ))
print 'lmr-tcr 60S-60N median r: ' + str(lt_rmean60)
lt_cemean = str(float('%.2g' % np.median(np.median(ce_lt)) ))
print 'lmr-tcr all-grid median ce: ' + str(lt_cemean)
lt_cemean60 = str(float('%.2g' % np.median(np.median(ce_lt[7:34,:])) ))
print 'lmr-tcr 60S-60N median ce: ' + str(lt_cemean60)
lg_rmean = str(float('%.2g' % np.median(np.median(r_lg)) ))
print 'lmr-gis all-grid median r: ' + str(lg_rmean)
lg_rmean60 = str(float('%.2g' % np.median(np.median(r_lg[7:34,:])) ))
print 'lmr-gis 60S-60N median r: ' + str(lg_rmean60)
lg_cemean = str(float('%.2g' % np.median(np.median(ce_lg)) ))
print 'lmr-gis all-grid median ce: ' + str(lg_cemean)
lg_cemean60 = str(float('%.2g' % np.median(np.median(ce_lg[7:34,:])) ))
print 'lmr-gis 60S-60N median ce: ' + str(lg_cemean60)
lb_rmean = str(float('%.2g' % np.median(np.median(r_lb)) ))
print 'lmr-be all-grid median r: ' + str(lb_rmean)
lb_rmean60 = str(float('%.2g' % np.median(np.median(r_lb[7:34,:])) ))
print 'lmr-be 60S-60N median r: ' + str(lb_rmean60)
lb_cemean = str(float('%.2g' % np.median(np.median(ce_lb)) ))
print 'lmr-be all-grid median ce: ' + str(lb_cemean)
lb_cemean60 = str(float('%.2g' % np.median(np.median(ce_lb[7:34,:])) ))
print 'lmr-be 60S-60N median ce: ' + str(lb_cemean60)
tg_rmean = str(float('%.2g' % np.median(np.median(r_tg)) ))
print 'tcr-gis all-grid median r: ' + str(tg_rmean)
tg_rmean60 = str(float('%.2g' % np.median(np.median(r_tg[7:34,:])) ))
print 'tcr-gis 60S-60N median r: ' + str(tg_rmean60)
tg_cemean = str(float('%.2g' % np.median(np.median(ce_tg)) ))
print 'tcr-gis all-grid median ce: ' + str(tg_cemean)
tg_cemean60 = str(float('%.2g' % np.median(np.median(ce_tg[7:34,:])) ))
print 'tcr-gis 60S-60N median ce: ' + str(tg_cemean60)

#
# END r and CE
#



# r and ce plots
iplot = True

nlevs = 11

if iplot:
    fig = plt.figure()
    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(r_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-TCR T r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_rmean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-TCR T CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_cemean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(r_lg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-GIS T r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lg_rmean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,4)    
    LMR_plotter(ce_lg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-GIS T CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lg_cemean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,5)    
    LMR_plotter(r_lg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-BE T r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lb_rmean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,6)    
    LMR_plotter(ce_lg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-BE T CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lb_cemean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,7)    
    LMR_plotter(r_tg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('TCR-GIS T r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(tg_rmean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,8)    
    LMR_plotter(ce_tg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('TCR-GIS T CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(tg_cemean60))
    plt.clim(-1,1)
  
    fig.tight_layout()
    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_verify_grid_r_ce_'+str(trange[0])+'-'+str(trange[1]))   
    

#ensemble calibration

print np.shape(lt_err)
print np.shape(xam_var)
LMR_smatch, LMR_ematch = find_date_indices(LMR_time,trange[0],trange[1])
print LMR_smatch, LMR_ematch
svar = xam_var[LMR_smatch:LMR_ematch,:,:]
print np.shape(svar)

calib = lt_err.var(0)/svar.mean(0)
print np.shape(calib)
print calib[0:-1,:].mean()



fig = plt.figure()
cb = LMR_plotter(calib,lat2_new,lon2_new,'Oranges',10,0,10)
cb.set_ticks(range(11))
# overlay stations!
plt.title('ratio of ensemble-mean error variance to mean ensemble variance')
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_ensemble_calibration_'+str(trange[0])+'-'+str(trange[1]))   

if iplot:
    plt.show()

# in loop over lat,lon, add a call to the rank histogram function; need to move up the function def
