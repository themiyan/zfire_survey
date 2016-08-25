from __future__ import unicode_literals
import numpy as np
from astropy.io import ascii, fits 


import matplotlib as mp
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm

import pandas as pd
import pylab as py
import scipy as sp
from scipy import optimize
import string as s
from statsmodels import robust
import statsmodels as stat

import os
import glob

#latex for matplotlib
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
from   matplotlib import rc
mp.rcParams['text.usetex']=True
mp.rcParams['text.latex.unicode']=True
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#turn off some pandas warnings
pd.options.mode.chained_assignment = None  

def rest_colours_df(U_spec, V_spec, J_spec, string):
        
        U = pd.Series(data=U_spec['L153'], index=U_spec['id'])
        V = pd.Series(data=V_spec['L155'], index=V_spec['id'])
        J = pd.Series(data=J_spec['L161'], index=J_spec['id'])

        rest_colours           = pd.DataFrame(U,columns=['U'+string] )
        rest_colours['V'+string] = V
        rest_colours['J'+string] = V
        
        return rest_colours
        
        



def prepare_cosmos():
   
    COSMOS_mastertable, ZF_cat, ZF_EAZY, ZF_FAST, U_spec, V_spec, J_spec,\
    U_photo, V_photo, J_photo, UV_lee, VJ_lee, UV_IR_SFRs, MOSDEF_ZFOURGE,\
    U_ZM,V_ZM, J_ZM, VUDS_ZFOURGE, VUDS_extra, U_ZV, V_ZV, J_ZV = open_cosmos_files()
    

    ZF      = pd.DataFrame(np.asarray(ZF_cat), index=ZF_cat['id'])
    ZF_EAZY = pd.DataFrame(np.asarray(ZF_EAZY), index=ZF_EAZY['id'])
    ZF_FAST = pd.DataFrame(np.asarray(ZF_FAST), index=ZF_FAST['id'])


    rest_colours_spec = rest_colours_df(U_spec, V_spec, J_spec, '_rest') 
    ZFOURGE = pd.merge(ZF, rest_colours_spec,how='left', left_index=True, right_index=True  )
    ZFOURGE.set_index(ZFOURGE.id, inplace=True)

    
    rest_colours_photo = rest_colours_df(U_photo, V_photo, J_photo, '_rest') 
    
    ZFOURGE['U_rest_photo'] = rest_colours_photo['U_rest']
    ZFOURGE['V_rest_photo'] = rest_colours_photo['V_rest']
    ZFOURGE['J_rest_photo'] = rest_colours_photo['J_rest']
    
    UV_lee = pd.Series(data=-2.5*np.log10(UV_lee['L153']/UV_lee['L155']), index=UV_lee['id'])
    VJ_lee = pd.Series(data=-2.5*np.log10(VJ_lee['L155']/VJ_lee['L161']), index=VJ_lee['id'])

    rest_colours_photo                = pd.DataFrame(UV_lee,columns=['UV_rest_Lee'] )
    rest_colours_photo['VJ_rest_Lee'] = VJ_lee

    ZFOURGE['UV_rest_Lee'] = rest_colours_photo['UV_rest_Lee']
    ZFOURGE['VJ_rest_Lee'] = rest_colours_photo['VJ_rest_Lee']


    ZFOURGE['U_rest'] = ZFOURGE['U_rest'].fillna(ZFOURGE['U_rest_photo'])
    ZFOURGE['V_rest'] = ZFOURGE['V_rest'].fillna(ZFOURGE['V_rest_photo'])
    ZFOURGE['J_rest'] = ZFOURGE['J_rest'].fillna(ZFOURGE['J_rest_photo'])

    ZFOURGE['UV'] = -2.5*np.log10(ZFOURGE['U_rest']/ZFOURGE['V_rest'])
    ZFOURGE['VJ'] = -2.5*np.log10(ZFOURGE['V_rest']/ZFOURGE['J_rest'])

    ZFOURGE['UV_photo'] = -2.5*np.log10(ZFOURGE['U_rest_photo']/ZFOURGE['V_rest_photo'])
    ZFOURGE['VJ_photo'] = -2.5*np.log10(ZFOURGE['V_rest_photo']/ZFOURGE['J_rest_photo'])


    UV_IR_SFRs = pd.DataFrame(np.asarray(UV_IR_SFRs), index=UV_IR_SFRs['id'])

    UV_IR_SFRs['UV_IR_SFRs'] = UV_IR_SFRs['sfr_luv'] + UV_IR_SFRs['sfr_luv']
    ZFOURGE['firflag']    = UV_IR_SFRs['firflag']
    ZFOURGE['UV_IR_SFRs'] = UV_IR_SFRs['UV_IR_SFRs']


    ZFOURGE = pd.merge(ZFOURGE, ZF_EAZY,how='inner',left_on='id', right_on='id', left_index=True )
    ZFOURGE = pd.merge(ZFOURGE, ZF_FAST,how='inner',left_on='id', right_on='id', left_index=True )

    ZFOURGE.index = ZFOURGE.index.astype('str')

    COSMOS = pd.merge(COSMOS_mastertable, ZFOURGE, how='left', left_index=True,right_index=True,
                         suffixes=('_ZFIRE', '_ZFOURGE'))
    
#   print COSMOS
#   COSMOS = COSMOS.drop('id', 1)
#   COSMOS = COSMOS.set_index('Nameobj')
    COSMOS['Ks_mag']     = 25-2.5*np.log10(COSMOS['Kstot'])
    COSMOS['Ks_mag_err'] = 2.5*0.434*COSMOS['eKstot']/COSMOS['Kstot']
    COSMOS['Nameobj']    = COSMOS.index


#     spec_z = ZFIRE[(ZFIRE['conf']>1)]

   
#     spec_z = pd.merge(ZFOURGE, spec_z, how='left', left_index=True, right_on='Nameobj', suffixes=('_ZFOURGE', '_ZFIRE'))
#     spec_z = spec_z.set_index(spec_z['id'])
#     spec_z['redshifts'] = spec_z['zspec']
#     spec_z['redshifts'] = spec_z['redshifts'].fillna(spec_z['z_peak'])

#     spec_z['masses'] = spec_z['lmass_ZFIRE']
#     spec_z['masses'] = spec_z['masses'].fillna(spec_z['lmass_ZFOURGE'])

#     spec_z['Av'] = spec_z['Av_ZFIRE']
#     spec_z['Av'] = spec_z['Av'].fillna(spec_z['Av_ZFOURGE'])
    

    MOSDEF_ZFOURGE = MOSDEF_ZFOURGE[MOSDEF_ZFOURGE['Z_MOSFIRE']>0] #remove non-detections from MOSDEF
    MOSDEF_ZFOURGE = pd.DataFrame(np.asarray(MOSDEF_ZFOURGE), index=MOSDEF_ZFOURGE['id'])
    MOSDEF_ZFOURGE = pd.merge(MOSDEF_ZFOURGE, ZF_FAST, how='inner', left_index=True, right_index=True)
    MOSDEF_ZFOURGE.index = MOSDEF_ZFOURGE.index.astype('str')


    rest_colours_spec = rest_colours_df(U_ZM, V_ZM, J_ZM, '_rest') 
    rest_colours_spec.index = rest_colours_spec.index.astype('str')

    MOSDEF_ZFOURGE = pd.merge(MOSDEF_ZFOURGE, rest_colours_spec,how='left', left_index=True, right_index=True  )

    MOSDEF_ZFOURGE['UV'] = -2.5*np.log10(MOSDEF_ZFOURGE['U_rest']/MOSDEF_ZFOURGE['V_rest'])
    MOSDEF_ZFOURGE['VJ'] = -2.5*np.log10(MOSDEF_ZFOURGE['V_rest']/MOSDEF_ZFOURGE['J_rest'])


    VUDS_ZFOURGE = VUDS_ZFOURGE[(VUDS_ZFOURGE['zflags']==3) | (VUDS_ZFOURGE['zflags']==4) ] #select only >3-sigma detections
    VUDS_ZFOURGE = pd.DataFrame(np.asarray(VUDS_ZFOURGE), index=VUDS_ZFOURGE['id'])


    #do not delete this. Since mass information is given by VUDS this is used in a plot
    #rather than the ZFOURGE matched sample. The ZFOURGE matched sample is used for UVJ colours etc. 
    VUDS_extra = VUDS_extra['vuds_ident','age','log_stellar_mass','log_star_formation_rate','k','ek', 'zflags', 'z_spec']
    VUDS_extra = pd.DataFrame(np.asarray(VUDS_extra), index=VUDS_extra['vuds_ident'])
    VUDS_ZFOURGE = pd.merge(VUDS_ZFOURGE, VUDS_extra, left_on='vuds_ident', right_on='vuds_ident', how='inner')
    VUDS_ZFOURGE.set_index('id', inplace=True)
    VUDS_ZFOURGE.index = VUDS_ZFOURGE.index.astype('str')
    
    

    rest_colours_spec = rest_colours_df(U_ZV, V_ZV, J_ZV, '_rest') 
    rest_colours_spec.index = rest_colours_spec.index.astype('str')

    VUDS_ZFOURGE = pd.merge(VUDS_ZFOURGE, rest_colours_spec,how='left', left_index=True, right_index=True  )


    VUDS_ZFOURGE['UV'] = -2.5*np.log10(VUDS_ZFOURGE['U_rest']/VUDS_ZFOURGE['V_rest'])
    VUDS_ZFOURGE['VJ'] = -2.5*np.log10(VUDS_ZFOURGE['V_rest']/VUDS_ZFOURGE['J_rest'])
    
    

    return COSMOS, MOSDEF_ZFOURGE, VUDS_ZFOURGE, VUDS_extra
    
def open_cosmos_files():
    """
    This function opens files related to the COSMOS field.
    Returns:
        A lot of stuff. Check the code to see what it returns 
    """
    
    COSMOS_mastertable = pd.read_csv('data/zfire/zfire_cosmos_master_table_dr1.1.csv',index_col='Nameobj')
    
    ZF_cat  = ascii.read('data/zfourge/spitler2014/cosmos.v0.10.7.a.cat')
    ZF_EAZY = ascii.read('data/zfourge/spitler2014/cosmos.v0.10.7.a.zout')
    ZF_FAST = ascii.read('data/zfourge/spitler2014/cosmos.v0.10.7.a.fout')
    
    
    #load in colours using spec-z
    #only ZFIRE
    U_spec = ascii.read('data/zfourge/uvj/specz_zfire/cosmos.v0.10.7.a.153.rf')
    V_spec = ascii.read('data/zfourge/uvj/specz_zfire/cosmos.v0.10.7.a.155.rf')
    J_spec = ascii.read('data/zfourge/uvj/specz_zfire/cosmos.v0.10.7.a.161.rf')

    #load in colours using photo-z
    U_photo = ascii.read('data/zfourge/uvj/photoz/cosmos.v0.10.7.a.153.rf')
    V_photo = ascii.read('data/zfourge/uvj/photoz/cosmos.v0.10.7.a.155.rf')
    J_photo = ascii.read('data/zfourge/uvj/photoz/cosmos.v0.10.7.a.161.rf')


    #galaxy colours derived by Lee's catalogue
    #This uses the older EAZY method of fitting colours
    UV_lee = ascii.read('data/zfourge/spitler2014/cosmos.v0.10.7.a.153-155.rf')
    VJ_lee = ascii.read('data/zfourge/spitler2014/cosmos.v0.10.7.a.155-161.rf')
    
    UV_IR_SFRs = ascii.read('data/zfourge/sfrs/cosmos.sfr.v0.5.cat')
    
    MOSDEF_ZFOURGE = ascii.read('data/catalogue_crossmatch/MOSDEF_COSMOS.dat')
    
    
    #ZFIRE and MOSDEF colours 
    U_ZM = ascii.read('data/zfourge/uvj/specz_zfire_mosdef/cosmos.v0.10.7.a.153.rf')
    V_ZM = ascii.read('data/zfourge/uvj/specz_zfire_mosdef/cosmos.v0.10.7.a.155.rf')
    J_ZM = ascii.read('data/zfourge/uvj/specz_zfire_mosdef/cosmos.v0.10.7.a.161.rf')

    VUDS_ZFOURGE = ascii.read('data/catalogue_crossmatch/VUDS_COSMOS.dat')
    
    VUDS_extra = ascii.read('data/vuds/cesam_vuds_spectra_dr1_cosmos_catalog_additional_info.txt')

    #ZFIRE and VUDS colours 
    U_ZV = ascii.read('data/zfourge/uvj/specz_vuds/cosmos.v0.10.7.a.153.rf')
    V_ZV = ascii.read('data/zfourge/uvj/specz_vuds/cosmos.v0.10.7.a.155.rf')
    J_ZV = ascii.read('data/zfourge/uvj/specz_vuds/cosmos.v0.10.7.a.161.rf')


    
    return COSMOS_mastertable, ZF_cat, ZF_EAZY, ZF_FAST, U_spec, V_spec, J_spec,\
            U_photo, V_photo, J_photo, UV_lee, VJ_lee, UV_IR_SFRs, MOSDEF_ZFOURGE,\
            U_ZM,V_ZM, J_ZM, VUDS_ZFOURGE, VUDS_extra, U_ZV, V_ZV, J_ZV


def prepare_uds():
    """
    This function will load data for the UDS field
    Returns:
        1. ZFIRE UDS catalogue with extra info from UKIDSS photometry.
        2. UKIDSS photometry data for all galaxies in UDS field within the z=1.62 cluster region.
    """

    UDS, UDS_photometry, all_UKIDSS, all_UKIDSS_photo_z, all_UKIDSS_U, all_UKIDSS_V, all_UKIDSS_J = open_uds_files()
    
    UDS_photometry       = pd.DataFrame(np.asarray(UDS_photometry), index=UDS_photometry['Keckid'])
    UDS_photometry.index = UDS_photometry.index.astype(str)
    UDS = pd.merge(UDS, UDS_photometry, left_index=True, right_index=True, how='left')
    
    
    all_UKIDSS   = pd.DataFrame(np.asarray(all_UKIDSS), index=all_UKIDSS['id'])
    
    all_UKIDSS_photo_z   = pd.DataFrame(np.asarray(all_UKIDSS_photo_z), index=all_UKIDSS_photo_z['id'])
    all_UKIDSS_U = pd.Series(np.asarray(all_UKIDSS_U['L13']), index=all_UKIDSS_U['id'])
    all_UKIDSS_V = pd.Series(np.asarray(all_UKIDSS_V['L15']), index=all_UKIDSS_V['id'])
    all_UKIDSS_J = pd.Series(np.asarray(all_UKIDSS_J['L21']), index=all_UKIDSS_J['id'])

    all_UKIDSS_photo_z['U'] = all_UKIDSS_U
    all_UKIDSS_photo_z['V'] = all_UKIDSS_V
    all_UKIDSS_photo_z['J'] = all_UKIDSS_J

    all_UKIDSS_photo_z['UV'] = -2.5*np.log10(all_UKIDSS_photo_z['U']/all_UKIDSS_photo_z['V'])
    all_UKIDSS_photo_z['VJ'] = -2.5*np.log10(all_UKIDSS_photo_z['V']/all_UKIDSS_photo_z['J'])

    all_UKIDSS =pd.merge(all_UKIDSS,all_UKIDSS_photo_z, left_index=True, right_index=True,how='inner' )
    UDS = pd.merge(UDS,all_UKIDSS , left_on='DR8id', right_index=True, how='left' )
    UDS = UDS[~UDS.index.duplicated()]
    
    
    #take params for the cluster
    BCG_RA, BCG_DEC, z, dz = uds_clus_param()

    UKIDSS_selected = all_UKIDSS[(all_UKIDSS.ra>(BCG_RA-(0.167*15))) & (all_UKIDSS.ra<(BCG_RA+(0.167*15))) & (all_UKIDSS.dec>(BCG_DEC-0.167)) & 
                       (all_UKIDSS.dec<(BCG_DEC+0.167)) & (all_UKIDSS.z_peak>(z-dz)) &(all_UKIDSS.z_peak<(z+dz))]
    
    
    return UDS, UKIDSS_selected


def open_uds_files():
    """
    This function opens files related to the UDS field.
    Returns:
        zfire catalogue for uds field: UDS 
        basic UKIDSS photometry: UDS_photometry, all_UKIDSS, all_UKIDSS_photo_z 
        UVJ info: all_UKIDSS_U, all_UKIDSS_V, all_UKIDSS_J
    """
    
    #zfire UDS master catalogue
    UDS = pd.read_csv('data/zfire/zfire_uds_master_table_dr1.1.csv', index_col='Nameobj')
    
    #don't ask me why there are so many different files with information repeated.  
    #its the beauty of a collaboration
    
    #uds photometry data: provided by Kim-Vy Tran
    UDS_photometry       = ascii.read('data/ukidss/photometry/keckz-mags.txt')
    
    #rest-frame colour info: Provided by Ryan Quadri
    all_UKIDSS = ascii.read('data/ukidss/uvj/uds8_v0.2.test.cat')
    all_UKIDSS_photo_z = ascii.read('data/ukidss/uvj/udsz.zout')
    all_UKIDSS_U = ascii.read('data/ukidss/uvj/udsz.13.rf')
    all_UKIDSS_V = ascii.read('data/ukidss/uvj/udsz.15.rf')
    all_UKIDSS_J = ascii.read('data/ukidss/uvj/udsz.21.rf')
    
    return UDS, UDS_photometry, all_UKIDSS, all_UKIDSS_photo_z, all_UKIDSS_U, all_UKIDSS_V, all_UKIDSS_J
    
    



def uds_clus_param():
    """
    Define co-ordinates for the Papovich(2010) cluster. 
    Returns:
         Centre RA, DEC, redshift, and the size of redshift bin
    """

    BCG_RA = 2.30597 *15
    BCG_DEC=-5.172194
    z=1.62
    dz=0.05 
    
    return  BCG_RA, BCG_DEC, z, dz
    
    
    
#Exposure Times(in seconds)
Hbandmask1      = 19920 ; Hbandmask2      = 11520
DeepKband1      = 7560  ; DeepKband2      = 7200
KbandLargeArea3 = 11880 ; KbandLargeArea4 = 10260
shallowmask1    = 7200  ; shallowmask2    = 7200
shallowmask3    = 7200  ; shallowmask4    = 3960
UDS_Y1          = 20340 ;
UDS_H1          = 5640  ; UDS_J1          = 3360
UDS_H2          = 6720  ; UDS_J2          = 2880  
UDS_H3          = 2880  ; UDS_J3          = 2880

def open_fits_COSMOS(galaxy, band ):
    """
    v2.1.1: commented the emission line fits opening since it is not needed for the 
    purpose of p(z) stacks
    
    version 2.1: 
    added to open the emission line fit files
    
    upgrading to open multiple object spectra correctly. 
    
    
    version 1.3 21/11/14
    upgrade to take spectra from the master table. The file open order should be:
    1. the common folder to priorotize the objects observed in multiple observing runs
    2. objects in individual observing runs
    For H band option 1 doesn't apply
    Multiobject option has been removed. 
    
    """
    
    if pd.isnull(galaxy.doubles) is True:
        suffix_string = ''
        ID = galaxy.Nameobj
    else:
        double_string = galaxy.doubles
        
        if s.find(double_string, 'b')!=-1:
            suffix_string = '-2'
            ID = s.rstrip(galaxy.doubles, 'b')
        elif s.find(double_string, 'c')!=-1:
            suffix_string = '-3'
            ID = s.rstrip(galaxy.doubles, 'c')
    
    
    if band =='H':
            
        path='../../spectra/spectra_1d/2014feb_1d/after_scaling/spectra/'
      
        try:
            fits = glob.glob(str(path)+'Hbandmask*'+ str(ID) +'_*_1D'+suffix_string+'.fits');# print fits; print path
            Name = fits[0]; eps = pf.open(Name)
        except IndexError:
            print str(Object['Nameobj'])+" Object not found in H band"
            
            return -99, -99, -99
        
        print 'Opened ' + str(Name)        
        
        if   s.find(Name, 'Hbandmask1')!=-1:
            mask = 'Hbandmask1'; ET = Hbandmask1; print 'This is----> ' + str(mask)
        elif s.find(Name, 'Hbandmask2')!=-1:
            mask = 'Hbandmask2'; ET = Hbandmask2; print 'This is----> ' + str(mask) 
        else:
            mask= 'unknown' ; print '**ERROR** mask not recognized: Please Check'
    
        obs_run= 'feb'
    
    
    elif band=='K':
        
        try:
            path='../../spectra/spectra_1d/common_1d/DR1/after_scaling_common_1D/'
            
            fits = glob.glob(str(path)+'*_'+str(ID)+'_coadd1D.fits')
            Name = fits[0]; eps = pf.open(Name); print "opened "+ str(Name)
            
            assert len(fits)==1, "There are 0/multiple matches"
            
           
        except IndexError:
            path='../../spectra/spectra_1d/201*_1d/after_scaling/spectra/'
            fits = glob.glob(str(path)+'*_K_K_*_'+ str(ID) +'_*_1D'+suffix_string+'.fits')
          
            assert len(fits)==1, "There are 0/multiple matches"
            
        if   s.find(Name, 'DeepKband1')!=-1:
            mask = 'DeepKband1' ; ET = DeepKband1 ;  obs_run= 'feb'; print 'This is ' + str(mask); mask='DK1'
                    
        elif s.find(Name, 'DeepKband2')!=-1:
            mask = 'DeepKband2'; ET = DeepKband2 ;  obs_run= 'feb'; print 'This is ' + str(mask); mask='DK2'
                    
        elif s.find(Name, 'KbandLargeArea3')!=-1:
            mask = 'KbandLargeArea3'; ET = KbandLargeArea3 ; obs_run= 'feb'; print 'This is ' + str(mask); mask='KL3'
                    
        elif s.find(Name, 'KbandLargeArea4')!=-1:
            mask = 'KbandLargeArea4'; ET = KbandLargeArea4 ;  obs_run= 'feb';print 'This is ' + str(mask); mask='KL4'
                    
        elif s.find(Name, 'shallowmask1')!=-1:
            mask = 'shallowmask1' ; ET = shallowmask1; obs_run= 'dec'; print 'This is ' + str(mask); mask='SK1'
            
        elif s.find(Name, 'shallowmask2')!=-1:
            mask = 'shallowmask2'; ET = shallowmask2 ; obs_run= 'dec'; print 'This is ' + str(mask); mask='SK2'
            
        elif s.find(Name, 'shallowmask3')!=-1:
            mask = 'shallowmask3'; ET = shallowmask3;  obs_run= 'dec'; print 'This is ' + str(mask); mask='SK3'
            
        elif s.find(Name, 'shallowmask4')!=-1:
            mask = 'shallowmask4'; ET = shallowmask4 ; obs_run= 'dec'; print 'This is ' + str(mask); mask='SK4'
                
        else:
            mask= 'COM' ; ET = -100 ; obs_run= 'decfeb'; print 'Object in both observing runs'
    
    
    return eps, mask, ET, obs_run


def set_sky_weights(band, wave, w):
    """version 1.0
    sets weights around sky lines to be 0 and everything else to be 1 
    the range is determined according to the spectral resolution"""
    
    if    band=='H': sky_lines= sky_H['wavelength']; spec_res = 4.5
    elif  band=='K': sky_lines= sky_K['wavelength']; spec_res = 5.5
    elif  band=='J': sky_lines= sky_J['wavelength']; spec_res = 4.0
    elif  band=='Y': sky_lines= sky_Y['wavelength']; spec_res = 3.5 #not checked
    else: print 'Unknown Band' 
    
    for i, v in enumerate(sky_lines):    
        wave_mask    = np.ma.masked_outside(wave, sky_lines[i]-spec_res, sky_lines[i]+spec_res)
        masked_array = np.ma.getmaskarray(wave_mask)
        np.place(w, masked_array==False, [0])
    
    return w



def get_limits(galaxy):
        
        data = open_fits_COSMOS(galaxy,'K')
        
        scidata, sddata, wavelength, hdr = data[0].data, data[1].data, data[2].data, data[0].header
        
        CRVAL1, CD1_1 , CRPIX1 = hdr['CRVAL1'], hdr['CD1_1'], hdr['CRPIX1']
        i_w        = np.arange(len(scidata))+1 #i_w should start from 1
        wavelength = ((i_w - CRPIX1) * CD1_1 ) + CRVAL1 

        limits                   = np.nonzero(sddata)#masking procedure is fine. Checked 1/09/14
        photometry_mask          = np.ma.masked_inside(wavelength, wavelength[limits[0][0]], wavelength[limits[0][-1]] , copy=True)
        photometry_mask          = np.ma.getmaskarray(photometry_mask)
        Lambda_limits            = wavelength[photometry_mask]
        flux_limits              = scidata[photometry_mask]
        error_limits             = sddata[photometry_mask]
        
        sky_weights        = np.ones_like(Lambda_limits)
        sky_weights        = set_sky_weights('K',Lambda_limits, sky_weights)
        
        print sky_weights[0:20]
        
        fraction_lost = float(len(Lambda_limits[sky_weights==0]))/len(Lambda_limits)
        
        print "fraction lost due to sky = ",fraction_lost
        limit_low  =  (Lambda_limits[0]/6565)-1
        limit_upper = (Lambda_limits[-1]/6565)-1
        
        #make_spectra_plot(Lambda_limits,flux_limits,error_limits,galaxy.zspec, galaxy.Nameobj)
     
        return limit_low, limit_upper, fraction_lost
    


def get_lambda(scidata, hdr):
    
    CRVAL1, CD1_1 , CRPIX1 = hdr['CRVAL1'], hdr['CD1_1'], hdr['CRPIX1']
    i_w        = np.arange(len(scidata)) + 1
    wavelength = ((i_w - CRPIX1) * CD1_1 ) + CRVAL1 
    
    return wavelength
    
def get_limits(hdr,low_x, high_x):
    
    CRVAL1, CD1_1 , CRPIX1        = hdr['CRVAL1'], hdr['CD1_1'], hdr['CRPIX1']
    pix_low, pix_high = np.int(((low_x - CRVAL1) / CD1_1 ) + CRPIX1), np.int(((high_x - CRVAL1) / CD1_1 ) + CRPIX1)
    
    return pix_low, pix_high




def make_subplots_1D(ax,flux_1D,error_1D,wavelength, xlim,z,Name,Band, conf):
    
    
    ax.step(wavelength ,flux_1D, linewidth=1.0,ls='-',
             color='b', alpha=1.0, label='$\mathrm{Flux}$')
    
    ax.step(wavelength ,error_1D, linewidth=0.5,ls='-',
             color='r', alpha=1.0, label='$\mathrm{Error}$')
    
    ax.fill_between(wavelength, flux_1D-error_1D, flux_1D+error_1D,linewidth=0,
                 facecolor='cyan', interpolate=True, edgecolor='white')
   
    if (Name!='9593') and (Name!='7547')  and (Name!='5155') :

        plt.axvline(x=(z+1)*5008.240, ls='--', c='k')
        ax.text(((z+1)*5008.240)+5,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                '$\mathrm{[OIII]}$' )

        plt.axvline(x=(z+1)*4960.295, ls='--', c='k')
        ax.text(((z+1)*4960.295)+5,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                '$\mathrm{[OIII]}$' )

        plt.axvline(x=(z+1)*3728.000, ls='--', c='k')
        ax.text(((z+1)*3728.000)+5,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                '$\mathrm{[OII]}$' )

        plt.axvline(x=(z+1)*4862.680, ls='--', c='k')
        ax.text(((z+1)*4862.680)+5,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                r'$\mathrm{H\beta}$' )

        plt.axvline(x=(z+1)*6564.610, ls='--', c='k')
        ax.text(((z+1)*6564.610)+5,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                r'$\mathrm{H\alpha}$' )

        plt.axvline(x=(z+1)*6585.270, ls='--', c='k')
        ax.text(((z+1)*6585.270)+20,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                '$\mathrm{[NII]}$' )

        plt.axvline(x=(z+1)*6549.860, ls='--', c='k')
        ax.text(((z+1)*6549.860)-120,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                '$\mathrm{[NII]}$' )

        plt.axvline(x=(z+1)*6718.290, ls='--', c='k')
        ax.text(((z+1)*6718.290)-120,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                '$\mathrm{[SII]}$' )

        plt.axvline(x=(z+1)*6732.670, ls='--', c='k')
        ax.text(((z+1)*6732.670)+10,
                np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.85, 
                '$\mathrm{[SII]}$' )

    

    plt.ylabel(r'$\mathrm{Flux\ (10^{-17}ergs/s/cm^2/\AA)}$' ,fontsize=10)
    plt.xlim(xlim[0], xlim[1])
    
    if Name=='3633':
        plt.ylim(-0.2,0.78)
        ax.text( xlim[1]-350, 0.6, 
                        ('$\mathrm{'+str(Name)+'\ '+str(Band)+'}$'+'\n'+'$\mathrm{'+ 'Q_z='+str(conf)+'}$'))
    elif Name=='9593':
        plt.ylim(-0.2,0.2)
        ax.text( xlim[1]-350, 0.10, 
                        ('$\mathrm{'+str(Name)+'\ '+str(Band)+'}$'+'\n'+'$\mathrm{'+ 'Q_z='+str(conf)+'}$'))
    elif Name=='3883':
        plt.ylim(-0.1,0.2)
        ax.text( xlim[1]-330, 0.12, 
                        ('$\mathrm{'+str(Name)+'\ '+str(Band)+'}$'+'\n'+'$\mathrm{'+ 'Q_z='+str(conf)+'}$'))
    elif Name=='7547':
        plt.ylim(-0.2,0.4)
        ax.text( xlim[1]-350, -0.2, 
                        ('$\mathrm{'+str(Name)+'\ '+str(Band)+'}$'+'\n'+'$\mathrm{'+ 'Q_z='+str(conf)+'}$'))
    elif Name=='5155':
        plt.ylim(-0.2,0.38)
        ax.text( xlim[1]-350, 0.2, 
                        ('$\mathrm{'+str(Name)+'\ '+str(Band)+'}$'+'\n'+'$\mathrm{'+ 'Q_z='+str(conf)+'}$'))
        
    else:
    
        plt.ylim(np.min(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.95, 
                 np.max(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*1.05)
        
        if Band=='H\ band':
            ax.text( xlim[1]-250, 
                    np.max(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.60, 
                    ('$\mathrm{'+str(Name)+'\ '+str(Band))+'}$'+'\n'+'$\mathrm{'+ 'Q_z='+str(conf)+'}$')
        else:
            ax.text( xlim[1]-350, 
                        np.max(flux_1D[(wavelength>xlim[0]) & (wavelength<xlim[1])])*0.75, 
                        ('$\mathrm{'+str(Name)+'\ '+str(Band))+'}$'+'\n'+'$\mathrm{'+ 'Q_z='+str(conf)+'}$')
    # We change the fontsize of minor ticks label 
    plt.tick_params(axis='both', which='major', labelsize=10)
    #plt.tick_params(axis='both', which='minor', labelsize=15)
    
    

def make_subplots_2D(spectra_2D,xlim, xlabel=False):
    
    
    pix_limit_low, pix_limit_high = get_limits(hdr, xlim[0], xlim[1])
    
    
    spectra_2D = spectra_2D[:, pix_limit_low: pix_limit_high]
    
    
    plt.imshow(spectra_2D, aspect = 'auto', cmap='gist_gray',
               extent= ( xlim[0], xlim[1] , 40 , 0) ,vmin=-1e-19, vmax=9e-20 )
    
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #extent = left, right, bottom, top
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    if xlabel==True:
        plt.xlabel(r'$\mathrm{Wavelength\ (\AA)}$',fontsize=12)


    
#vmin=-1e-19, vmax=9e-20, cmap='gray', aspect=1,interpolation='none'
    

    
