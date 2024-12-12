import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import wcs
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Circle
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from convert_coordinates import get_relative_coords
from gauss_conv import conv_with_gauss
from reprojection_hex import reproject_hex
from get_large_scale_quantities import get_weighted_averages
from plot_hexagon_patches import get_hexagon_patches

def get_densegas_vs_cloud_data(data_dir, table_path, chcorr_path,
                               glxy, cloud_res, large_res, densegas_list, 
                               mask_dir=None, cloud_mask=False, 
                               avg_method='convolution', Rstar_path=None,
                               diag_dir=None, savediag=True, showdiag=False):
    """
    Takes the data directories of the pystructures (.npy) and the SFR maps (.fits), 
    a list of galaxies, cloud-scale and large-scale resolutions, dates of the 
    structure files and returns the densegas/CO against intensity-weighted averaged 
    CO data. Here densegas is a representative for HCN, HCOP or CS. All output 
    quantities match the large-scale input resolution. The average method defines
    if the intensity weighted averages are computed inside a sharp aperture or via
    a weighted convolution.
    
    INPUT:
    - data_dir_pystruct: string
    - glxys: list of strings
    - cloud_res: string or list of strings (must include unit, i.e. as, pc)
    - large_res: string or list of strings (must include unit, i.e. as, pc)
    - cloud_date: string or list of strings of the form yyyy_mm_dd
    - large_date: string or list of strings of the form yyyy_mm_dd
    - densegas: string or list of strings
    - cloud_mask: list of the form [bool, float, string, string]
    - avg_method: string: 'aperture', 'convolution'
    
    OUTPUT:
    - output dictionary:
        output['int_co21'] = list of CO(2-1) intensities
        output['int_co21_uc'] = uncertaintyies of above
        output['int_densegas'] = list of dense gas intensities
        output['int_densegas_uc'] = uncertainties of above
        output['SigSFR'] = star-formation rate as obtained from GALEX and WISE
        output['SigSFR_uc'] = uncertainties of above
        output['int_co21_avg'] = int.-weighted averaged CO(2-1) intensities
        output['int_co21_avg_uc'] = uncertainties of above # TBD
        output['veldisp_avg'] = int.-weighted averaged velocity dispersion
        output['veldisp_avg_uc'] = uncertainties of above # TBD
        output['alphavir_avg'] = int.-weighted averaged virial parameter
        output['alphavir_avg_uc'] = uncertainties of above # TBD
    
    MODIFIKATION HISTORY:
    - v 1.0: 23/Oct/2020
    - v 1.1: 04/Nov/2020: added masking for cloud-scale data
    - v 1.2: 11/Nov/2020: added velocity dispersion and virial parameter
    - v 1.3: 12/Nov/2020: changed output to dictionary
    - v 1.4: 18/Nov/2020: added SFR data
    - v 1.5: 21/Nov/2020: added ch-ch correlation data for vel. disp.
    - v 2.0: 13/Dec/2020: added uncertainties in the weighted quantities
    - v 2.1: 23/Dec/2020: output of coordinates; diagnostic plots optimisation
    - v 2.2: 11/Jan/2021: add cloud-scale output; bug fix (CO unc. input)
    - v 3.0: 22/Feb/2021: added option for sharp aperture averages
    - v 3.1: 06/May/2021: added stellar masses
    - v 3.2: 18/May/2021: added dynamical equilibrium pressure
    - v 3.3: 31/May/2021: multiple densegas intensities simultaneously
    """

    ####################################################
    # LOAD DATA
    ####################################################

    # filename of CO(2-1) data at cloud-scale resolution
    fname_cloud = glxy + '_data_struct_' + cloud_res + '.npy'
    fpath_cloud = data_dir + fname_cloud

    # filename of densegas and CO(2-1) data at large-scale resolution
    fname_large = glxy + '_data_struct_' + large_res + '.npy'
    fpath_large = data_dir + fname_large  

    if cloud_mask:
        # filename of cloud-scale mask
        fname_mask_cloud = glxy + '_' + cloud_res + '_mask_edge_15as.npy'
        fpath_mask_cloud = mask_dir + fname_mask_cloud

    print('=================================')
    print('[INFO] Galaxy: NGC%s' % glxy[-4:])
    print('=================================')
    print('[INFO] Load cloud-scale structure: ', fpath_cloud)
    print('[INFO] Load large-scale structure: ', fpath_large)
#    print('[INFO] Load cloud-scale mask:      ', fpath_mask_cloud)

    # get CO(2-1) data at cloud-scale
    struct_cloud = np.load(fpath_cloud, allow_pickle=True).item()
    ra_cloud = struct_cloud['ra_deg']               # right ascencion [deg]
    dec_cloud = struct_cloud['dec_deg']             # declination [deg]
    rgal_as_cloud = struct_cloud['rgal_as']         # galactocentric radius [arcsec]
    int_co21_cloud = struct_cloud['INT_VAL_CO21']   # CO(2-1) intensity [K km/s]
    int_co21_uc_cloud = struct_cloud['INT_UC_CO21'] # CO(2-1) intensity uncertainty
    Tpeak_cloud = struct_cloud['SPEC_TPEAK_CO21']   # CO(2-1) peak (specific) intensity [K]
    delta_v_ch = struct_cloud['SPEC_DELTAV_CO21']   # channel width [m/s]
    spec_co21_cloud = struct_cloud['SPEC_VAL_CO21'] # CO(2-1) spectra [K]

    # get densegas, CO(2-1) and SFR data at large-scale
    struct_large = np.load(fpath_large, allow_pickle=True).item()
    ra_large = struct_large['ra_deg']                 # right ascencion [deg]
    dec_large = struct_large['dec_deg']               # declination [deg]
    rgal_as_large = struct_large['rgal_as']           # galactocentric radius [arcsec]
    int_co21_large = struct_large['INT_VAL_CO21']     # CO(2-1) intensity [K km/s]
    int_co21_uc_large = struct_large['INT_UC_CO21']   # CO(2-1) intensity uncertainty

    int_densegas_large_list = []
    int_densegas_uc_large_list = []
    for densegas in densegas_list:
        int_densegas_large_list.append(struct_large['INT_VAL_'+densegas])    # densegas (e.g. HCN) intensity [K km/s]
        int_densegas_uc_large_list.append(struct_large['INT_UC_'+densegas])  # densegas (e.g. HCN) intensity uncertainty            

    SigSFR_large = struct_large['INT_VAL_SFR']        # SFR from GALEX and WISE [Msun/yr/kpc^2]
    SigSFR_uc_large = struct_large['INT_UC_SFR']      # SFR uncertainty

    try:
        Sigstar_large = struct_large['INT_VAL_MSTAR']   # stellar mass (from WISE1) [Msun/pc^2]
        Sigstar_uc_large = struct_large['INT_UC_MSTAR'] # stellar mass uncertainty
        Sigstar_key = True
    except:
        Sigstar_key = False

    try:
        int_HI_large = struct_large['INT_VAL_HI_21CM']       # HI 21cm line intensity [K km/s]
        int_HI_uc_large = struct_large['INT_UC_HI_21CM']     # uncertainty (not yet in structure)
        HI_key = True
    except:
        HI_key = False

    # convert cloud-scale data to relative (galactocentric) coordinates
    delta_ra_cloud, delta_dec_cloud = get_relative_coords(ra_cloud, dec_cloud, rgal_as_cloud)

    # convert large-scale data to relative (galactocentric) coordinates
    delta_ra_large, delta_dec_large = get_relative_coords(ra_large, dec_large, rgal_as_large)

    # get scale unit
    if (cloud_res[-2:] == 'as') & (large_res[-2:] == 'as'):
        scale = 'angular'
    elif (cloud_res[-2:] == 'pc') & (large_res[-2:] == 'pc'):
        scale = 'physical'
    else:
        print('[ERROR] Cloud-scale resolution has to be a string of the form <number>as or <number>pc')


    ####################################################
    # MASKING
    ####################################################        

    # apply cloud-scale mask to cut edge noise
    if cloud_mask:
        # get mask
        mask = np.load(fpath_mask_cloud, allow_pickle=True)
        # convert to bool
        mask_bool = mask.astype(bool)
        # apply
        ra_cloud = ra_cloud[mask_bool]
        dec_cloud = dec_cloud[mask_bool]
        delta_ra_cloud = delta_ra_cloud[mask_bool]
        delta_dec_cloud = delta_dec_cloud[mask_bool]
        rgal_as_cloud = rgal_as_cloud[mask_bool]
        int_co21_cloud = int_co21_cloud[mask_bool]
        int_co21_uc_cloud = int_co21_uc_cloud[mask_bool]
        Tpeak_cloud = Tpeak_cloud[mask_bool]


    ####################################################
    # VELOCITY DISPERSION (following Sun et al. 2018)
    ####################################################

    # in Tpeak change zeros to nan values
    Tpeak_cloud[Tpeak_cloud == 0] = np.nan

    # compute measured effective width (Heyer et al. 2001)
    sigma_measured = int_co21_cloud / np.sqrt(2*np.pi) / Tpeak_cloud  # in km/s
    sigma_measured_uc = int_co21_uc_cloud / np.sqrt(2*np.pi) / Tpeak_cloud  # Gaussian error propagation

    # channel-to-channel correlation coefficient
    # import corr. data
    corr = pd.read_csv(chcorr_path, comment='#')
    idx = np.where(corr['Galaxy']==glxy)[0][0]

    if cloud_res[-2:] == 'as':
        r = float(corr['r_1-2as'][idx])
    elif cloud_res == '150pc':
        r = float(corr['r_150pc'][idx])
    elif cloud_res == '120pc':
        r = float(corr['r_120pc'][idx])
    elif cloud_res == '75pc':
        r = float(corr['r_75pc'][idx])
    elif cloud_res == '123pc':
        # special case: NGC4321
        r = float(corr['r_120pc'][idx])
    else:
        print('[ERROR] Channel-to-channel correlation coefficient r could not be assigned!')

    # neglect channel-to-channel correlation (for this run)
#    r = 0

    # coupling between adjacent channels (Leroy et al. 2016)
    k = 0.0 + 0.47*r - 0.23*r**2 - 0.16*r**3 + 0.43*r**4

    # compute spectral response curve width (Leroy et al. 2016)
    delta_v_ch *= 1e-3  # convert from m/s to km/s
    delta_v_ch = np.abs(delta_v_ch)  # take absolute value
    sigma_response = delta_v_ch / np.sqrt(2) * (1.0 + 1.18*k + 10.4*k**2)  # in km/s

    # change NaNs to zeros for valid comparison (next step)
    sigma_measured[np.isnan(sigma_measured) & np.isnan(sigma_measured_uc)] = 0

    # change sigma_measured to inf values if sigma_measured < sigma_response
    id_invalid_sqrt = sigma_measured < sigma_response
    sigma_measured[id_invalid_sqrt] = np.nan
    sigma_measured_uc[id_invalid_sqrt] = np.nan

    # compute velocity dispersion (Rosolowsky & Leroy 2006)
    veldisp_cloud = np.sqrt(sigma_measured**2 - sigma_response**2)
    veldisp_uc_cloud = sigma_measured/veldisp_cloud * sigma_measured_uc

    # replace nan values with zeros (for convolution function to work)
    veldisp_cloud[np.isnan(veldisp_cloud)] = 0
    veldisp_uc_cloud[np.isnan(veldisp_uc_cloud)] = 0


    ####################################################
    # VIRIAL PARAMETER (following Sun et al. 2018)
    ####################################################

    # constants to compute molecular gas surface density
    alpha_co10 = 4.3  # [M_Sun/pc^2/(K km/s)] (Bolatto et al. 2013)
    alpha_co21 = 1/0.64 * alpha_co10  # (den Brok et al. 2021)

    # compute cloud-scale molecular gas surface density from CO(2-1) intensity
    Sigmol_cloud = alpha_co21 * int_co21_cloud  # [M_Sun/pc^2]
    Sigmol_uc_cloud = alpha_co21 * int_co21_uc_cloud  # Gaussian error propagation

    # geometry factor for a density profile proportional to 1/r
    f = 10/9

    # gravitational constant
    G = 4.30091e-3  # [pc M_Sun^(-1) (km/s)^2]

    # get distances from data table
    table = pd.read_csv(table_path, comment='#', skiprows=1)
    idx = np.where(table['name']==glxy)[0][0]
    dist = table['dist'][idx]
    incl = table['orient_incl'][idx]

    # get cloud-scale resolution in parsec (cutout unit and cast to float)
    if scale == 'angular':
        cloud_res_pc = float(cloud_res[:-2]) * np.pi/180 / 3600 * dist * 1e6
    elif scale == 'physical':
        cloud_res_pc = float(cloud_res[:-2])  

    # put zeros and negative values in Sigmol and veldisp to NaN to allow division
    Sigmol_cloud[Sigmol_cloud <= 0] = np.nan
    veldisp_cloud[veldisp_cloud <= 0] = np.nan

    # compute beam radius from resolution (beam diameter)
    r_beam_pc = cloud_res_pc / 2

    # compute the virial parameter (Sun et al. 2018)
    alphavir_cloud = 5*np.log(2)/np.pi/f/G * veldisp_cloud**2 / Sigmol_cloud / r_beam_pc  # [dimensionless]
    alphavir_uc_cloud = alphavir_cloud * np.sqrt(4*(veldisp_uc_cloud/veldisp_cloud)**2 + (Sigmol_uc_cloud/Sigmol_cloud)**2)  # Gaussian error propagation

    # redo nan replacement of Sigmol and veldisp
    Sigmol_cloud[np.isnan(Sigmol_cloud)] = 0
    veldisp_cloud[np.isnan(veldisp_cloud)] = 0

    # replace nan values with zeros
    alphavir_cloud[np.isnan(alphavir_cloud)] = 0
    alphavir_uc_cloud[np.isnan(alphavir_uc_cloud)] = 0


    ####################################################
    # INTERNAL TURBULENT PRESSURE (following Sun et al. 2018)
    ####################################################

    # compute internal turbulent pressure
    Pturb_cloud = 61.3 * Sigmol_cloud * veldisp_cloud**2 / (r_beam_pc/40)  # [K cm^(-3) kB]

    # compute uncertainty
    #Pturb_uc_cloud = Pturb_cloud * np.sqrt((Sigmol_uc_cloud/Sigmol_cloud)**2 + (2*veldisp_uc_cloud/veldisp_cloud)**2)

    Pturb_uc_cloud = 61.3 * np.sqrt((veldisp_cloud**2/(r_beam_pc/40)*Sigmol_uc_cloud)**2 \
                                    + (2*Sigmol_cloud*veldisp_cloud/(r_beam_pc/40)*veldisp_uc_cloud)**2)


    ####################################################
    # INTENSITY-WEIGHTED AVERAGES (following Leroy et al. 2016)
    ####################################################

    # get cloud-scale and large-scale resolution in arcsec (cutout unit and cast to float)
    if scale == 'angular':
        cloud_res_as = float(cloud_res[:-2])  
        large_res_as = float(large_res[:-2])
    elif scale == 'physical':
        cloud_res_as = 180/np.pi * 3600 * float(cloud_res[:-2]) / dist * 1e-6
        large_res_as = 180/np.pi * 3600 * float(large_res[:-2]) / dist * 1e-6

    # get intensity-weighted average of the CO(2-1) intensity
    print('---------------------------------')
    print('Weighted CO(2-1) intensity average')
    int_co21_avg, int_co21_avg_uc = get_weighted_averages(ra_cloud, dec_cloud, int_co21_cloud, int_co21_cloud, ra_large, dec_large, 
                                                          cloud_res_as, large_res_as, unc=int_co21_uc_cloud, weight_unc=int_co21_uc_cloud, 
                                                          method=avg_method)


    # get intensity-weighted average of the velocity dispersion
    print('---------------------------------')
    print('Weighted velocity dispersion average')
    veldisp_avg, veldisp_avg_uc = get_weighted_averages(ra_cloud, dec_cloud, veldisp_cloud, int_co21_cloud, ra_large, dec_large, 
                                                        cloud_res_as, large_res_as, unc=veldisp_uc_cloud, weight_unc=int_co21_uc_cloud,
                                                        method=avg_method)

    # get intensity-weighted average of the virial parameter
    print('---------------------------------')
    print('Weighted virial parameter average')
    alphavir_avg, alphavir_avg_uc = get_weighted_averages(ra_cloud, dec_cloud, alphavir_cloud, int_co21_cloud, ra_large, dec_large, 
                                                          cloud_res_as, large_res_as, unc=alphavir_uc_cloud, weight_unc=int_co21_uc_cloud,
                                                          method=avg_method)

    # get intensity-weighted average of the internal turbulent pressure
    print('---------------------------------')
    print('Weighted internal turbulent pressure average')
    Pturb_avg, Pturb_avg_uc = get_weighted_averages(ra_cloud, dec_cloud, Pturb_cloud, int_co21_cloud, ra_large, dec_large, 
                                                          cloud_res_as, large_res_as, unc=Pturb_uc_cloud, weight_unc=int_co21_uc_cloud,
                                                          method=avg_method)


    ####################################################
    # STAR FORMATION RATE SURFACE DENSITY
    ####################################################

    # correct for inclination
    SigSFR_large *= np.cos(np.deg2rad(incl))
    SigSFR_uc_large *= np.cos(np.deg2rad(incl))


    ####################################################
    # STELLAR MASS SURFACE DENSITY (following PHANGS: Querejeta 2019)
    ####################################################

    if Sigstar_key:
        # convert list to numpy array
        Sigstar_large = np.array(Sigstar_large) * np.cos(np.deg2rad(incl))
        Sigstar_uc_large = np.array(Sigstar_uc_large) * np.cos(np.deg2rad(incl))

## old version below (new version from WISE1 already in Msun/pc^2)
#        # convert from MJy/sr to Lsun/pc^2
#        Sigstar_large *= 704.04 * np.cos(np.deg2rad(incl))  # (Querejeta et al. 2015) [Lsun/pc^2]
#        Sigstar_uc_large *= 704.04 * np.cos(np.deg2rad(incl))
#
#        # convert from Lsun/pc^2 to Msun/pc^2
#        Sigstar_large *= 0.47  # (McGaugh & Schombert 2014) [Msun/pc^2]
#        Sigstar_uc_large *= 0.47



    ####################################################
    # DYNAMICAL EQUILIBRIUM PRESSURE (following Sun et al. 2020)
    ####################################################    

    if Sigstar_key & HI_key:
        
        ## METHOD 1: P_DE from cloud-scale mol. gas data and large-scale atomic and stellar data
        
        # compute large-scale molecular gas surface density
        Sigmol_large = alpha_co21 * int_co21_large * np.cos(np.deg2rad(incl))  # [M_Sun/pc^2]
        Sigmol_uc_large = alpha_co21 * int_co21_uc_large * np.cos(np.deg2rad(incl))  # Gaussian error propagation

        # load radial scale length of the stellar disk (from S4G)
        table_Rstar = pd.read_csv(Rstar_path)
        id_Rstar = np.where(table_Rstar['name']==glxy)[0][0]
        Rstar_as = np.float(table_Rstar['R_star'][id_Rstar])  # [arcsec]
        Rstar_pc = Rstar_as * np.pi/180 / 3600 * dist * 1e6  # [pc]

        # compute stellar mass density from surface density
        rhostar_large = Sigstar_large / (0.54 * Rstar_pc)  # [Msun/pc^3]

        # compute Sigatom from HI 21cm line intensity
        Sigatom_large = 1.97e-2 * int_HI_large * np.cos(np.deg2rad(incl))  # [Msun/pc^2]
        #Sigatom_uc_large = 1.97e-2 * int_HI_uc_large * np.cos(np.deg2rad(incl))  # TBD

        # sample large-scale Sigmol and rhostar at cloud-scale resolution
        Sigmol_large_samp = reproject_hex(data=Sigmol_large, ra_in=ra_large, dec_in=dec_large, ra_out=ra_cloud, dec_out=dec_cloud)
        # uc TBD
        rhostar_large_samp = reproject_hex(data=rhostar_large, ra_in=ra_large, dec_in=dec_large, ra_out=ra_cloud, dec_out=dec_cloud)
        # uc TBD

        # compute molecular gas weight at cloud-scale
        Wmol_cloud_self   = 24.8 * Sigmol_cloud**2
        Wmol_cloud_extmol = 33.1 * Sigmol_cloud * Sigmol_large_samp
        Wmol_cloud_star   = 49.7 * Sigmol_cloud * rhostar_large_samp * cloud_res_pc

        Wmol_cloud = Wmol_cloud_self + Wmol_cloud_extmol + Wmol_cloud_star     

        # get mass-weighted average of molecular gas self weight
        print('---------------------------------')
        print('Weighted dynamical equilibrium pressure average')
        Wmol_self_avg = get_weighted_averages(ra_cloud, dec_cloud, Wmol_cloud_self, int_co21_cloud, ra_large, dec_large, 
                                         cloud_res_as, large_res_as, unc=None, weight_unc=None, method=avg_method)

        # get mass-weighted average of molecular gas weight
        print('---------------------------------')
        print('Weighted dynamical equilibrium pressure average')
        Wmol_avg = get_weighted_averages(ra_cloud, dec_cloud, Wmol_cloud, int_co21_cloud, ra_large, dec_large, 
                                         cloud_res_as, large_res_as, unc=None, weight_unc=None, method=avg_method)

        # compute atomic gas weight at large-scale
        Watom_large  = 33.1 * Sigatom_large**2
        Watom_large += 66.2 * Sigatom_large * Sigmol_large
        Watom_large += 321 * Sigatom_large * rhostar_large**0.5

        # compute mass-weighted dynamical equilibirum pressure (large-scale)
        PDE_avg = Wmol_avg + Watom_large
        
        
        ## Method 2: P_DE at large-scale (no weighted average)
        
        # compute total gas surface density
        Siggas_large = Sigmol_large + Sigatom_large  # [Msun/pc^2]
        
        # compute molecular gas fraction (of total gas mass)
        fmol_large = Sigmol_large / Siggas_large  # [dimensionless]
        
        # compute weighted velocity dispersion of molecular + atomic gas (Sun+20, Equ. 14)
        vdis_atom = 10  # [km/s]
        vdis_gas_large = fmol_large * veldisp_avg + (1 - fmol_large) * vdis_atom  # [km/s]
        
        # compute dynamical equilibrium pressure  (Sun+20, Equ. 12)
        PDE_large = 33.1 * Siggas_large*2 + 32.1 * Siggas_large * rhostar_large**0.5 * vdis_gas_large  # [kB K km/s]


    ####################################################
    # POLISH DATA (remove zeros and non-overlapping pixels)
    #################################################### 

    # remove data where CO(2-1) intensity or dense gas intensity is zero
    mask_COnull = (int_co21_large != 0) & (int_densegas_large_list[0] != 0)

    delta_ra_large = delta_ra_large[mask_COnull]
    delta_dec_large = delta_dec_large[mask_COnull]
    rgal_as_large = rgal_as_large[mask_COnull]
    int_co21_large = int_co21_large[mask_COnull]
    int_co21_uc_large = int_co21_uc_large[mask_COnull]
    for i in range(len(int_densegas_large_list)):
        int_densegas_large_list[i] = int_densegas_large_list[i][mask_COnull]
        int_densegas_uc_large_list[i] = int_densegas_uc_large_list[i][mask_COnull]
    SigSFR_large = SigSFR_large[mask_COnull]
    SigSFR_uc_large = SigSFR_uc_large[mask_COnull]
    int_co21_avg = int_co21_avg[mask_COnull]
    int_co21_avg_uc = int_co21_avg_uc[mask_COnull]
    veldisp_avg = veldisp_avg[mask_COnull]
    veldisp_avg_uc = veldisp_avg_uc[mask_COnull]
    alphavir_avg = alphavir_avg[mask_COnull]
    alphavir_avg_uc = alphavir_avg_uc[mask_COnull]
    Pturb_avg = Pturb_avg[mask_COnull]
    Pturb_avg_uc = Pturb_avg_uc[mask_COnull]
    if Sigstar_key:
        Sigstar_large = Sigstar_large[mask_COnull]
        Sigstar_uc_large = Sigstar_uc_large[mask_COnull]
    if HI_key:
        Sigatom_large = Sigatom_large[mask_COnull]
    if Sigstar_key & HI_key:
        PDE_avg = PDE_avg[mask_COnull]
        Wmol_self_avg = Wmol_self_avg[mask_COnull]
        Wmol_avg = Wmol_avg[mask_COnull]
        PDE_large = PDE_large[mask_COnull]


    # remove data outside overlap with cloud-scale map
    mask_overlap = get_overlap_ind(delta_ra_large, delta_dec_large, delta_ra_cloud, delta_dec_cloud, rounding=0.1, tolerance=5)
    mask_overlap = mask_overlap.astype(bool)

    delta_ra_large = delta_ra_large[mask_overlap]
    delta_dec_large = delta_dec_large[mask_overlap]
    rgal_as_large = rgal_as_large[mask_overlap]
    int_co21_large = int_co21_large[mask_overlap]
    int_co21_uc_large = int_co21_uc_large[mask_overlap]
    for i in range(len(int_densegas_large_list)):
        int_densegas_large_list[i] = int_densegas_large_list[i][mask_overlap]
        int_densegas_uc_large_list[i] = int_densegas_uc_large_list[i][mask_overlap]
    SigSFR_large = SigSFR_large[mask_overlap]
    SigSFR_uc_large = SigSFR_uc_large[mask_overlap]
    int_co21_avg = int_co21_avg[mask_overlap]
    int_co21_avg_uc = int_co21_avg_uc[mask_overlap]
    veldisp_avg = veldisp_avg[mask_overlap]
    veldisp_avg_uc = veldisp_avg_uc[mask_overlap]
    alphavir_avg = alphavir_avg[mask_overlap]
    alphavir_avg_uc = alphavir_avg_uc[mask_overlap]
    Pturb_avg = Pturb_avg[mask_overlap]
    Pturb_avg_uc = Pturb_avg_uc[mask_overlap]
    if Sigstar_key:
        Sigstar_large = Sigstar_large[mask_overlap]
        Sigstar_uc_large = Sigstar_uc_large[mask_overlap]
    if HI_key:
        Sigatom_large = Sigatom_large[mask_overlap]
    if Sigstar_key & HI_key:
        PDE_avg = PDE_avg[mask_overlap]
        Wmol_self_avg = Wmol_self_avg[mask_overlap]
        Wmol_avg = Wmol_avg[mask_overlap]
        PDE_large = PDE_large[mask_overlap]
        

    ####################################################
    # GALATOCENTRIC RADIUS
    ####################################################  
        
    # get galactocentric radius in parsec
    rgal_pc_cloud = rgal_as_cloud * np.pi/180 / 3600 * dist * 1e6
    rgal_pc_large = rgal_as_large * np.pi/180 / 3600 * dist * 1e6


    ####################################################
    # DIAGNOSTIC PLOTS
    ####################################################

    if showdiag | savediag:

        # matplotlib settings
        rc('font', family='serif', size=16)
        rc('mathtext', fontset='dejavuserif')
        rc('figure', figsize=(1.41421356237*6.,6.)) 
        rc('lines', linewidth=1.2, marker=None, markersize=8)
        rc('axes', linewidth=1.2, labelsize=12, prop_cycle=plt.cycler(color=('k','r','b','darkorange','steelblue','hotpink','gold','c','maroon','darkgreen')) )
        rc(('xtick.major','ytick.major'), size=5, width=1)
        rc(('xtick.minor','ytick.minor'), size=3, width=1, visible=True)
        rc(('xtick','ytick'), labelsize=12, direction='in')
        rc(('xtick'), top=True, bottom=True) # For some stupid reason you have to do these seperately
        rc(('ytick'), left=True, right=True)
        rc('legend', numpoints=1, scatterpoints=1, labelspacing=0.2, fontsize=16, fancybox=True, handlelength=1.5, handletextpad=0.5)
        rc('savefig', dpi=150, format='png',bbox='tight')
        rc('errorbar', capsize=3.)

        # convert angular units to arcsec and physical units to kpc
        if scale == 'angular':
            pass  # arcsec
        elif scale == 'physical':
            delta_ra_cloud *= np.pi/180/3600 * dist*1e6 * 1e-3   # kpc
            delta_dec_cloud *= np.pi/180/3600 * dist*1e6 * 1e-3  # kpc
            delta_ra_large *= np.pi/180/3600 * dist*1e6 * 1e-3   # kpc
            delta_dec_large *= np.pi/180/3600 * dist*1e6 * 1e-3  # kpc

        # set hexagon size
        if cloud_res[-2:] == 'as':
            marker_size_cloud = .2
        elif cloud_res == '75pc':
            marker_size_cloud = .2
        elif cloud_res == '120pc':
            marker_size_cloud = .25
        elif cloud_res == '150pc':
            marker_size_cloud = .3

        # figure size
        plt.figure(figsize=(16,16))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        plt.suptitle(r'NGC%s' % glxy[3:], y=0.92, fontsize=24)

        # plot 1 - CO(2-1) intensity at cloud-scale
        ax1 = plt.subplot(3, 4, 1)
        im1 = ax1.scatter(delta_ra_cloud, delta_dec_cloud, c=np.log10(int_co21_cloud), marker="o", s=marker_size_cloud, cmap="RdYlBu_r", zorder=2)
        ax1_divider = make_axes_locatable(ax1)
        cax1 = ax1_divider.append_axes("top", size="5%", pad="2%")
        cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
        cbar1.ax.set_title(r'$\log_{10}(I_\mathrm{CO(2-1)})$ (K km/s)', size=16)

        # plot 2 - velocity dispersion at cloud-scale
        ax2 = plt.subplot(3, 4, 2)
        im2 = ax2.scatter(delta_ra_cloud, delta_dec_cloud, c=np.log10(veldisp_cloud), marker="o", s=marker_size_cloud, cmap="RdYlBu_r", zorder=2)
        ax2_divider = make_axes_locatable(ax2)
        cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
        cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
        cbar2.ax.set_title(r'$\log_{10}(\sigma)$ (km/s)', size=16)

        # plot 3 - virial parameter at cloud-scale
        ax3 = plt.subplot(3, 4, 3)
        im3 = ax3.scatter(delta_ra_cloud, delta_dec_cloud, c=np.log10(alphavir_cloud), marker="o", s=marker_size_cloud, cmap="RdYlBu_r", zorder=2)
        ax3_divider = make_axes_locatable(ax3)
        cax3 = ax3_divider.append_axes("top", size="5%", pad="2%")
        cbar3 = plt.colorbar(im3, cax=cax3, orientation='horizontal')
        cbar3.ax.set_title(r'$\log_{10}(\alpha_\mathrm{vir})$', size=16)

        # plot 4 - internal turbulent pressure at cloud-scale
        ax4 = plt.subplot(3, 4, 4)
        im4 = ax4.scatter(delta_ra_cloud, delta_dec_cloud, c=np.log10(Pturb_cloud), marker="o", s=marker_size_cloud, cmap="RdYlBu_r", zorder=2)
        ax4_divider = make_axes_locatable(ax4)
        cax4 = ax4_divider.append_axes("top", size="5%", pad="2%")
        cbar4 = plt.colorbar(im4, cax=cax4, orientation='horizontal')
        cbar4.ax.set_title(r'$\log_{10}(P_\mathrm{turb}/k_B)\;(\mathrm{K}\,\mathrm{cm}^{-3})$', size=16)


        # plot 5 - intensity-weighted average of CO(2-1) intensity
        ax5 = plt.subplot(3, 4, 5)
        Sm = alpha_co21 * int_co21_avg
        c = np.log10(Sm[Sm > 0])
        x = delta_ra_large[Sm > 0]
        y = delta_dec_large[Sm > 0]
        patch_list = get_hexagon_patches(x=x, y=y, c=c, cmap='RdYlBu_r', zorder=2,)
        for patch in patch_list: ax5.add_patch(patch)
        im5 = ax5.scatter(x, y, c=c, marker='None', cmap='RdYlBu_r')
        ax5_divider = make_axes_locatable(ax5)
        cax5 = ax5_divider.append_axes("top", size="5%", pad="2%")
        cbar5 = plt.colorbar(im5, cax=cax5, orientation='horizontal')
        cbar5.ax.set_title(r'$\log_{10}(\langle \Sigma_\mathrm{mol}\rangle\;(M_\odot\,\mathrm{pc}^{-2}))$', size=16) 

        # plot 6 - intensity-weighted average of velocity dispersion
        ax6 = plt.subplot(3, 4, 6)
        c = np.log10(veldisp_avg[veldisp_avg > 0])
        x = delta_ra_large[veldisp_avg > 0]
        y = delta_dec_large[veldisp_avg > 0]
        patch_list = get_hexagon_patches(x=x, y=y, c=c, cmap='RdYlBu_r', zorder=2)
        for patch in patch_list: ax6.add_patch(patch)
        im6 = ax6.scatter(x, y, c=c, marker='None', cmap='RdYlBu_r')
        ax6_divider = make_axes_locatable(ax6)
        cax6 = ax6_divider.append_axes("top", size="5%", pad="2%")
        cbar6 = plt.colorbar(im6, cax=cax6, orientation='horizontal')
        cbar6.ax.set_title(r'$\log_{10}(\langle\sigma\rangle)$ (km/s)', size=16) 

        # plot 7 - intensity-weighted average of virial parameter
        ax7 = plt.subplot(3, 4, 7)
        c = np.log10(alphavir_avg[alphavir_avg > 0])
        x = delta_ra_large[alphavir_avg > 0]
        y = delta_dec_large[alphavir_avg > 0]
        patch_list = get_hexagon_patches(x=x, y=y, c=c, cmap='RdYlBu_r', zorder=2)
        for patch in patch_list: ax7.add_patch(patch)
        im7 = ax7.scatter(x, y, c=c, marker='None', cmap='RdYlBu_r')
        ax7_divider = make_axes_locatable(ax7)
        cax7 = ax7_divider.append_axes("top", size="5%", pad="2%")
        cbar7 = plt.colorbar(im7, cax=cax7, orientation='horizontal')
        cbar7.ax.set_title(r'$\log_{10}(\langle\alpha_\mathrm{vir}\rangle)$', size=16)

        # plot 8 - intensity-weighted average of internal turbulent pressure
        ax8 = plt.subplot(3, 4, 8)
        c = np.log10(Pturb_avg[Pturb_avg > 0])
        x = delta_ra_large[Pturb_avg > 0]
        y = delta_dec_large[Pturb_avg > 0]
        patch_list = get_hexagon_patches(x=x, y=y, c=c, cmap='RdYlBu_r', zorder=2)
        for patch in patch_list: ax8.add_patch(patch)
        im8 = ax8.scatter(x, y, c=c, marker='None', cmap='RdYlBu_r')
        ax8_divider = make_axes_locatable(ax8)
        cax8 = ax8_divider.append_axes("top", size="5%", pad="2%")
        cbar8 = plt.colorbar(im8, cax=cax8, orientation='horizontal')
        cbar8.ax.set_title(r'$\log_{10}(\langle P_\mathrm{turb}/k_B\rangle)\;(\mathrm{K}\,\mathrm{cm}^{-3})$', size=16)

        # plot 7 - CO(2-1) intensity at large-scale
        #ax7 = plt.subplot(4, 3, 7)
        #patch_list = get_hexagon_patches(x=delta_ra_large, y=delta_dec_large, c=int_co21_large, cmap='RdYlBu_r', zorder=2)
        #for patch in patch_list: ax7.add_patch(patch)
        #im7 = ax7.scatter(delta_ra_large, delta_dec_large, c=int_co21_large, marker='None', cmap='RdYlBu_r')
        #ax7_divider = make_axes_locatable(ax7)
        #cax7 = ax7_divider.append_axes("top", size="5%", pad="2%")
        #cbar7 = plt.colorbar(im7, cax=cax7, orientation='horizontal')
        #cbar7.ax.set_title(r'$I_\mathrm{CO(2-1)}$ (K km/s)', size=18)

        # plot 9 - densegas intensity at large-scale
        ax9 = plt.subplot(3, 4, 9)
        patch_list = get_hexagon_patches(x=delta_ra_large, y=delta_dec_large, c=np.log10(int_densegas_large_list[0]), cmap='RdYlBu_r', zorder=2)
        for patch in patch_list: ax9.add_patch(patch)
        im9 = ax9.scatter(delta_ra_large, delta_dec_large, c=np.log10(int_densegas_large_list[0]), marker='None', cmap='RdYlBu_r')
        ax9_divider = make_axes_locatable(ax9)
        cax9 = ax9_divider.append_axes("top", size="5%", pad="2%")
        cbar9 = plt.colorbar(im9, cax=cax9, orientation='horizontal')
        if densegas_list[0] == 'HCOP':
            cbar9.ax.set_title(r'$\log_{10}(I_{\mathrm{HCO}^+})$ (K km/s)', size=16)
        else:
            cbar9.ax.set_title(r'$\log_{10}(I_\mathrm{%s})$ (K km/s)' % densegas_list[0], size=16)

        # plot 10 - densegas/CO(2-1) at large-scale
        ax10 = plt.subplot(3, 4, 10)
        patch_list = get_hexagon_patches(x=delta_ra_large, y=delta_dec_large, c=np.log10(int_densegas_large_list[0]/int_co21_large), cmap='RdYlBu_r', zorder=2,
                                         vmin=-2.5, vmax=-0.5)
        for patch in patch_list: ax10.add_patch(patch)
        im10 = ax10.scatter(delta_ra_large, delta_dec_large, c=np.log10(int_densegas_large_list[0]/int_co21_large), marker='None', cmap='RdYlBu_r',
                            vmin=-2.5, vmax=-0.5)
        ax10_divider = make_axes_locatable(ax10)
        cax10 = ax10_divider.append_axes("top", size="5%", pad="2%")
        cbar10 = plt.colorbar(im10, cax=cax10, orientation='horizontal', extend='both')
        if densegas_list[0] == 'HCOP':
            cbar10.ax.set_title(r'$\log_{10}(I_{\mathrm{HCO}^+}/I_\mathrm{CO(2-1)})$', size=16)
        else:
            cbar10.ax.set_title(r'$\log_{10}(I_\mathrm{%s}/I_\mathrm{CO(2-1)})$' % densegas_list[0], size=16)

        # plot 11 - star-formation rate at large-scale
        ax11 = plt.subplot(3, 4, 11)
        patch_list = get_hexagon_patches(x=delta_ra_large, y=delta_dec_large, c=np.log10(SigSFR_large), cmap='RdYlBu_r', zorder=2)
        for patch in patch_list: ax11.add_patch(patch)
        im11 = ax11.scatter(delta_ra_large, delta_dec_large, c=np.log10(SigSFR_large), marker='None', cmap='RdYlBu_r')
        ax11_divider = make_axes_locatable(ax11)
        cax11 = ax11_divider.append_axes("top", size="5%", pad="2%")
        cbar11 = plt.colorbar(im11, cax=cax11, orientation='horizontal')
        cbar11.ax.set_title(r'$\log_{10}(\Sigma_\mathrm{SFR})$', size=16)
        #cbar11.ax.set_title(r'$\Sigma_\mathrm{SFR}$ (M$_\odot$/yr/kpc$^2$)', size=18)

        # plot 12 - star-formation effience of dense gas at large-scale
        ax12 = plt.subplot(3, 4, 12)
        patch_list = get_hexagon_patches(x=delta_ra_large, y=delta_dec_large, c=np.log10(SigSFR_large/int_densegas_large_list[0]), cmap='RdYlBu_r', zorder=2, 
                                         vmin=-2, vmax=0.5)
        for patch in patch_list: ax12.add_patch(patch)
        im12 = ax12.scatter(delta_ra_large, delta_dec_large, c=np.log10(SigSFR_large/int_densegas_large_list[0]), marker='None', cmap='RdYlBu_r', 
                            vmin=-2, vmax=0.5)
        ax12_divider = make_axes_locatable(ax12)
        cax12 = ax12_divider.append_axes("top", size="5%", pad="2%")
        cbar12 = plt.colorbar(im12, cax=cax12, orientation='horizontal', extend='both')
        if densegas_list[0] == 'HCOP':
            cbar12.ax.set_title(r'$\log_{10}(\Sigma_\mathrm{SFR}/I_{\mathrm{HCO}^+})$', size=16)
            #cbar12.ax.set_title(r'$\Sigma_\mathrm{SFR}/I_{\mathrm{HCO}^+}$ (M$_\odot$/yr/kpc$^2/$(K km/s))', size=18)
        else:
            cbar12.ax.set_title(r'$\log_{10}(\Sigma_\mathrm{SFR}/I_\mathrm{%s})$' % densegas_list[0], size=16)
            #cbar12.ax.set_title(r'$\Sigma_\mathrm{SFR}/I_\mathrm{%s}$ (M$_\odot$/yr/kpc$^2/$(K km/s))' % densegas_list[0], size=18)


        # axis labels
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
            ax.invert_xaxis()
            ax.axis('equal')
            ax.tick_params(axis='both', labelsize=0)
            if scale == 'angular':
                ax.set_xlim(125, -125)
                ax.set_ylim(-125, 125)
            elif scale == 'physical':
                ax.set_xlim(10, -10)
                ax.set_ylim(-10, 10)

        # x-axis labels
        for ax in [ax9, ax10, ax11, ax12]:
            if scale == 'angular':
                ax.set_xlabel(r'$\Delta\,$R.A. ($^{\prime\prime}$)', size=16)
            elif scale == 'physical':
                ax.set_xlabel(r'$\Delta\,$R.A. (kpc)', size=16)
            ax.tick_params(axis='x', labelsize=12)

        # y-axis labels
        for ax in [ax1, ax5, ax9]:
            if scale == 'angular':
                cloud_res_label = cloud_res[:-2] + '$^{\prime\prime}$'
                large_res_label = large_res[:-2] + '$^{\prime\prime}$'
                ax.set_ylabel(r'$\Delta\,$Dec. ($^{\prime\prime}$)', size=16)
            elif scale == 'physical':
                cloud_res_label = cloud_res[:-2] + ' pc'
                large_res_label = large_res[:-2] + ' pc'
                ax.set_ylabel(r'$\Delta\,$Dec. (kpc)', size=16)
            ax.tick_params(axis='y', labelsize=12) 

        # row labels
        ax4.text(1.2, 0.5, 'cloud-scale (%s)' % cloud_res_label, horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, rotation=90, size=16)
        ax8.text(1.2, 0.5, 'weigthed averages (%s)' % large_res_label, horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes, rotation=90, size=16)
        ax12.text(1.2, 0.5, 'dense gas (%s)' % large_res_label, horizontalalignment='center', verticalalignment='center', transform=ax12.transAxes, rotation=90, size=16)


        # colorbar
        for cax in [cax1, cax2, cax3, cax4, cax5, cax6, cax7, cax8, cax9, cax10, cax11, cax12]:
            cax.xaxis.set_ticks_position("top")
            cax.tick_params(labelsize=12)

        # cloud-scale beam size
        for ax in [ax1, ax2, ax3, ax4]:
            if scale == 'angular': 
                beam_size = float(cloud_res[:-2])
                beam = Circle((110-beam_size/2,-110+beam_size/2), beam_size/2, color='k', fill=1, zorder=3)
            elif scale == 'physical':
                beam_size = float(cloud_res[:-2])*1e-3  # in kpc
                beam = Circle((9.3-beam_size/2,-9.3+beam_size/2), beam_size/2, color='k', fill=1, zorder=3)
            ax.add_patch(beam)

        # large-scale beam size
        for ax in [ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:    
            if scale == 'angular':
                beam_size = float(large_res[:-2])
                beam = Circle((115-beam_size/2,-115+beam_size/2), beam_size/2, color='k', fill=1, zorder=3)
            elif scale == 'physical':
                beam_size = float(large_res[:-2])*1e-3  # in kpc
                beam = Circle((9.3-beam_size/2,-9.5+beam_size/2), beam_size/2, color='k', fill=1, zorder=3)
            ax.add_patch(beam)

        # grid
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
            ax.grid(color='grey', linestyle='--', linewidth=1, zorder=1, alpha=.5)

        # save diagnostic plots
        if savediag:
            savename = diag_dir + '%s_cloud_%s_large_%s.png' % (glxy, cloud_res, large_res)
            plt.savefig(savename)

        # show plot
        if showdiag:
            plt.show()
        else:
            plt.close()
        
        
    ####################################################
    # OUTPUT DICTIONARY
    ####################################################

    keys = ['delta_ra_cloud', 'delta_dec_cloud', 'rgal_pc_cloud', 'int_co21_cloud', 'int_co21_uc_cloud', 
            'Sigmol_cloud', 'Sigmol_uc_cloud', 'vdis_cloud', 'vdis_uc_cloud', 'avir_cloud', 'avir_uc_cloud', 'Ptur_cloud', 'Ptur_uc_cloud',
            'delta_ra', 'delta_dec', 'rgal_pc', 'int_co21', 'int_co21_uc', 'int_densegas', 'int_densegas_uc', 'SigSFR', 'SigSFR_uc', 'Sigmol', 'Sigmol_uc',
            'int_co21_avg', 'int_co21_avg_uc', 'Sigmol_avg', 'Sigmol_avg_uc', 'vdis_avg', 'vdis_avg_uc', 'avir_avg', 'avir_avg_uc', 'Ptur_avg', 'Ptur_avg_uc',
            'PDE_avg','Sigstar', 'Sigstar_uc', 'Sigatom', 'Wmol_avg', 'Wmol_self_avg', 'PDE']

    output = dict.fromkeys(keys)
    
    # cloud-scale data
    output['delta_ra_cloud'] = delta_ra_cloud
    output['delta_dec_cloud'] = delta_dec_cloud
    output['rgal_pc_cloud'] = rgal_pc_cloud
    output['int_co21_cloud'] = int_co21_cloud
    output['int_co21_uc_cloud'] = int_co21_uc_cloud
    output['Sigmol_cloud'] = alpha_co21 * int_co21_cloud
    output['Sigmol_uc_cloud'] = alpha_co21 * int_co21_uc_cloud
    output['vdis_cloud'] = veldisp_cloud
    output['vdis_uc_cloud'] = veldisp_uc_cloud
    output['avir_cloud'] = alphavir_cloud
    output['avir_uc_cloud'] = alphavir_uc_cloud
    output['Ptur_cloud'] = Pturb_cloud
    output['Ptur_uc_cloud'] = Pturb_uc_cloud
    
    # large-scale data
    output['delta_ra'] = delta_ra_large
    output['delta_dec'] = delta_dec_large
    output['rgal_pc'] = rgal_pc_large
    output['int_co21'] = int_co21_large
    output['int_co21_uc'] = int_co21_uc_large
    output['Sigmol'] = alpha_co21 * int_co21_large * np.cos(np.deg2rad(incl))
    output['Sigmol_uc'] = alpha_co21 * int_co21_uc_large * np.cos(np.deg2rad(incl))
    output['int_densegas'] = int_densegas_large_list
    output['int_densegas_uc'] = int_densegas_uc_large_list
    output['SigSFR'] = SigSFR_large
    output['SigSFR_uc'] = SigSFR_uc_large
    output['int_co21_avg'] = int_co21_avg
    output['int_co21_avg_uc'] = int_co21_avg_uc
    output['Sigmol_avg'] = alpha_co21 * int_co21_avg
    output['Sigmol_avg_uc'] = alpha_co21 * int_co21_avg_uc
    output['vdis_avg'] = veldisp_avg
    output['vdis_avg_uc'] = veldisp_avg_uc
    output['avir_avg'] = alphavir_avg
    output['avir_avg_uc'] = alphavir_avg_uc
    output['Ptur_avg'] = Pturb_avg
    output['Ptur_avg_uc'] = Pturb_avg_uc
    if Sigstar_key & HI_key:
        output['PDE_avg'] = PDE_avg
        output['Wmol_avg'] = Wmol_avg
        output['Wmol_self_avg'] = Wmol_self_avg
        output['PDE'] = PDE_large
    if Sigstar_key:
        output['Sigstar'] = Sigstar_large
        output['Sigstar_uc'] = Sigstar_uc_large
    if HI_key:
        output['Sigatom'] = Sigatom_large

    return output



def get_overlap_ind(x1, y1, x2, y2, rounding=0.1, tolerance=5):
    """
    Takes two maps and returns the overlap.
    Rounding and tolerance should be in arcsec.
    For faster computing (x1, y1) should be the coarser grid.
    
    Input:
    - x1, y1, x2, y2: list (respectively)
    
    Output:
    - mask_overlap: list
    """
    
    # make array to store overlap indices
    ind_overlap = []

    ######################################
    # Constant Right Ascension
    ######################################

    # get data with constant x1 and indices where y1 is inside (x2,y2)
    for i in range(len(x1)):

        # get mask for constant x1 within tolerance
        x1_const = np.abs(x1 - x1[i]) < rounding

        # get corresponding (i.e. within tolerance) y2 data
        y2_ref = y2[np.abs(x2 - np.mean(x1[x1_const])) < tolerance]
        
        if len(y2_ref) == 0:
            continue
        
        # take data inside (x2,y2) grid, i.e. inside y2-range
        ind_inside = (y1[x1_const] > (np.nanmin(y2_ref)-tolerance)) & (y1[x1_const] < (np.nanmax(y2_ref)+tolerance))
        
        # get indices of constant x1
        ind_const = np.where(x1_const)[0]
        
        # get indices (w.r.t. x1 and y1) inside (x2,y2)
        ind_inside_all = ind_const[ind_inside]
        
        # concatenate lists into overlap indices list
        ind_overlap = np.concatenate((ind_overlap, ind_inside_all))
    
    
    ######################################
    # Constant Declination
    ######################################

    # get data with constant y1 and indices where x1 is inside (x2,y2)
    for i in range(len(y1)):

        # get mask for constant y1 within tolerance
        y1_const = np.abs(y1 - y1[i]) < rounding

        # get corresponding (i.e. within tolerance) x2 data
        x2_ref = x2[np.abs(y2 - np.mean(y1[y1_const])) < tolerance]
        
        if len(x2_ref) == 0:
            continue
        
        # take data inside (x2,y2) grid, i.e. inside x2-range
        ind_inside = (x1[y1_const] > (np.nanmin(x2_ref)-tolerance)) & (x1[y1_const] < (np.nanmax(x2_ref)+tolerance))
        
        # get indices of constant y1
        ind_const = np.where(y1_const)[0]
        
        # get indices (w.r.t. x1 and y1) inside (x2,y2)
        ind_inside_all = ind_const[ind_inside]
        
        # concatenate lists into overlap indices list
        ind_overlap = np.concatenate((ind_overlap, ind_inside_all))

        
    # convert to integers
    ind_overlap = [int(x) for x in list(ind_overlap)]

    # make mask such that overlap is 1 and rest 0
    mask_overlap = np.zeros(np.shape(x1))
    mask_overlap[ind_overlap] = 1
    
    return mask_overlap
