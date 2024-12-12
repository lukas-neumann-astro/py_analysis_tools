import numpy as np
from astropy import units as u
from astropy import constants as c


###############################
# GAS MASS
###############################

def get_NH2_from_Sigmagas(Sigma_gas):
    """
    Takes surface density of the gas and returns the molecular hydrogen number 
    column density assuming all hydrogen is molecular.
    
    Input:
    - Sigma_gas : float or np.array; gas surface density [g cm-2]
    
    Output:
    - NH2       : float or np.array; H2 column density [cm-2]
    """
    
    # constants
    amu = 1.66053906660e-24  # atomic mass units [g]
    mH = 1.0079 * amu  # atomic hydrogen mass [g]
    mu = 2.8  # mean particle weight per H2 (Kauffmann+08)
    
    # conversion
    NH2 = Sigma_gas / (mu * mH)  # H2 column density [cm-2]
    
    # return H2 column density
    return NH2



def get_atomic_gas_surface_density(W_21cm, W_21cm_err):
    """
    Compute atomic gas surface density following Walter+08
    
    Input:
    - W_21cm  : 21-cm line integrated intensity [K km/s]
    
    Output: 
    - Sd_atom : atomic gas surface density [Msun/pc^2]
    """

    # compute atomic gas surface density
    Sd_atom = 1.97e-2 * W_21cm * u.Msun/u.pc**2  # (Walter+08)
    Sd_atom_err = 1.97e-2 * W_21cm_err * u.Msun/u.pc**2  # (Walter+08)
    
    # convert to common units
    Sd_atom_value = Sd_atom.value
    Sd_atom_err_value = Sd_atom_err.value
    
    return Sd_atom_value, Sd_atom_err_value


def get_dense_gas_surface_density(W_hcn, W_hcn_err, aHCN=10):
    """
    Compute dense gas surface density, e.g. like Gao&Solomon04 or others, but
    allowing more general, i.e. varying, HCN-to-dense gas conversion factor
    
    Input:
    - W_hcn  : HCN(1-0) line integrated intensity [K km/s]
    - aHCN   : HCN-to-dense gas mass conversion factor [Msun/pc^2/(K km/s)]
    
    Output: 
    - Sd_dense : dense gas surface density [Msun/pc^2]
    """

    # compute dense gas surface density
    Sd_dense = aHCN * W_hcn * u.Msun/u.pc**2 
    Sd_dense_err = aHCN * W_hcn_err * u.Msun/u.pc**2
    
    # convert to common units
    Sd_dense_value = Sd_dense.value
    Sd_dense_err_value = Sd_dense_err.value
    
    return Sd_dense_value, Sd_dense_err_value


def get_aCO_teng2023(sigma_mol):
    """
    Compute CO-to-H2 conversion factor using the conversion from Teng et al. 2023 (Equ. 2),
    which is based on the cloud-scale molecular gas (CO) velocity dispersion.
    
    Input:
    - sigma_mol  : molecular gas velocity dispersion [km/s]
    
    Output: 
    - alpha_co : CO(1-0)-to-H2 conversion factor [Msun/pc^2/(K.km/s)]
    """

    # compute conversion factor
    alpha_co = 10**(-0.81 * np.log10(sigma_mol) + 1.05)
    
    return alpha_co



def get_aHCN_based_on_aCO(alpha_co, alpha_co_cal=4.35, alpha_hcn_cal=15):
    """
    Compute alpha_HCN conversion factor following variations of alpha_CO.
    
    Input:
    - alpha_co : CO(1-0)-to-H2 conversion factor [Msun/pc^2/(K.km/s)]
    - alpha_co_cal : CO(1-0)-to-H2 conversion factor calibration reference [Msun/pc^2/(K.km/s)]
    - alpha_hcn_cal : HCN(1-0)-to-Mdense conversion factor calibration reference [Msun/pc^2/(K.km/s)]
    
    Output: 
    - alpha_hcn : HCN(1-0)-to-Mdense conversion factor [Msun/pc^2/(K.km/s)]
    """

    # compute conversion factor
    alpha_hcn = alpha_co * alpha_hcn_cal/alpha_co_cal
    
    return alpha_hcn


def get_aHCN_bemis2024(W_hcn):
    """
    Get alpha_HCN conversion factor from Bemis 2024, Equ. 19.
    
    Input:
    - W_hcn  : HCN(1-0) line integrated intensity [K km/s]
    
    Output: 
    - alpha_hcn : HCN(1-0)-to-Mdense conversion factor [Msun/pc^2/(K.km/s)]
    """

    # compute conversion factor
    alpha_hcn = 10**(-0.55*np.log10(W_hcn) + 2.55)
    
    return alpha_hcn


###############################
# PRESSURE
###############################

def get_ism_pressure(W_co21, W_co21_err, W_21cm, W_21cm_err, Sd_star, Sd_star_err, l_star, 
                     sigma_gas_z=15, aCO=4.35, R21=0.65):
    """
    Compute dynamical equilibirium pressure following Sun+20
    
    Input:
    - W_co21  : CO(2-1) integrated intensity [K km/s]
    - W_21cm  : 21-cm line integrated intensity [K km/s]
    - Sd_star : stellar mass surface density [Msun/pc2]
    - l_star  : stellar disc scale length [kpc]
    - sigma_gas_z : vertical gas velocity dispersion [km/s]
    - aCO     : CO-H2 conversion factor [Msun/pc^2/(K km/s)]
    - R21     : CO(2-1)/CO(1-0) line ratio
    
    Output: 
    - pressure [kB K cm-3]
    """
    
    # assign units to input values
    Sd_star *= u.Msun/u.pc**2
    Sd_star_err *= u.Msun/u.pc**2
    l_star *= u.kpc
    sigma_gas_z *= u.km/u.s 

    # compute atomic gas surface density
    Sd_atom = 1.97e-2 * W_21cm * u.Msun/u.pc**2  # (Walter+08)
    Sd_atom_err = 1.97e-2 * W_21cm_err * u.Msun/u.pc**2  # (Walter+08)

    # compute molecular gas surface density
    Sd_mol = aCO / R21 * W_co21 * u.Msun/u.pc**2
    Sd_mol_err = aCO / R21 * W_co21_err * u.Msun/u.pc**2

    # compute total gas surface density
    Sd_gas = Sd_atom + Sd_mol
    Sd_gas_err = np.sqrt(Sd_atom_err**2 + Sd_mol_err**2)

    # compute stellar mass volume density
    rho_star = Sd_star / (0.55 * l_star)
    rho_star_err = Sd_star_err / (0.55 * l_star)

    # remove zeros and negatives
    rho_star[rho_star <= 0] = np.nan
    rho_star_err[rho_star <= 0] = np.nan

    # compute ISM (dynamical equilibrium) pressure
    P_DE = np.pi*c.G/2 * Sd_gas**2 + Sd_gas * np.sqrt(2*c.G * rho_star) * sigma_gas_z
    P_DE_err = np.sqrt( ((np.pi*c.G*Sd_gas + np.sqrt(2*c.G*rho_star)*sigma_gas_z)*Sd_gas_err)**2 
                     + ((Sd_gas/np.sqrt(2*c.G*rho_star)*c.G*sigma_gas_z)*rho_star_err)**2 )

    # convert to common units
    P_DE_value = (P_DE / c.k_B).cgs.value
    P_DE_err_value = (P_DE_err / c.k_B).cgs.value
    
    return P_DE_value, P_DE_err_value
    

###############################
# MOLECULAR CLOUD PROPERTIES
###############################

# MOLECULAR GAS SURFACE DENSITY
def get_cloud_surface_density(W_co21, W_co21_err, aCO=4.35, R21=0.65):
    """
    Takes the CO(2-1) integrated intensity and computes the molecular gas 
    surface density given the CO(2-1)-to-CO(1-0) and CO-H2 conversion factors.
    
    Input:
    - W_co21    : CO(2-1) line intensity [K km/s]
    - W_co21_err : uncertainty  [K km/s]
    - aCO:      : CO-H2 conversion factor [Msun/pc^2/(K km/s)]
    - R21:      : CO(2-1)/(1-0) line ratio
    
    Output:
    - Sd_mol    : molecular gas surface density [Msun/pc2]
    - Sd_mol_err : propagated uncertainty [Msun/pc2]
    """
    Sd_mol = aCO / R21 * W_co21
    Sd_mol_err = aCO / R21 * W_co21_err
    
    return Sd_mol, Sd_mol_err
    
    
# VELOCITY DISPERSION
def get_cloud_velocity_dispersion(W_co21, W_co21_err, Tpeak_co21, Tpeak_co21_err, v_ch, r_ch=0):
    """
    Takes the CO(2-1) integrated intensity and computes the molecular gas 
    surface density given the CO(2-1)-to-CO(1-0) and CO-H2 conversion factors.
    
    Input:
    - W_co21        : CO(2-1) line intensity [K km/s]
    - W_co21_err     : uncertainty  [K km/s]
    - Tpeak_co21    : CO(2-1) peak brightness temperature [K]
    - Tpeak_co21_err : uncertainty  [K]
    - v_ch          : channel width [km/s]
    - r_ch          : channel-to-channel correaltion coefficient
    
    Output:
    - vdis    : velocity dispersion of the molecular gas [km/s]
    - vdis_err : propagated uncertainty [km/s]
    """
    
    # in Tpeak change zeros to nan values
    Tpeak_co21[Tpeak_co21 == 0] = np.nan

    # compute measured effective width (Heyer et al. 2001)
    sigma_measured = W_co21 / np.sqrt(2*np.pi) / Tpeak_co21  # [km/s]
    sigma_measured_err = W_co21_err / np.sqrt(2*np.pi) / Tpeak_co21  # Gaussian error propagation
    
    # coupling between adjacent channels (Leroy et al. 2016)
    k = 0.0 + 0.47*r_ch - 0.23*r_ch**2 - 0.16*r_ch**3 + 0.43*r_ch**4

    # compute spectral response curve width (Leroy et al. 2016)
    v_ch = np.abs(v_ch)  # take absolute value
    sigma_response = v_ch / np.sqrt(2) * (1.0 + 1.18*k + 10.4*k**2)  # [km/s]

    # change NaNs to zeros for valid comparison (next step)
    sigma_measured[np.isnan(sigma_measured) & np.isnan(sigma_measured_err)] = 0

    # change sigma_measured to inf values if sigma_measured < sigma_response
    id_invalid_sqrt = sigma_measured < sigma_response
    sigma_measured[id_invalid_sqrt] = np.nan
    sigma_measured_err[id_invalid_sqrt] = np.nan

    # compute velocity dispersion (Rosolowsky & Leroy 2006)
    vdis_mol = np.sqrt(sigma_measured**2 - sigma_response**2)
    vdis_mol_err = sigma_measured/vdis_mol * sigma_measured_err
    
    return vdis_mol, vdis_mol_err
    
        
# VIRIAL PARAMETER
def get_cloud_virial_parameter(Sd_mol, Sd_mol_err, vdis_mol, vdis_mol_err, D_beam):
    """
    Takes the surface density and velocity dispersion of the molecular gas as
    well as the beam size as cloud size and computes the virial parameter of
    the molecular cloud.
    
    Input:
    - Sd_mol    : molecular gas surface density [Msun/pc2]
    - Sd_mol_err : uncertainty [Msun/pc2]
    - vdis      : velocity dispersion of the molecular gas [km/s]
    - vdis_err   : uncertainty [km/s]
    - D_beam    : beam size (FWHM) [pc]
    
    Output:
    - alphavir    : virial paramter of the molecular cloud
    - alphavir_err : propagated uncertainty
    """
    
    # gravitational constant
    G = 4.30091e-3  # [pc M_Sun^(-1) (km/s)^2]

    # put zeros and negative values in Sigmol and veldisp to NaN to allow division
    Sd_mol[Sd_mol <= 0] = np.nan
    vdis_mol[vdis_mol <= 0] = np.nan
    
    # beam radius from diameter
    R_beam_pc = D_beam / 2  # [pc]

    # geometry factor for a density profile proportional to 1/r
    f = 10/9
    
    # compute the virial parameter (Sun et al. 2018)
    alphavir = 5*np.log(2)/np.pi/f/G * vdis_mol**2 / Sd_mol / R_beam_pc  # [dimensionless]
    alphavir_err = alphavir * np.sqrt(4*(vdis_mol_err/vdis_mol)**2 + (Sd_mol_err/Sd_mol)**2)  # Gaussian error propagation

    return alphavir, alphavir_err


# (INTERNAL) TURBULENT PRESSURE
def get_cloud_turbulent_pressure(Sd_mol, Sd_mol_err, vdis_mol, vdis_mol_err, D_beam):
    """
    Takes the surface density and velocity dispersion of the molecular gas as
    well as the beam size as cloud size and computes the turbulent pressure of
    the molecular cloud.
    
    Input:
    - Sd_mol    : molecular gas surface density [Msun/pc2]
    - Sd_mol_err : uncertainty [Msun/pc2]
    - vdis      : velocity dispersion of the molecular gas [km/s]
    - vdis_err   : uncertainty [km/s]
    - D_beam    : beam size (FWHM) [pc]
    
    Output:
    - Pturb    : internal turbulent pressure of the molecular cloud [kB K cm-3]
    - Pturb_err : propagated uncertainty [km/s]
    """
        
    # beam radius from diameter
    R_beam_pc = D_beam / 2

    # compute internal turbulent pressure
    Pturb = 61.3 * Sd_mol * vdis_mol**2 / (R_beam_pc/40)  # [K cm^(-3) kB]

    # compute uncertainty
    Pturb_err = 61.3 * np.sqrt((vdis_mol**2/(R_beam_pc/40)*Sd_mol_err)**2 \
                                    + (2*Sd_mol*vdis_mol/(R_beam_pc/40)*vdis_mol_err)**2)
    
    return Pturb, Pturb_err


###############################
# STAR FORMATION RATE
###############################

# SFR from JWST 21 micron band
def get_sfr_f2100w_leroy23(W_f2100w, W_f2100w_err, mode='linear'):
    """
    Takes the mid-infrared intensity and returns the sfr following two recipes
    from Leroy+23. The 'linear' method assumes a linear relation between 21 micron
    and SFR that is anchored to K&E12. The 'powerlaw' method takes the empirical
    relation between Halpha and MIR from Leroy+23 and assumes the Calzetti+07
    conversion to go from Halpha to SFR.
    
    Input:
    - W_f2100w    : JWST F2100W intensity [MJy/sr]
    - W_f2100w_err : uncertainty [MJy/sr]
    - mode        : can be 'linear' or 'powerlaw'
    
    Output:
    - Sd_sfr      : SFR surface density [Msun/yr/kpc2]
    - Sd_sfr_err   : uncertainty [Msun/yr/kpc2]
    """
    
    # conversion factors: 21 micron (JWST) over 24 micron (MIPS24) intensity
    R_X_24 = 0.80  # 0.11 dex scatter

    if mode == 'linear':
        
        # MIPS 24 micron conversion factor
        C_24 = 10**(-42.7)

        # compute star formation rate surface density (Equation 5)
        # W_f2100w in MJy/sr
        # Sd_sfr in Msun/yr/kpc2
        Sd_sfr = 2.97e-3 * 1/R_X_24 * C_24*10**42.7 * W_f2100w
        Sd_sfr_err =  2.97e-3 * 1/R_X_24 * C_24*10**42.7 * W_f2100w_err
        
    elif mode == 'powerlaw':
        
        # power law (Halpha vs F2100W) fit (in log-log space) parameters
        #b = -5.35  # all galaxies (Leroy+23)
        #m = 1.29   # all galaxies (Leroy+23)
        b = -5.22436   # intercept (NGC 4321 fit)
        m = 1.298408   # slope (NGC 4321 fit)
        
        # convert MIR into Halpha intensity via power law
        W_Halpha = 10**b * W_f2100w**m
        W_Halpha_err = 10**b * W_f2100w_err**m
        
        # Halpha to SFR conversion (Calzetti+07)
        c_Ha = 5.3e-42  # (Msun/yr)/(erg/s)
        
        # convert factor: cm2 to kpc2
        kpc_to_cm = u.kpc.to(u.cm)
        
        # conversion factor to sr-1
        per_sr = 4*np.pi # full sphere covers 4 pi steradians
        
        # compute star formation rate surface density (Equation 3)
        # W_Halpha in erg/s/cm2/sr
        # Sd_sfr in Msun/yr/kpc2
        #Sd_sfr = 642 * W_Halpha
        #Sd_sfr_err =  624 * W_Halpha_err
        Sd_sfr = c_Ha * kpc_to_cm**2 * per_sr * W_Halpha
        Sd_sfr_err =  c_Ha * kpc_to_cm**2 * per_sr * W_Halpha_err
        
    return Sd_sfr, Sd_sfr_err


# SFR from free-free intensity (e.g. 33 GHz)
def get_sfr_free_free(I_ff_MJysr, I_ff_MJysr_err, freq_GHz=33, Te_K=1e4, alpha_thermal=0.1, frac_thermal=0.76):
    """
    Takes .. to SFR.
    
    Input:
    - I_ff     : free-free intensity [MJy/sr]
    - I_ff_err : uncertainty [MJy/sr]
    
    Output:
    - Sd_sfr      : SFR surface density [Msun/yr/kpc2]
    - Sd_sfr_err   : uncertainty [Msun/yr/kpc2]
    """

    # assign units to quantities
    I_ff = I_ff_MJysr * u.MJy * 4*np.pi # 4pi to eliminate steradian
    I_ff_err = I_ff_MJysr_err * u.MJy * 4*np.pi # 4pi to eliminate steradian
    Te = Te_K * u.K
    freq = freq_GHz * u.GHz

    kpc_to_cm = u.kpc.to(u.cm)
    Jy_to_cgs = 1e-23

    # compute SFR from free-free emission [Murphy+2012, Equation (6)]
    Sd_sfr = 4.6e-28 * (Te/(1e4*u.K))**(-0.45) * (freq/u.GHz)**alpha_thermal * frac_thermal *  I_ff/u.Jy * Jy_to_cgs * kpc_to_cm**2  # Msun/yr/kpc2
    Sd_sfr_err = 4.6e-28 * (Te/(1e4*u.K))**(-0.45) * (freq/u.GHz)**alpha_thermal * frac_thermal *  I_ff_err/u.Jy * Jy_to_cgs * kpc_to_cm**2  # Msun/yr/kpc2

    print( 4.6e-28 * kpc_to_cm**2 * 4*np.pi)
    print((freq/u.GHz)**alpha_thermal)
    
    # get value
    Sd_sfr = Sd_sfr.decompose().value
    Sd_sfr_err = Sd_sfr_err.decompose().value
        
    return Sd_sfr, Sd_sfr_err


###############################
# LINE RATIOS
###############################


# CO(2-1)/CO(1-0) calibration from Leroy+22
def get_R21_from_global_SFR_and_Mstar(sfr, mstar):
    """
    Takes global star formation rate (sfr) and global stellar mass (mstar)
    and returns global CO(2-1)/CO(1-0) (R_21) line ratio.
    This calibration is meant to correct for galaxy-to-galaxy variations
    in the excitation conditions.
    
    Input:
    - sfr   : 
    - mstar : 
    
    Output:
    - R_21  : CO(2-1)/CO(1-0) line ratio [dimensionless]
    """
    
    # calibration from Leroy+22, table 4, first row
    x_low = -10.78
    x_high = -9.02
    y_low = -0.3
    m = (0-y_low) / (x_high-x_low)

    # apply calibration
    x = np.log10(sfr/mstar)
    if x <= x_low:
        R_21_log = y_low
    elif (x > x_low) & (x <= x_high):
        R_21_log = y_low + m*(x - x_low)
    else:
        R_21_log = 0

    # convert to linear scale
    R_21 = 10**R_21_log
    
    return R_21


# CO(2-1)/CO(1-0) calibration from Schinnerer & Leroy+24 (Tab. 1)
def get_R21_from_SFR(Sd_sfr):
    """
    Takes (resolved) star formation rate surface density and returns
    (resolved) CO(2-1)/CO(1-0) (R_21) line ratio.
    Can take multi-dim. arrays.
    
    Input:
    - Sd_sfr : star formation rate surface density [Msun/yr/kpc2]
    
    Output:
    - R_21   : CO(2-1)/CO(1-0) line ratio [dimensionless]
    """

    # set negatives to 0
    sfr = np.copy(Sd_sfr)
    sfr[sfr<0] = 0
    
    # apply scaling relation from table 1
    R_21 = 0.65 * (sfr/1.8e-2)**0.125


    # apply limits (table 1)
    R_21[R_21<0.35] = 0.35
    R_21[R_21>1] = 1
    
    return R_21
