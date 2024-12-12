import numpy as np
import scipy.integrate as integrate # for PDF integration

def pn_sigma(Mach):
    """
    Returns standard deviation of the PDF for input (3D) Mach number;
    based on Padoan 2014 (review); w/o magnetised turbulence
    """
    return np.sqrt(np.log(1 + 0.25 * Mach**2))

def km_sigma(Mach, beta, b):
    """
    Standard deviation of the log-normal PDF
    """
    return np.sqrt( np.log( 1 + b**2 * Mach**2 * beta / (1 + beta) ) )

def km_nSF(n_mean, Mach, avir=1.3, phi=1.12, p=0.5):
    """
    Critical overdensity for collapse
    """
    return np.pi**2/15 * phi**2 * avir * Mach**(2/p - 2) * n_mean

def PDF_volume(s, sigma):
    """
    Volume-weighted probability distribution function (PDF):
    Returns Gaussian function at input variable s and standard deviation sigma;
    becomes log-normal for volume density n with s=ln(n/n_mean), where n_mean
    is the mean volumen density.
    """
    # mean of the Gaussian
    s0 = -0.5 * sigma**2
    
    # Gausssian
    return (2*np.pi * sigma**2)**(-0.5) * np.exp(-0.5 * (s - s0)**2 / sigma**2)

def PDF_mass(s, sigma):
    """
    Mass-weighted probability distribution function (PDF):
    Returns Gaussian function at input variable s and standard deviation sigma;
    becomes log-normal for volume density n with s=ln(n/n_mean), where n_mean
    is the mean volumen density.
    """
    # mean of the Gaussian
    s0 = -0.5 * sigma**2

    # Gausssian
    return (2*np.pi * sigma**2)**(-0.5) * np.exp(-0.5 * (s - s0)**2 / sigma**2) * np.exp(s)

## Define functions to compute the area under the PDF and thus HCN/CO and SFR/HCN.

# define function to integrate the PDF
def PDF_integral(n_mean=1e3, Mach=15, n_min=0, n_max=np.inf, weight='mass'):
    """
    Integrates the PDF given the parameters (mean density, Mach number) of 
    the PDF and (optionally) the density range.
    
    Input:
    - n_mean : mean number density [cm-3]
    - Mach   : Mach number
    - n_min  : lower limit of integration [cm-3]
    - n_max  : upper limit of integration [cm-3]
    - weight : PDF weight, e.g. 'mass' or 'none'
    
    Output:
    - area enclosed by the PDF and the density limits
    """
    # convert limits from number density to s=ln(n/n_mean)
    s_min = np.log(n_min/n_mean)
    s_max = np.log(n_max/n_mean)
    
    # get sigma from Mach number
    sigma = pn_sigma(Mach)
    
    # integration
    if weight == 'volume':
        # by volume
        A = integrate.quad(PDF_volume, s_min, s_max, args=(sigma))[0]
    elif weight == 'mass':
        # by mass
        A = integrate.quad(PDF_mass, s_min, s_max, args=(sigma))[0]
    
    # return area
    return A

# define functions to compute HCN/CO and SFR/HCN using PDF integration
def fdense_from_PDF(n_mean=1e3, Mach=15, n_CO_lim=(5e2, 5e3), n_HCN_lim=(6e4, 6e5)):
    """
    Takes PDF parameters, and density ranges for CO and HCN
    and returns the ratio of the integrated areas under the PDF.
    
    Input:
    - n_mean    : mean number density [cm-3]
    - Mach      : Mach number
    - n_CO_lim  : density range (tuple) traced by CO [cm-3]
    - n_HCN_lim : density range (tuple) traced by CO [cm-3]
    
    Output:
    - HCN/CO PDF area ratio
    """
    # compute area under the PDF traced by CO
    CO_area = PDF_integral(n_mean=n_mean, Mach=Mach, n_min=n_CO_lim[0], n_max=n_CO_lim[1], weight='mass')
    
    # compute area under the PDF traced by HCN
    HCN_area = PDF_integral(n_mean=n_mean, Mach=Mach, n_min=n_HCN_lim[0], n_max=n_HCN_lim[1], weight='mass')
    
    # return HCN/CO ratio
    return HCN_area / CO_area

def SFEdense_from_PDF(n_mean=1e3, Mach=15, n_HCN_lim=(6e4, 6e5), avir=1.3):
    """
    Takes PDF parameters, and density ranges for HCN
    and returns the ratio of SFR and HCN areas.
    
    Input:
    - n_mean    : mean number density [cm-3]
    - Mach      : Mach number
    - n_HCN_lim : density range (tuple) traced by CO [cm-3]
    
    Output:
    - SFR/HCN PDF area ratio
    """
    
    # compute density threshold for star formation
    n_SF = km_nSF(n_mean, Mach, avir=avir)
    
    # compute area under the PFD above n_SF
    SFR_area = PDF_integral(n_mean=n_mean, Mach=Mach, n_min=n_SF, n_max=1e99, weight='mass')  # mass which forms stars
    SFR_model = SFR_area * np.sqrt(n_mean)  # accounts for density dependence on free-fall time
    #SFR_model = SFR_area * 3.25e-2 * np.sqrt(n_mean)  # accounts for density dependence on free-fall time
    # factor 3.25e-2 accounts for the prefactor when going from sqrt(1/t_ff) to sqrt(n_0)
    #SFR_model = SFR_area  # old version (wrong!)
    
    # compute area under the PDF traced by HCN
    HCN_area = PDF_integral(n_mean=n_mean, Mach=Mach, n_min=n_HCN_lim[0], n_max=n_HCN_lim[1], weight='mass')
    
    # return SFR/HCN ratio
    return SFR_model / HCN_area