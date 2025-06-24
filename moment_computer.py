import numpy as np
from astropy import units as u
# from astropy import constants as c

# MOMENT-0
def get_mom0(spec, vaxis, mask=None):
    
    """
    Compute moment-0 (integrated intensity) of a list of spectra.

    Parameters
    ----------
    cube : 2-dimensional numpy array, float
        Position-velocity (PV) array which contains the brightness
        temperature in units of K.
    vaxis : 1-dimensional numpy array, float
        Velocity axis of the PV array in units of m/s.
    mask : 2-dimensional numpy array, int, optional
        Velocity-integration mask matching the dimensions of the PV array,
        which is used to define the velocity range over which the moment
        is computed. 
    """

    # assign units to input quantities
    if hasattr(spec, 'unit'):
        spec = spec.to(u.K)
    else:
        spec *= u.K
    if hasattr(vaxis, 'unit'):
        vaxis = vaxis.to(u.km/u.s)
    else:
        vaxis *= u.km/u.s

    # velocity-integration mask
    if mask is None:
        no_mask = True
        # if no mask is provided, integrate over all velocities
        mask = np.ones_like(spec.value, dtype=int)
    else:
        no_mask = False

    # get channel width from velocity axis
    delta_v = np.abs(vaxis[1] - vaxis[0])

    # compute moment-0
    mom0 = np.nansum(spec * mask, axis=(spec.ndim-1)) * delta_v

    # compute uncertainty
    if no_mask:
        # compute rms over all channels (including potential emission)
        rms = np.nanstd(spec, axis=(spec.ndim-1))
        print('[Warning] RMS is computed over all channels and is potentially biased high!')
    else:
        # compute noise over emission-free channels
        spec_noise = np.copy(spec)  # copy spectrum
        spec_noise[mask==1] = np.nan  # clip emission
        rms = np.nanstd(spec_noise, axis=(spec.ndim-1)) # rms noise
    n_mask = np.nansum(mask, axis=(spec.ndim-1))  # number of channels inside integration window
    e_mom0 = rms * np.sqrt(n_mask) * delta_v  # moment-0 uncertainty

    # return moment-0 and its uncertainty
    return mom0, e_mom0


# MOMENT-1
def get_mom1(spec, vaxis, mask=None):
    
    """
    Compute moment-1 (central velocity) of a list of spectra.

    Parameters
    ----------
    cube : 2-dimensional numpy array, float
        Position-velocity (PV) array which contains the brightness
        temperature in units of K.
    vaxis : 1-dimensional numpy array, float
        Velocity axis of the PV array in units of m/s.
    mask : 2-dimensional numpy array, int, optional
        Velocity-integration mask matching the dimensions of the PV array,
        which is used to define the velocity range over which the moment
        is computed. 
    """

    # get moment-0
    mom0, e_mom0 = get_mom0(spec, vaxis, mask)

    # assign units to input quantities
    if hasattr(spec, 'unit'):
        spec = spec.to(u.K)
    else:
        spec *= u.K
    if hasattr(vaxis, 'unit'):
        vaxis = vaxis.to(u.km/u.s)
    else:
        vaxis *= u.km/u.s

    # velocity-integration mask
    if mask is None:
        no_mask = True
        # if no mask is provided, integrate over all velocities
        mask = np.ones_like(spec.value, dtype=int)
    else:
        no_mask = False

    # get channel width from velocity axis
    delta_v = np.abs(vaxis[1] - vaxis[0])

    # compute moment-1
    mom1 = 1/mom0 * np.nansum(spec * mask * vaxis, axis=(spec.ndim-1)) * delta_v

    # compute uncertainty
    if no_mask:
        # compute rms over all channels (including potential emission)
        rms = np.nanstd(spec, axis=(spec.ndim-1))
        print('[Warning] RMS is computed over all channels and is potentially biased high!')
    else:
        # compute noise over emission-free channels
        spec_noise = np.copy(spec)  # copy spectrum
        spec_noise[mask==1] = np.nan  # clip emission
        rms = np.nanstd(spec_noise, axis=(spec.ndim-1)) # rms noise
    sum_vaxis = np.nansum(vaxis * mask, axis=(spec.ndim-1))
    e_mom1 = 1/mom0 * np.sqrt((mom1 * e_mom0)**2 + (rms * sum_vaxis * delta_v)**2)

    # return moment-1 and its uncertainty
    return mom1, e_mom1


# MOMENT-2
def get_mom2(spec, vaxis, mask=None):
    
    """
    Compute moment-2 (velocity dispersion) of a list of spectra.
    Returns the FWHM of the dispersion (i.e. square root of moment-2).

    Parameters
    ----------
    cube : 2-dimensional numpy array, float
        Position-velocity (PV) array which contains the brightness
        temperature in units of K.
    vaxis : 1-dimensional numpy array, float
        Velocity axis of the PV array in units of m/s.
    mask : 2-dimensional numpy array, int, optional
        Velocity-integration mask matching the dimensions of the PV array,
        which is used to define the velocity range over which the moment
        is computed. 
    """

    # get moment-0
    mom0, e_mom0 = get_mom0(spec, vaxis, mask)

    # get moment-1
    mom1, e_mom1 = get_mom1(spec, vaxis, mask)

    # assign units to input quantities
    if hasattr(spec, 'unit'):
        spec = spec.to(u.K)
    else:
        spec *= u.K
    if hasattr(vaxis, 'unit'):
        vaxis = vaxis.to(u.km/u.s)
    else:
        vaxis *= u.km/u.s

    # velocity-integration mask
    if mask is None:
        no_mask = True
        # if no mask is provided, integrate over all velocities
        mask = np.ones_like(spec.value, dtype=int)
    else:
        no_mask = False

    # get channel width from velocity axis
    delta_v = np.abs(vaxis[1] - vaxis[0])

    # compute moment-2
    if mom1.ndim == 0:
        mom2 = np.sqrt(1/mom0 * np.nansum(spec * mask * (vaxis - mom1)**2, axis=(spec.ndim-1)) * delta_v)
    else:
        mom2 = np.sqrt(1/mom0 * np.nansum(spec * mask * (vaxis[np.newaxis, :] - mom1[:, np.newaxis])**2, axis=(spec.ndim-1)) * delta_v)

    # compute uncertainty
    if no_mask:
        # compute rms over all channels (including potential emission)
        rms = np.nanstd(spec, axis=(spec.ndim-1))
        print('[Warning] RMS is computed over all channels and is potentially biased high!')
    else:
        # compute noise over emission-free channels
        spec_noise = np.copy(spec)  # copy spectrum
        spec_noise[mask==1] = np.nan  # clip emission
        rms = np.nanstd(spec_noise, axis=(spec.ndim-1)) # rms noise

    if mom1.ndim == 0:
        e_mom2 = 1/(2*mom2*mom0) * np.sqrt((mom2**2/delta_v * e_mom0)**2 + (rms * np.nansum(mask * (vaxis - mom1)**2, axis=(spec.ndim-1)))**2 
                                        + (2 * e_mom1 * np.nansum(spec * mask * (vaxis - mom1), axis=(spec.ndim-1)))**2) * delta_v
    else:
        e_mom2 = 1/(2*mom2*mom0) * np.sqrt((mom2**2/delta_v * e_mom0)**2 + (rms * np.nansum(mask * (vaxis[np.newaxis,:] - mom1[:,np.newaxis])**2, axis=(spec.ndim-1)))**2 
                                        + (2 * e_mom1 * np.nansum(spec * mask * (vaxis[np.newaxis,:] - mom1[:,np.newaxis]), axis=(spec.ndim-1)))**2) * delta_v

    # return moment-2 and its uncertainty
    return mom2, e_mom2