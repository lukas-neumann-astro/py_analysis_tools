import numpy as np
from astropy.coordinates import SkyCoord, FK5
from astropy import units as u

def get_flat_data(data_list):
    """
    Takes a list of arrays and concatenates all arrays of the list,
    thus returns a single array.
    """
    data_array = []
    for x in data_list:
        # cast to array if float
        if len(x) == 0:
            continue
        if type(x) != np.ndarray:
            x = np.array([x])
        # concatenate to flat array
        data_array = np.concatenate((data_array, x))
        
    return data_array


def get_relative_coords(ra, dec, rgal):
    
    """
    Takes right ascension (in deg), declination (in deg) and
    galactocentric radius (in arbitrary unit) and returns right ascension
    and declination relative to the galactic center.
    """
    
    ind_center = np.where(rgal == 0)[0][0]  # get index of center
    coords_ref = [str(ra[ind_center]) + ' ' + str(dec[ind_center])]
    skycoords_ref = SkyCoord(coords_ref, frame=FK5, unit=(u.deg, u.deg))
    coords_map = (ra, dec) *u.deg
    skycoords_map = SkyCoord(ra=coords_map[0], dec=coords_map[1], frame=FK5)
    aframe = skycoords_ref.skyoffset_frame()
    delta_ra = skycoords_map.transform_to(aframe).lon.arcsec
    delta_dec = skycoords_map.transform_to(aframe).lat.arcsec
    
    return delta_ra, delta_dec


def get_relative_coords2(ra, dec, ra_center=287.6353625, dec_center=9.0934811):
    
    """
    Takes right ascension (in deg), declination (in deg) and
    center coordinates (in deg) and returns right ascension
    and declination relative to the center.
    default center is W49
    """
    
    ind_center = np.where( (ra == ra_center) & (dec == dec_center) )[0][0] # get index of center
    coords_ref = [str(ra[ind_center]) + ' ' + str(dec[ind_center])]
    skycoords_ref = SkyCoord(coords_ref, frame=FK5, unit=(u.deg, u.deg))
    coords_map = (ra, dec) *u.deg
    skycoords_map = SkyCoord(ra=coords_map[0], dec=coords_map[1], frame=FK5)
    aframe = skycoords_ref.skyoffset_frame()
    delta_ra = skycoords_map.transform_to(aframe).lon.arcsec
    delta_dec = skycoords_map.transform_to(aframe).lat.arcsec
    
    return delta_ra, delta_dec


def get_overlap(xdata, ydata, data, xnew, ynew, tol=0):
    """
    Takes data and corresponding coordinates (xdata, ydata), checks overlap with
    new coordinates (xnew, ynew) and returns data corresponding to new coordinates.
    """
    
    data_new = np.ones_like(xnew) * np.nan

    # loop over coordinates
    for i in range(len(xnew)):

        # get coordinates of reference pixel
        xref, yref = xnew[i], ynew[i]

        # loop over other map
        for j in range(len(xdata)):

            # get coordinates
            x, y = xdata[j], ydata[j]

            # check overlap with other map and assign value if overlap exists
            if (abs(x-xref)<tol) and (abs(y-yref)<tol):
                data_new[i] = data[j]
                
    return data_new