import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import linmix

def fit_OLS_bisector(x, y):
    """
    Takes x and y data, determines the OLS bisector fit and returns the fit values
    and it's uncertainties. In addition it returns the scatter of the data relative
    to the line fit (i.e. the standard deviation of the residuals).
    """

    try:
        alpha, beta, avar, bvar = leastsq(x, y, method=3)
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    # compute standard deviation of residuals
    # remark: it does not matter if the residuals are computed in x or y, because at the end
    # std(y_residuals) = std(x_residuals) * slope
    y_residuals = [yi - (alpha + beta * xi) for xi, yi in zip(x, y)]
    y_scatter = np.std(y_residuals)
    
    x_residuals = [xi - (-alpha/beta + 1/beta * yi) for xi, yi in zip(x, y)]
    x_scatter = np.std(x_residuals)
    
    if (x_scatter == 0) & (y_scatter == 0):
        scatter = 0
    else:
        scatter = x_scatter*y_scatter/np.sqrt(x_scatter**2 + y_scatter**2)
    
    return alpha, beta, avar, bvar, x_scatter, y_scatter, scatter



def fit_linmix(x_det, y_det, x_det_err=None, y_det_err=None, x_cens=None, y_cens=None, x_cens_err=None, y_cens_err=None,
               xycov_det=None, xycov_cens=None, method=None, x_cre=None, K=3, nchains=4, seed=7, MC_iter=10000,
               silent=True, debug=False, data_output=False, fitcens=True):
    """
    Takes x and y data with x and y errors and computes a linear regression
    based on a hierarchical Bayesian approach (linmix module). It returns
    the median fit values from the MCMC chain and it's standard deviation. 
    Also the credibility areas and various statistics are returned.
    
    It is assumed that the input data is already converted to log-log-space
    and uncertainties are given in dex.
    """
    # fit censored data?
    if not fitcens:
        x_cens = None
        y_cens = None
        x_cens_err = None
        y_cens_err = None
        xycov_cens = None
        method = None
    
    # make censored data compatible for linmix function
    if method is not None:

        # concatenate data
        x = np.concatenate((x_det, x_cens))
        y = np.concatenate((y_det, y_cens))


        # concatenate errors
        if x_det_err is not None and x_cens_err is not None:
            x_err = np.concatenate((x_det_err, x_cens_err))
        elif x_det_err is not None:
            x_err = x_det_err
        elif x_cens_err is not None:
            x_err = x_cens_err
        else:
            x_err = None

        if y_det_err is not None and y_cens_err is not None:
            y_err = np.concatenate((y_det_err, y_cens_err))
        elif y_det_err is not None:
            y_err = y_det_err
        elif y_cens_err is not None:
            y_err = y_cens_err
        else:
            y_err = None
            
        # concatenate covariance in errors
        if xycov_det is not None and xycov_cens is not None:
            xycov = np.concatenate((xycov_det, xycov_cens))
        elif x_det_err is not None:
            xycov = xycov_det
        elif x_cens_err is not None:
            xycov = xycov_cens
        else:
            xycov = None
            
        # censored data are upper limits
        if method == 'upplim':
            pass
            
        # censored data are lower limits
        if method == 'lowlim':
            # invert y-axis to convert lower to upper limits
            y = -y
            y_err = -y_err
            
        # delta is one for detections and zero for non-detections
        delta = np.concatenate((np.ones_like(x_det), np.zeros_like(x_cens)))
        
    else:
        x = x_det
        y = y_det
        x_err = x_det_err
        y_err = y_det_err
        delta = None
        xycov = None

    ## save linmix input data to file
    if data_output:
        linmix_data_array = np.column_stack((x, y, x_err, y_err, delta))
        np.save('/home/lneumann/Documents/University/MA_Astrophysics/Master_Thesis/Products/Misc/fdense_vs_ICOavg_centers_linmix_format.npy', linmix_data_array)
    ##
        
    # run linmix    
    print('[INFO] Initialise LinMix fitting routine.')
    lm = linmix.LinMix(x, y, xsig=x_err, ysig=y_err, xycov=xycov, delta=delta, 
                       K=K, nchains=nchains, parallelize=True, seed=seed) 
    #lm = linmix.LinMix(x_det, y_det, xsig=None, ysig=None, xycov=None, delta=None, 
    #                   K=K, nchains=nchains, parallelize=True, seed=seed) 

    print('[INFO] Run LinMix MCMC.')
    try:
        lm.run_mcmc(silent=silent, miniter=MC_iter//2, maxiter=MC_iter)

    except:
        print('[ERROR] LinMix did not convergence!')
        return None
    
    print('[INFO] LinMix finished. Number of MCMC iterations at convergence:', len(lm.chain))
    

    if method == 'lowlim':
        lm.chain['beta'] = -lm.chain['beta']
        lm.chain['alpha'] = -lm.chain['alpha']
        lm.chain['corr'] = -lm.chain['corr']
    
        
    # scatter of the residuals about the linear regression lines
    scatter_list = []
    for alpha, beta in zip(lm.chain['alpha'], lm.chain['beta']):  
        # get residuals w.r.t. to y-axis
        y_residuals = np.array([yi - (alpha + beta * xi) for xi, yi in zip(x_det, y_det)])
        y_scatter = np.sqrt(sum(y_residuals**2)/len(y_residuals))
        
        # get residuals w.r.t. to x-axis
        x_residuals = np.array([xi - (-alpha/beta + 1/beta * yi) for xi, yi in zip(x_det, y_det)])
        x_scatter = np.sqrt(sum(x_residuals**2)/len(x_residuals))
        
        # get standard deviation of orthogonal residuals, i.e. scatter
        if (x_scatter == 0) & (y_scatter == 0):
            scatter_xy = 0
        else:
            scatter_xy = x_scatter*y_scatter/np.sqrt(x_scatter**2 + y_scatter**2)    
        scatter_list.append(scatter_xy)
    
    # get median parameters of linear regression function
    slope = np.nanmedian(lm.chain['beta'])
    intercept = np.nanmedian(lm.chain['alpha'])
    corr = np.nanmedian(lm.chain['corr'])
    
    # get scatter as std in the residuals about the (median) fit line
    # get residuals w.r.t. to y-axis
    y_residuals = np.array([yi - (intercept + slope * xi) for xi, yi in zip(x_det, y_det)])
    y_scatter = np.sqrt(sum(y_residuals**2)/len(y_residuals))
    
    # get residuals w.r.t. to x-axis
    x_residuals = np.array([xi - (-intercept/slope + 1/slope * yi) for xi, yi in zip(x_det, y_det)])
    x_scatter = np.sqrt(sum(x_residuals**2)/len(x_residuals))
    
    # get standard deviation of orthogonal residuals, i.e. scatter
    if (x_scatter == 0) & (y_scatter == 0):
        scatter_residuals = 0
    else:
        scatter_residuals = x_scatter*y_scatter/np.sqrt(x_scatter**2 + y_scatter**2)       
    
    # sigmas of credibility intervals
    sigma_1 = 68.27
    sigma_2 = 95.45
    sigma_3 = 99.73
    
    # take uncertainties of parameters as 1-sigma credibility intervals
    slope_sig1_n = np.percentile(lm.chain['beta'], (100 - sigma_1)/2) # value at lower interval edge
    slope_sig1_p = np.percentile(lm.chain['beta'], (100 + sigma_1)/2)  # value at higher interval edge
    slope_unc_n = slope_sig1_n - slope  # 1-sigma uncertainty in negative direction
    slope_unc_p = slope_sig1_p - slope # 1-sigma uncertainty in positive direction
    slope_unc = (slope_unc_p, slope_unc_n)  # tuple with negative and positive 1-sigma uncertainties
    
    intercept_sig1_n = np.percentile(lm.chain['alpha'], (100 - sigma_1)/2)  # value at lower interval edge
    intercept_sig1_p = np.percentile(lm.chain['alpha'], (100 + sigma_1)/2)  # value at higher interval edge
    intercept_unc_n = intercept_sig1_n - intercept # 1-sigma uncertainty in negative direction
    intercept_unc_p = intercept_sig1_p - intercept # 1-sigma uncertainty in positive direction
    intercept_unc = (intercept_unc_p, intercept_unc_n)  # tuple with negative and positive 1-sigma uncertainties
    
    corr_sig1_n = np.percentile(lm.chain['corr'], (100 - sigma_1)/2)  # value at lower interval edge
    corr_sig1_p = np.percentile(lm.chain['corr'], (100 + sigma_1)/2)  # value at higher interval edge
    corr_unc_n = corr_sig1_n - corr  # 1-sigma uncertainty in negative direction
    corr_unc_p = corr_sig1_p - corr  # 1-sigma uncertainty in positive direction
    corr_unc = (corr_unc_p, corr_unc_n)  # tuple with negative and positive 1-sigma uncertainties
    
    # make dictionary for statistics
    keys = ['p_value', 'scatter_residuals_x', 'scatter_residuals_y', 'scatter_residuals_ortho', '-1sigma', '+1sigma', '-2sigma', '+2sigma', '-3sigma', '+3sigma']
    stats = dict.fromkeys(keys)
    
    # orthogonal scatter in the residuals
    stats['scatter_residuals_x'] = x_scatter
    stats['scatter_residuals_y'] = y_scatter
    stats['scatter_residuals_ortho'] = scatter_residuals
    
    # compute p-value from Pearson correlation
    n = len(x)
    dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
    p_value = 2*dist.cdf(-abs(corr))
    stats['p_value'] = p_value
    
    # intrinsic scatter of the linear regression
    scatter_intrinsic = np.nanmedian(lm.chain['sigsqr'])
    scatter_intrinsic_unc_n = np.percentile(lm.chain['sigsqr'], (100 + sigma_1)/2) - np.nanmedian(lm.chain['sigsqr'])
    scatter_intrinsic_unc_p = np.percentile(lm.chain['sigsqr'], (100 - sigma_1)/2) - np.nanmedian(lm.chain['sigsqr'])
    scatter_intrinsic_unc = (scatter_intrinsic_unc_n, scatter_intrinsic_unc_p)
    
    # get credibility areas
    if x_cre is not None:
        
        # make list for y-axis values evaluated for different chain elements but constant x_cre
        y_list = np.zeros(len(lm.chain))
        
        # make lists for sigma intervals as a function of x_cre
        y_1sig_p = np.zeros_like(x_cre)
        y_1sig_n = np.zeros_like(x_cre)
        y_2sig_p = np.zeros_like(x_cre)
        y_2sig_n = np.zeros_like(x_cre)
        y_3sig_p = np.zeros_like(x_cre)
        y_3sig_n = np.zeros_like(x_cre)
        
        # loop over credibility x axis values
        for k in range(len(x_cre)):
            
            # loop over MCMC chain and compute y values of fit line
            for i in range(len(lm.chain)):
                y_list[i] = lm.chain[i]['alpha'] + x_cre[k] * lm.chain[i]['beta']
                
            # get percentiles of 1-, 2-, 3-sigma levels
            y_1sig_n[k] = np.percentile(y_list, (100 - sigma_1)/2)
            y_1sig_p[k] = np.percentile(y_list, (100 + sigma_1)/2)
            y_2sig_n[k] = np.percentile(y_list, (100 - sigma_2)/2)
            y_2sig_p[k] = np.percentile(y_list, (100 + sigma_2)/2)
            y_3sig_n[k] = np.percentile(y_list, (100 - sigma_3)/2)
            y_3sig_p[k] = np.percentile(y_list, (100 + sigma_3)/2)
            
        # assign credibility intervals
        stats['-1sigma'] = y_1sig_n
        stats['+1sigma'] = y_1sig_p
        stats['-2sigma'] = y_2sig_n
        stats['+2sigma'] = y_2sig_p
        stats['-3sigma'] = y_3sig_n
        stats['+3sigma'] = y_3sig_p
        
    if debug:
        # make figure
        fig = plt.figure(figsize=(12,5))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # make plotting grid
        gs = gridspec.GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[:, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[1, 3])
        
        # textbox
        textbox = dict(boxstyle='round', facecolor='wheat', linewidth=0.5, alpha=1)  # set box

        #####################################
        # subplot 1 - data and fit
        #####################################
        ax1.scatter(x_det, y_det, color='k', zorder=6)
        ax1.errorbar(x_det, y_det, yerr=y_det_err, color='k', ls='none', zorder=5)
        if x_cens is not None and y_cens is not None:
            if method == 'upplim':
                ax1.errorbar(x_cens, y_cens, yerr=0.1, uplims=True, linestyle='none', color='grey', zorder=5)
            elif method == 'lowlim':
                ax1.errorbar(x_cens, y_cens, yerr=0.1, lolims=True, linestyle='none', color='grey', zorder=5)
        # regression line of median parameters
        if x_cre is not None:
            xs = np.copy(x_cre)
        elif x_cens is not None:
            xs = np.linspace(min(np.nanmin(x_det),np.nanmin(x_cens))-1, max(np.nanmax(x_det),np.nanmax(x_cens))+1, 10)
        else:
            xs = np.linspace(np.nanmin(x_det)-1, np.nanmax(x_det)+1, 10)
        ys_median = intercept + xs * slope
        ax1.plot(xs, ys_median, color='k', lw=3, zorder=5)
        # credibility intervals
        if x_cre is not None:
            ax1.plot(x_cre, y_1sig_n, color='k', ls='dashed', zorder=4)
            ax1.plot(x_cre, y_1sig_p, color='k', ls='dashed', zorder=4)
            ax1.plot(x_cre, y_2sig_n, color='k', ls='dotted', zorder=4)
            ax1.plot(x_cre, y_2sig_p, color='k', ls='dotted', zorder=4)
            ax1.plot(x_cre, y_3sig_n, color='k', ls='dashdot', zorder=4)
            ax1.plot(x_cre, y_3sig_p, color='k', ls='dashdot', zorder=4)
            ax1.fill_between(x_cre, y_1sig_n, y_1sig_p, color='tab:orange', alpha=0.5, zorder=3)
            ax1.fill_between(x_cre, y_2sig_n, y_1sig_n, color='tab:green', alpha=0.5, zorder=3)
            ax1.fill_between(x_cre, y_1sig_p, y_2sig_p, color='tab:green', alpha=0.5, zorder=3)
            ax1.fill_between(x_cre, y_3sig_n, y_2sig_n, color='tab:purple', alpha=0.5, zorder=3)
            ax1.fill_between(x_cre, y_2sig_p, y_3sig_p, color='tab:purple', alpha=0.5, zorder=3)
            
        # intrinsic scatter about the median fit line
        #y_scatter = 1/np.sqrt(1+slope**2) * scatter
        y_scatter = scatter_intrinsic
        y_sca_n = intercept + xs * slope - y_scatter
        y_sca_p = intercept + xs * slope + y_scatter
        ax1.fill_between(xs, y_sca_n, y_sca_p, color='grey', alpha=0.5, zorder=2)
        
        # axis
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        #####################################
        # subplot 2 - Slope
        #####################################
        n,_,_ = ax2.hist(lm.chain['beta'], bins=40, color="grey", edgecolor='k', alpha=.5)
        # median line
        ax2.axvline(slope, color='k', lw=3)
        # 1-sigma credibility interval
        sig_p = str(np.round(np.percentile(lm.chain['beta'], (1+sigma_1)/2*len(lm.chain['beta'])/100) - slope, 3))
        sig_n = str(np.round(slope - np.percentile(lm.chain['beta'], (1-sigma_1)/2*len(lm.chain['beta'])/100), 3))
        text = str(np.round(slope, 3)) + r'$^{+%s}_{-%s}$' % (sig_p, sig_n)
        ax2.text(np.nanmedian(lm.chain['beta']), 0.3*max(n), text,c='k', bbox=textbox, ha='center')
        # 1-sigma interval
        ax2.axvline(np.percentile(lm.chain['beta'], (100 - sigma_1)/2), c='tab:orange', lw=2)
        ax2.axvline(np.percentile(lm.chain['beta'], (100 + sigma_1)/2), c='tab:orange', lw=2)
        # 2-sigma interval
        ax2.axvline(np.percentile(lm.chain['beta'], (100 - sigma_2)/2), c='tab:green', lw=2)
        ax2.axvline(np.percentile(lm.chain['beta'], (100 + sigma_2)/2), c='tab:green', lw=2)
        # 3-sigma interval
        ax2.axvline(np.percentile(lm.chain['beta'], (100 - sigma_3)/2), c='tab:purple', lw=2)
        ax2.axvline(np.percentile(lm.chain['beta'], (100 + sigma_3)/2), c='tab:purple', lw=2)
        ax2.set_xlabel('Slope')
        ax2.set_ylabel('counts')
        
        #####################################
        # subplot 3 - Intercept
        #####################################
        n,_,_ = ax3.hist(lm.chain['alpha'], bins=40, color="grey", edgecolor='k', alpha=.5)
        # median line
        ax3.axvline(intercept, color='k', lw=3)
        # 1-sigma credibility interval
        sig_p = str(np.round(np.percentile(lm.chain['alpha'], (1+sigma_1)/2*len(lm.chain['alpha'])/100) - intercept, 3))
        sig_n = str(np.round(intercept - np.percentile(lm.chain['alpha'], (1-sigma_1)/2*len(lm.chain['alpha'])/100), 3))
        text = str(np.round(intercept, 3)) + r'$^{+%s}_{-%s}$' % (sig_p, sig_n)
        ax3.text(np.nanmedian(lm.chain['alpha']), 0.3*max(n), text,c='k', bbox=textbox, ha='center')

        # 1-sigma interval
        ax3.axvline(np.percentile(lm.chain['alpha'], (100 - sigma_1)/2), c='tab:orange', lw=2)
        ax3.axvline(np.percentile(lm.chain['alpha'], (100 + sigma_1)/2), c='tab:orange', lw=2)
        # 2-sigma interval
        ax3.axvline(np.percentile(lm.chain['alpha'], (100 - sigma_2)/2), c='tab:green', lw=2)
        ax3.axvline(np.percentile(lm.chain['alpha'], (100 + sigma_2)/2), c='tab:green', lw=2)
        # 3-sigma interval
        ax3.axvline(np.percentile(lm.chain['alpha'], (100 - sigma_3)/2), c='tab:purple', lw=2)
        ax3.axvline(np.percentile(lm.chain['alpha'], (100 + sigma_3)/2), c='tab:purple', lw=2)
        ax3.set_xlabel('Intercept')
        ax3.set_ylabel('counts')
        
        #####################################
        # subplot 4 - Corr. coeff.
        #####################################
        n,_,_ = ax4.hist(lm.chain['corr'], bins=40, color="grey", edgecolor='k', alpha=.5)
        # textbox
        textbox = dict(boxstyle='round', facecolor='wheat', linewidth=0.5, alpha=1)  # set box
        # median line
        ax4.axvline(corr, color='k', lw=3)
        # 1-sigma credibility interval
        sig_p = str(np.round(np.percentile(lm.chain['corr'], (1+sigma_1)/2*len(lm.chain['corr'])/100) - corr, 3))
        sig_n = str(np.round(corr - np.percentile(lm.chain['corr'], (1-sigma_1)/2*len(lm.chain['corr'])/100), 3))
        text = str(np.round(corr, 3)) + r'$^{+%s}_{-%s}$' % (sig_p, sig_n)
        ax4.text(np.nanmedian(lm.chain['corr']), 0.3*max(n), text,c='k', bbox=textbox, ha='center')

        # 1-sigma interval
        ax4.axvline(np.percentile(lm.chain['corr'], (100 - sigma_1)/2), c='tab:orange', lw=2)
        ax4.axvline(np.percentile(lm.chain['corr'], (100 + sigma_1)/2), c='tab:orange', lw=2)
        # 2-sigma interval
        ax4.axvline(np.percentile(lm.chain['corr'], (100 - sigma_2)/2), c='tab:green', lw=2)
        ax4.axvline(np.percentile(lm.chain['corr'], (100 + sigma_2)/2), c='tab:green', lw=2)
        # 3-sigma interval
        ax4.axvline(np.percentile(lm.chain['corr'], (100 - sigma_3)/2), c='tab:purple', lw=2)
        ax4.axvline(np.percentile(lm.chain['corr'], (100 + sigma_3)/2), c='tab:purple', lw=2)
        ax4.set_xlabel(r'Correlation $\rho$')
        ax4.set_ylabel('counts')  
        
        #####################################
        # subplot 5 - Intrinsic Scatter
        #####################################
        n,_,_ = ax5.hist(lm.chain['sigsqr'], bins=40, color="grey", edgecolor='k', alpha=.5)
        # textbox
        textbox = dict(boxstyle='round', facecolor='wheat', linewidth=0.5, alpha=1)  # set box
        # median line
        ax5.axvline(scatter_intrinsic, color='k', lw=3)
        # 1-sigma credibility interval
        sig_p = str(np.round(np.percentile(lm.chain['sigsqr'], (1+sigma_1)/2*len(lm.chain['sigsqr'])/100) - scatter_intrinsic, 3))
        sig_n = str(np.round(scatter_intrinsic - np.percentile(lm.chain['sigsqr'], (1-sigma_1)/2*len(lm.chain['sigsqr'])/100), 3))
        text = str(np.round(scatter_intrinsic, 3)) + r'$^{+%s}_{-%s}$' % (sig_p, sig_n)
        ax5.text(np.nanmedian(lm.chain['sigsqr']), 0.3*max(n), text,c='k', bbox=textbox, ha='center')

        # 1-sigma interval
        ax5.axvline(np.percentile(lm.chain['sigsqr'], (100 - sigma_1)/2), c='tab:orange', lw=2)
        ax5.axvline(np.percentile(lm.chain['sigsqr'], (100 + sigma_1)/2), c='tab:orange', lw=2)
        # 2-sigma interval
        ax5.axvline(np.percentile(lm.chain['sigsqr'], (100 - sigma_2)/2), c='tab:green', lw=2)
        ax5.axvline(np.percentile(lm.chain['sigsqr'], (100 + sigma_2)/2), c='tab:green', lw=2)
        # 3-sigma interval
        ax5.axvline(np.percentile(lm.chain['sigsqr'], (100 - sigma_3)/2), c='tab:purple', lw=2)
        ax5.axvline(np.percentile(lm.chain['sigsqr'], (100 + sigma_3)/2), c='tab:purple', lw=2)
        ax5.set_xlabel('Scatter')
        ax5.set_ylabel('counts') 
        
        plt.show()
        
    return slope, slope_unc, intercept, intercept_unc, corr, corr_unc, scatter_intrinsic, scatter_intrinsic_unc, stats


# get fit parameter results from linmix (add-on to fit_linmix function)
def get_linmix_results(x, y, x_err, y_err, y_lim, x2log=True, y2log=True, x_off=0,
                       limits='upplim', MC_iter=10000, fitcens=True, SNR_threshold=3, 
                       diagnostics=False):
    """
    Takes x and y data, uncertainties and upper or lower limits, converts to log-scale 
    (if x2log, y2log set to True) and returns LinMix linear regression results as dictionary.
    """
    # compute SNR
    x_snr = x/x_err
    y_snr = y/y_err
    
    # mask for detections
    mask_det = y_snr >= SNR_threshold
    
    # convert data to log-scale
    if x2log:
        x = np.log10(x)
        x_err = np.abs(np.log10(1 + 1/x_snr)) # downward uncertainty (asymmetric uncertainties)
    if y2log:
        y = np.log10(y)
        y_err = np.abs(np.log10(1 + 1/y_snr)) # downward uncertainty (asymmetric uncertainties)
        y_lim = np.log10(y_lim)
    
    # get detected data
    x_det = x[mask_det]
    x_det_err = x_err[mask_det]
    y_det = y[mask_det]
    y_det_err = y_err[mask_det]

    # remove nan values
    nonnan_det = (~np.isnan(x_det) & ~np.isnan(y_det))
    x_det = x_det[nonnan_det]
    y_det = y_det[nonnan_det]
    x_det_err = x_det_err[nonnan_det]
    y_det_err = y_det_err[nonnan_det]
    
    # get censored data
    if limits == 'upplim':
        x_cens = x[~mask_det]
        y_cens = y_lim[~mask_det]
    elif limits == 'lowlim':
        x_cens = x[~mask_det]
        y_cens = y_lim[~mask_det]   
    x_cens_err = np.full_like(x_cens, np.log10(1+1/3)) # 1-sigma uncertainty relative to the upper limit
    y_cens_err = np.full_like(y_cens, np.log10(1+1/3)) # 1-sigma uncertainty relative to the upper limit
    
    # remove nan values
    nonnan_cens = (~np.isnan(x_cens) & ~np.isnan(y_cens) & ~np.isinf(x_cens) & ~np.isinf(y_cens))
    x_cens = x_cens[nonnan_cens]
    y_cens = y_cens[nonnan_cens]
    x_cens_err = x_cens_err[nonnan_cens]
    y_cens_err = y_cens_err[nonnan_cens]  

    # plot data
    if diagnostics:
        plt.errorbar(x_det, y_det, xerr=x_det_err, yerr=y_det_err, ls='none', color='k', marker='o', 
                     elinewidth=1, capthick=1)
        if limits == 'upplim':
            plt.errorbar(x_cens, y_cens, uplims=True, yerr=0.1, ls='none', color='grey', elinewidth=1, capsize=3, capthick=1)
        elif limits == 'lowlim':
            plt.errorbar(x_cens, y_cens, lolims=True, yerr=0.1, ls='none', color='grey', elinewidth=1, capsize=3, capthick=1)
        plt.show()
    
    # linmix fit
    out_linmix = fit_linmix(x_det-x_off, y_det, x_det_err=x_det_err, y_det_err=y_det_err, 
                            x_cens=x_cens-x_off, y_cens=y_cens, x_cens_err=x_cens_err, y_cens_err=y_cens_err, 
                            method=limits, x_cre=None, K=3, seed=7, MC_iter=MC_iter, debug=False, 
                            fitcens=fitcens)

    slope, slope_unc, intercept, intercept_unc, corr, _, scatter_intrinsic, _, stats = out_linmix
    
    # plot data and fit
    if diagnostics:
        plt.errorbar(x_det, y_det, xerr=x_det_err, yerr=y_det_err, ls='none', color='k', marker='o', 
                     elinewidth=1, capthick=1)
        if limits == 'upplim':
            plt.errorbar(x_cens, y_cens, uplims=True, yerr=0.1, ls='none', color='grey', elinewidth=1, capsize=3, capthick=1)
        elif limits == 'lowlim':
            plt.errorbar(x_cens, y_cens, lolims=True, yerr=0.1, ls='none', color='grey', elinewidth=1, capsize=3, capthick=1)
        # best fit line
        x_fit = np.linspace(np.nanmin(np.concatenate((x_det, x_cens))), np.nanmax(np.concatenate((x_det, x_cens))), 10)
        y_fit = intercept + (x_fit-x_off)*slope
        print(slope, intercept, x_off)
        plt.plot(x_fit, y_fit, color='r', lw=2)
        # scatter
        y_scatter_dw = y_fit - stats['scatter_residuals_y']
        y_scatter_up = y_fit + stats['scatter_residuals_y']
        plt.fill_between(x_fit, y_scatter_dw, y_scatter_up, color='r', alpha=0.3)
            
        plt.show()
    
    # create dictionary
    df = pd.DataFrame()
    df['slope'] = [slope]
    df['slope unc. lower'] = [slope_unc[0]]
    df['slope unc. upper'] = [slope_unc[1]]
    df['intercept'] = [intercept]
    df['intercept unc. lower'] = [intercept_unc[0]]
    df['intercept unc. upper'] = [intercept_unc[1]]
    df['x-axis offset'] = [x_off]
    df['pearson corr.'] = [corr]
    df['p-value.'] = [stats['p_value']]
    df['scatter (intrinsic)'] = [scatter_intrinsic]
    df['scatter (residuals)'] = [stats['scatter_residuals_y']]
    
    return df


def load_linmix_fit(fname, xlim, x_fit, xlog=True, ylog=True):

    # load fit results
    df = pd.read_csv(fname)
    intercept = df['intercept'][0]
    intercept_err_p = df['intercept unc. upper'][0]
    intercept_err_n = df['intercept unc. lower'][0]
    intercept_err = (abs(intercept_err_p) + abs(intercept_err_n)) / 2
    slope = df['slope'][0]
    slope_err_p = df['slope unc. upper'][0]
    slope_err_n = df['slope unc. lower'][0]
    slope_err = (abs(slope_err_p) + abs(slope_err_n)) / 2
    x_off = df['x-axis offset'][0]
    y_scatter = df['scatter (residuals)'][0]
    
    # linmix median line fit
    if ylog and xlog:
        y_fit = 10**(intercept + slope * (np.log10(x_fit)-x_off))
    elif ylog:
        y_fit = 10**(intercept + slope * (x_fit-x_off))
    elif xlog:
        y_fit = intercept + slope * (np.log10(x_fit)-x_off)
    else:
        y_fit = intercept + slope * (x_fit-x_off)

    # simulate Gaussian distribution
    n_draws = 10000
    slope_list = np.random.normal(loc=slope, scale=slope_err, size=n_draws)
    intercept_list = np.random.normal(loc=intercept, scale=intercept_err, size=n_draws)
    
    # create arrays for y-axis values of 1-sigma scatter
    y_1sig_n, y_1sig_p = np.ones_like(x_fit)*np.nan, np.ones_like(x_fit)*np.nan
    y_3sig_n, y_3sig_p = np.ones_like(x_fit)*np.nan, np.ones_like(x_fit)*np.nan
        
    # loop over x-axis values
    for i in range(len(x_fit)):
    
        # make list to store distribution of fit lines
        y_list = []
    
        # loop over realisations and compute distribution of y-axis values at x-axis value of index i
        for intercept, slope in zip(intercept_list, slope_list):
            if ylog and xlog:
                y_list.append(10**(intercept + slope * (np.log10(x_fit[i])-x_off)))
            elif ylog:
                y_list.append(10**(intercept + slope * (x_fit[i]-x_off)))
            elif xlog:
                y_list.append(intercept + slope * (np.log10(x_fit[i])-x_off))
            else:
                y_list.append(intercept + slope * (x_fit[i]-x_off))
        
        # get percentiles of 1-sigma levels
        y_1sig_n[i] = np.percentile(y_list, 16)
        y_1sig_p[i] = np.percentile(y_list, 84)
        y_3sig_n[i] = np.percentile(y_list, 0.135)
        y_3sig_p[i] = np.percentile(y_list, 99.865)

    # scatter about fit line (residuals)
    if ylog and xlog:
        y_sct_up = 10**(intercept + slope * (np.log10(x_fit)-x_off) + y_scatter)
        y_sct_dw = 10**(intercept + slope * (np.log10(x_fit)-x_off) - y_scatter)
    elif ylog:
        y_sct_up = 10**(intercept + slope * (x_fit-x_off) + y_scatter)
        y_sct_dw = 10**(intercept + slope * (x_fit-x_off) - y_scatter)
    elif xlog:
        y_sct_up = intercept + slope * (np.log10(x_fit)-x_off) + y_scatter
        y_sct_dw = intercept + slope * (np.log10(x_fit)-x_off) - y_scatter
    else:
        y_sct_up = intercept + slope * (x_fit-x_off) + y_scatter
        y_sct_dw = intercept + slope * (x_fit-x_off) - y_scatter

    return y_fit, y_1sig_n, y_1sig_p, y_3sig_n, y_3sig_p, y_sct_dw, y_sct_up, y_scatter