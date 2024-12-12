# working directory
wd = '/vol/alcina/data1/lneumann/phd/'

# Library Paths
import sys
sys.path.insert(0, wd + 'scripts/programs/')  # import modules from other directory
sys.path.insert(0, wd + 'scripts/PyStructure_v2/scripts/')

# Modules
import numpy as np
import pandas as pd
from weighted_averages_analysis import get_densegas_vs_cloud_data

#---------------------------
# Input directories
#---------------------------

# data directory
data_dir = wd + 'products/pystruct/dense_gas_vs_cloud_props/'  # pystructures
# directory to cloud-scale data masks
mask_dir = wd + 'products/pystruct/phangs_alma_masks/'
# path to distance table (from PHANGS)
table_path = wd + 'data/aca_paper/tables/phangs_sample_table_v1p6.csv'
# path to channel-to-channel correlation data
chcorr_path = wd + 'data/aca_paper/tables/channel_correlation_all_resolutions.csv'
# path to radial scale length of the stellar disk data (from S4G)
Rstar_path = wd + 'data/aca_paper/tables/stellardisc_sizescale.csv'


#---------------------------
# Output directories
#---------------------------

# directory to save data in table format
results_dir = wd + 'products/aca_paper/dense_gas_vs_cloud_props/'
# directory to save diagnostic plots
diag_dir = wd + 'products/aca_paper/diagnostics/pipeline_weighted_averages/'

#---------------------------
# Program settings
#---------------------------

# Save data and/or diagnostic plots?
savedata = True
savediag = False
showdiag = False

# Apply edge masking?
mask = True

# Aperture averages (in addition to convolution-based averages)
aperture_avg = True

# Dense gas tracers
#densegas_tracer_list = ['HCN', 'HCOP']
densegas_tracer_list = ['HCN10', 'HCOP10', 'CS21']

# Resolution Configurations
#res_configs = ['natres', 'lowres', 'midres', 'highres']
res_configs = ['natres']


#---------------------------
# Program
#---------------------------

for res_config in res_configs:

    # galaxies
    glxys = ['ngc0628','ngc1097','ngc1365','ngc1385','ngc1511','ngc1546','ngc1566','ngc1672','ngc1792','ngc2566',
             'ngc2903','ngc2997','ngc3059','ngc3521','ngc3621','ngc4303','ngc4321','ngc4535','ngc4536','ngc4569',
             'ngc4826','ngc5248','ngc5643','ngc6300','ngc7496']

    # resolutions
    cloud_natres = ['1.1as', '1.6as', '1.3as', '1.2as', '1.4as', '1.2as', '1.2as', '1.9as', '1.9as', '1.2as',
                    '1.4as', '1.7as', '1.2as', '1.3as', '1.8as', '1.8as', '1.6as', '1.5as', '1.4as', '1.6as', 
                    '1.2as', '1.2as', '1.2as', '1.0as', '1.6as']
    cloud_highres = '75pc'
    cloud_midres = '120pc'
    cloud_lowres = '150pc'

    large_natres = ['18.6as', '19.4as', '20.6as', '19.9as', '17.6as', '18.9as', '19.7as', '18.2as', '18.7as', '18.5as',
                    '18.3as', '20.4as', '16.7as', '21.1as', '18.9as', '20.2as', '19.6as', '22.8as', '21.5as', '19.2as',
                    '18.7as', '19.9as', '18.0as', '17.7as', '17.9as']
    large_highres = '1000pc'
    large_midres = '1500pc'
    large_lowres = '2100pc'

#    glxys = ['ngc3621']
#    cloud_natres = '1.8as'
#    large_natres = '18.9as'

    midres_tag = [True, True, False, False, True, False, False, False, False, False,
                  True, True, False, True, True, False, True, False, False, False,
                  True, True, True, True, False]

    highres_tag = [True, False, False, False, False, False, False, False, False, False,
                   True, False, False, False, True, False, False, False, False, False,
                   True, False, False, True, False]
    
    # assign resolution according to configuration
    if res_config == 'natres':
        cloud_res = cloud_natres
        large_res = large_natres

    elif res_config == 'lowres':
        cloud_res = cloud_lowres
        large_res = large_lowres

    elif res_config == 'midres':
        cloud_res = cloud_midres
        large_res = large_midres
        glxys = [x for x, y in zip(glxys, midres_tag) if y]

    elif res_config == 'highres':
        cloud_res = cloud_highres
        large_res = large_highres
        glxys = [x for x, y in zip(glxys, highres_tag) if y]

    else:
        print('[ERROR] Resolution configuration not known!')

    # convert to list if string
    if type(cloud_res) == str:
        cloud_res_list = [cloud_res] * len(glxys)
    elif type(cloud_res) == list:
        cloud_res_list = cloud_res
    else: print('[ERROR]')
        
    if type(large_res) == str:
        large_res_list = [large_res] * len(glxys)
    elif type(large_res) == list:
        large_res_list = large_res
    else: print('[ERROR]')
        
        
    # iterate over galaxies
    for glxy, cloud_res, large_res in zip(glxys, cloud_res_list, large_res_list):
    
        # run data analysis
        data_dict = get_densegas_vs_cloud_data(data_dir = data_dir, 
                                               table_path = table_path, 
                                               chcorr_path = chcorr_path,
                                               glxy = glxy, 
                                               cloud_res = cloud_res, 
                                               large_res = large_res, 
                                               densegas_list = densegas_tracer_list, 
                                               mask_dir = mask_dir, 
                                               cloud_mask = mask, 
                                               avg_method = 'convolution',
                                               Rstar_path = Rstar_path,
                                               diag_dir = diag_dir, 
                                               savediag = savediag, 
                                               showdiag = showdiag)

        # get data from output directory
        delta_ra_cloud = data_dict['delta_ra_cloud']
        delta_dec_cloud = data_dict['delta_dec_cloud']
        rgal_pc_cloud = data_dict['rgal_pc_cloud']

        int_co21_cloud = data_dict['int_co21_cloud']
        int_co21_uc_cloud = data_dict['int_co21_uc_cloud']
        Sigmol_cloud = data_dict['Sigmol_cloud']
        Sigmol_uc_cloud = data_dict['Sigmol_uc_cloud']
        vdis_cloud = data_dict['vdis_cloud']
        vdis_uc_cloud = data_dict['vdis_uc_cloud']
        avir_cloud = data_dict['avir_cloud']
        avir_uc_cloud = data_dict['avir_uc_cloud']
        Ptur_cloud = data_dict['Ptur_cloud']
        Ptur_uc_cloud = data_dict['Ptur_uc_cloud']

        delta_ra = data_dict['delta_ra']
        delta_dec = data_dict['delta_dec']
        rgal_pc = data_dict['rgal_pc']

        int_co21 = data_dict['int_co21']
        int_co21_uc = data_dict['int_co21_uc']
        Sigmol = data_dict['Sigmol']
        Sigmol_uc = data_dict['Sigmol_uc']
        int_densegas = data_dict['int_densegas']
        int_densegas_uc = data_dict['int_densegas_uc']
        SigSFR = data_dict['SigSFR']
        SigSFR_uc = data_dict['SigSFR_uc']
        Sigstar = data_dict['Sigstar']
        Sigstar_uc = data_dict['Sigstar_uc']
        Sigatom = data_dict['Sigatom']

        co21_avg = data_dict['int_co21_avg']
        co21_avg_uc = data_dict['int_co21_avg_uc']
        Sigmol_avg = data_dict['Sigmol_avg']
        Sigmol_avg_uc = data_dict['Sigmol_avg_uc']
        vdis_avg = data_dict['vdis_avg']
        vdis_avg_uc = data_dict['vdis_avg_uc']
        avir_avg = data_dict['avir_avg']
        avir_avg_uc = data_dict['avir_avg_uc']
        Ptur_avg = data_dict['Ptur_avg']
        Ptur_avg_uc = data_dict['Ptur_avg_uc']
        PDE_avg = data_dict['PDE_avg']
        Wmol_self_avg = data_dict['Wmol_self_avg']         
        Wmol_avg = data_dict['Wmol_avg'] 
        PDE = data_dict['PDE']
        
        if aperture_avg:
            # re-run data analysis with aperture averaging method
            data_dict_aper = get_densegas_vs_cloud_data(data_dir = data_dir, 
                                                        table_path = table_path, 
                                                        chcorr_path = chcorr_path,
                                                        glxy = glxy, 
                                                        cloud_res = cloud_res, 
                                                        large_res = large_res, 
                                                        densegas_list = densegas_tracer_list, 
                                                        mask_dir = mask_dir, 
                                                        cloud_mask = mask, 
                                                        avg_method = 'aperture',
                                                        Rstar_path = Rstar_path,
                                                        diag_dir = diag_dir, 
                                                        savediag = savediag, 
                                                        showdiag = showdiag)

            # check if data is consistent with above method
            aper_arr = [data_dict_aper['delta_ra_cloud'], data_dict_aper['delta_dec_cloud'], data_dict_aper['rgal_pc_cloud'],
                        data_dict_aper['int_co21_cloud'], data_dict_aper['Sigmol_cloud'], data_dict_aper['vdis_cloud'],
                        data_dict_aper['avir_cloud'], data_dict_aper['Ptur_cloud'], 
                        data_dict_aper['delta_ra'], data_dict_aper['delta_dec'], data_dict_aper['rgal_pc'],
                        data_dict_aper['int_co21'], data_dict_aper['Sigmol'], data_dict_aper['int_densegas'][0],
                        data_dict_aper['SigSFR'], data_dict_aper['Sigstar'], data_dict_aper['Sigatom']]
            conv_arr = [delta_ra_cloud, delta_dec_cloud, rgal_pc_cloud, int_co21_cloud, Sigmol_cloud, vdis_cloud,
                        avir_cloud, Ptur_cloud, delta_ra, delta_dec, rgal_pc, int_co21, Sigmol, int_densegas[0],
                        SigSFR, Sigstar, Sigatom]

            for arr1, arr2 in zip(aper_arr, conv_arr):
                Error = False if np.array_equal(arr1 , arr2) else True

            if Error:
                print('[ERROR] Aperture average arrays not compatible!')
            
            # get aperture average data
            co21_aperavg = data_dict_aper['int_co21_avg']
            co21_aperavg_uc = data_dict_aper['int_co21_avg_uc']
            Sigmol_aperavg = data_dict_aper['Sigmol_avg']
            Sigmol_aperavg_uc = data_dict_aper['Sigmol_avg_uc']
            vdis_aperavg = data_dict_aper['vdis_avg']
            vdis_aperavg_uc = data_dict_aper['vdis_avg_uc']
            avir_aperavg = data_dict_aper['avir_avg']
            avir_aperavg_uc = data_dict_aper['avir_avg_uc']
            Ptur_aperavg = data_dict_aper['Ptur_avg']
            Ptur_aperavg_uc = data_dict_aper['Ptur_avg_uc']
            PDE_aperavg = data_dict_aper['PDE_avg']
            Wmol_self_aperavg = data_dict_aper['Wmol_self_avg']
            Wmol_aperavg = data_dict_aper['Wmol_avg'] 


        # Save cloud-scale and large-scale + weighted average data as .csv table (for every galaxy separately)
        if savedata:

            # cloud-scale
            savename_cloud = '%s_cloud_%s_large_%s_%s_cloud_scale.csv' % (glxy, cloud_res, large_res, res_config)    
            df_cloud = pd.DataFrame()
            df_cloud['Delta R.A.'] = delta_ra_cloud
            df_cloud['Delta Dec.'] = delta_dec_cloud
            df_cloud['Rgal'] = rgal_pc_cloud
            df_cloud['Int. CO21'] = int_co21_cloud
            df_cloud['Int. CO21 uc.'] = int_co21_uc_cloud
            df_cloud['Sigmol'] = Sigmol_cloud
            df_cloud['Sigmol uc.'] = Sigmol_uc_cloud
            df_cloud['vdis'] = vdis_cloud
            df_cloud['vdis uc.'] = vdis_uc_cloud
            df_cloud['avir'] = avir_cloud
            df_cloud['avir uc.'] = avir_uc_cloud
            df_cloud['Pturb'] = Ptur_cloud
            df_cloud['Pturb uc.'] = Ptur_uc_cloud
            df_cloud.to_csv(results_dir+savename_cloud, index=False)

            # large-scale
            savename_large = '%s_cloud_%s_large_%s_%s_large_scale.csv' % (glxy, cloud_res, large_res, res_config)
            df_large = pd.DataFrame()
            df_large['Delta R.A.'] = delta_ra
            df_large['Delta Dec.'] = delta_dec
            df_large['Rgal'] = rgal_pc
            df_large['Int. CO21'] = int_co21
            df_large['Int. CO21 uc.'] = int_co21_uc
            df_large['Sigmol'] = Sigmol
            df_large['Sigmol uc.'] = Sigmol_uc
            for densegas_tracer in densegas_tracer_list:
                id_dense = densegas_tracer_list.index(densegas_tracer)
                df_large['Int. '+densegas_tracer] = int_densegas[id_dense]
                df_large['Int. '+densegas_tracer+' uc.'] = int_densegas_uc[id_dense]
            df_large['Int. Avg. CO21'] = co21_avg
            df_large['Int. Avg. CO21 uc.'] = co21_avg_uc
            df_large['Int. Avg. Sigmol'] = Sigmol_avg
            df_large['Int. Avg. Sigmol uc.'] = Sigmol_avg_uc
            df_large['Int. Avg. vdis'] = vdis_avg
            df_large['Int. Avg. vdis uc.'] = vdis_avg_uc
            df_large['Int. Avg. avir'] = avir_avg
            df_large['Int. Avg. avir uc.'] = avir_avg_uc
            df_large['Int. Avg. Pturb'] = Ptur_avg
            df_large['Int. Avg. Pturb uc.'] = Ptur_avg_uc
            if aperture_avg:
                df_large['Aper. Int. Avg. CO21'] = co21_aperavg
                df_large['Aper. Int. Avg. CO21 uc.'] = co21_aperavg_uc
                df_large['Aper. Int. Avg. Sigmol'] = Sigmol_aperavg
                df_large['Aper. Int. Avg. Sigmol uc.'] = Sigmol_avg_uc
                df_large['Aper. Int. Avg. vdis'] = vdis_aperavg
                df_large['Aper. Int. Avg. vdis uc.'] = vdis_aperavg_uc
                df_large['Aper. Int. Avg. avir'] = avir_aperavg
                df_large['Aper. Int. Avg. avir uc.'] = avir_aperavg_uc
                df_large['Aper. Int. Avg. Pturb'] = Ptur_aperavg
                df_large['Aper. Int. Avg. Pturb uc.'] = Ptur_aperavg_uc
            df_large['Sigma SFR'] = SigSFR
            df_large['Sigma SFR uc.'] = SigSFR_uc
            if PDE_avg is not None:
                df_large['Int. Avg. PDE'] = PDE_avg
                df_large['Int. Avg. Wmol_self'] = Wmol_self_avg
                df_large['Int. Avg. Wmol'] = Wmol_avg
                if aperture_avg:
                    df_large['Aper. Int. Avg. PDE'] = PDE_aperavg
                    df_large['Aper. Int. Avg. Wmol_self'] = Wmol_self_aperavg
                    df_large['Aper. Int. Avg. Wmol'] = Wmol_aperavg
                df_large['PDE'] = PDE
            if Sigstar is not None:
                df_large['Sigma star'] = Sigstar
                df_large['Sigma star uc.'] = Sigstar_uc
            if Sigatom is not None:
                df_large['Sigma atom'] = Sigatom
            df_large.to_csv(results_dir+savename_large, index=False)
