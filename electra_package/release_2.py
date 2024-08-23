import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from loguru import logger
import time
from electra_package.modules_preprocess import down_sample_lidar, scale_vertices,scale_conductor, rotate_vano, clean_outliers, clean_outliers_2
from electra_package.modules_utils import set_logger, unscale_fits, extract_vano_values
from electra_package.modules_main2 import analyze_backings, analyze_conductor_configuration, cluster_and_evaluate, extract_conductor_config, puntuate_and_save, fit_and_evaluate_conds, analyze_polilinia_values
from electra_package.modules_fits import stack_unrotate_fits
from electra_package.modules_plots import *
from electra_package.puntuacionparavano import puntuacion_apriori

def process_vano(vano):
    
    try:
            
        # Extract vano values: LIDAR points, extremos, polilinia....
        # Extract vano ID, vano length and save id
        idv, vano_length, cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(vano)
        del vano["LIDAR"]
        
        ###############################################
        
        logger.critical(f"\nReference {idv}")
        
        # Create new attributes for data element (dictionary)
        # Declare fit evaluation results as default
        response_vano = {}
        response_vano["ID_VANO"] = idv
        response_vano['CONDUCTORES_CORREGIDOS']={}
        response_vano['PARAMETROS(a,h,k)']={} #PARAMETROS(a,h,k)
        response_vano['FLAG'] = 'None'
        response_vano['NUM_CONDUCTORES'] = 0
        response_vano["NUM_CONDUCTORES_FIABLE"] = False
        response_vano["CONFIG_CONDUCTORES"] = "None"
        response_vano["COMPLETITUD"] = "None"
        response_vano["RECONSTRUCCION"] = ""
        response_vano["PUNTUACION_APRIORI"] = {"P_HUECO": -99, "DIFF2D": -99.0, "NOTA" : -99.0}
        response_vano["PUNTUACION_APOSTERIORI"] = {"P_VALUE": -99, "CORRELATION": -99.0, "RMSE" : -99.0, "NOTA" : -99.0}
        
        # response_vano["PORCENTAJE_HUECOS"] = 0
        # response_vano["ERROR_POLILINEA"] = 0
        # response_vano["ERROR_CATENARIA"] = 0
        
        ###############################################
        
        empty_pol_num, expected_conds = analyze_polilinia_values(vert_values, vano_length)
        
        logger.critical(f"Expected conductors from data: {expected_conds}")
        logger.critical(f"Number of empty lines from data: {empty_pol_num}")
        
        if expected_conds > 3:
            
            logger.error(f"3 conductors expected, found {expected_conds}")
            response_vano['FLAG'] = "bad_cond_number"
            
            return response_vano, -1
        
        elif expected_conds == 0:
            
            logger.warning("No polilineas...")
            response_vano['FLAG'] = "no_polilineas"
            
        elif empty_pol_num != 0:
            
            logger.warning(f"Empty polilinea = {empty_pol_num}")
            response_vano['FLAG'] = "found_empty_polis"
            
        ###############################################
        
        response_vano["PUNTUACION_APRIORI"] = puntuacion_apriori(cond_values, extremos_values, apoyo_values, vert_values, vano_length)
        
        logger.critical(f"Puntuaciones apriori: {response_vano['PUNTUACION_APRIORI']}")
        
        ###############################################

        logger.info(f"Downsampling LIDAR to 25%")
        apoyo_values, cond_values = down_sample_lidar(apoyo_values, cond_values)

        logger.info(f"Backings cloud shape: {apoyo_values.shape}, Conductor cloud shape: {cond_values.shape}")
        
        if cond_values.shape[1] < 100 or apoyo_values.shape[1] < 100:
        
            logger.error("Empty point cloud")
            response_vano['FLAG'] = 'empty_cloud'
            
            return response_vano, -1
        
        ###############################################
        
        extremos_values = analyze_backings(vano_length, apoyo_values, extremos_values)

        if extremos_values == -1:
        
            # Include flag of bad extreme values
            # Set the line value of this element as 0 ****
            logger.error(f"Bad backings")
            response_vano['FLAG'] = 'bad_backings'
            
            return response_vano, -1
        
        ###############################################
        
        mat, rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
        
        rotated_conds = clean_outliers(rotated_conds, rotated_extremos)
        rotated_conds = clean_outliers_2(rotated_conds)

        X_scaled,scaler_x,scaler_y,scaler_z = scale_conductor(rotated_conds)
        scaled_vertices = scale_vertices(rotated_vertices, scaler_x,scaler_y,scaler_z)
        
        ###############################################
        
        config, max_var = analyze_conductor_configuration(X_scaled)
        num_empty, finc, md = extract_conductor_config(X_scaled, rotated_extremos, rotated_conds)
        
        response_vano["CONFIG_CONDUCTORES"] = config
        response_vano["COMPLETITUD"] = finc
        response_vano["PORCENTAJE_HUECOS"] = + num_empty*10
        response_vano['NUM_CONDUCTORES'] = md
        
        if finc == "incomplete" or (md == 0 and finc != "full"):
            
            logger.error(f"Empty conductor")
            response_vano['FLAG'] = "empty_conductor"

            return response_vano, -1
            
                
        if config == 0:
            coord = 0
            
        elif config == 1:
            coord = 2
                    
        else:
            
            logger.error(f"Bad config")
            response_vano['FLAG'] = "bad_configuration" 
            return response_vano, -1
        
        ###############################################
        
        max_conds = 6
        good_clust = False
        n_conds = 3
        
        logger.info(f"Kmeans clustering for (config, max_conds, md, max_var): {config, max_conds, md, max_var}")
        
        for p in range(2):
            
            if good_clust:
                break
            
            elif p == 1:
                
                if coord == 0:
                    coord = 2
                else:
                    coord = 0
                        
            for n_conds in range(3,max_conds):
                
                logger.success(f"Kmeans clustering for {n_conds} clusters")
                
                good_clust, clusters = cluster_and_evaluate(X_scaled, n_conds, coord)
                
                if good_clust:
                    
                    if n_conds == md:
                        logger.critical(f"Conductor number confirmation for {md} lines")
                        response_vano['NUM_CONDUCTORES'] = int(md)
                        response_vano["NUM_CONDUCTORES_FIABLE"] = True
                        break
                    
                    elif coord == 0 and n_conds == 3 and md not in [6,7]:
                        response_vano['NUM_CONDUCTORES'] = int(n_conds)
                        response_vano["NUM_CONDUCTORES_FIABLE"] = False
                        logger.warning("Horizontal clustering cond number not confirmed")
                        break
                    
                    elif coord != 0 and n_conds == 3 and md in [6,7]:
                        response_vano['NUM_CONDUCTORES'] = int(n_conds)
                        response_vano["NUM_CONDUCTORES_FIABLE"] = False
                        logger.warning("Vertical clustering cond number not confirmed")
                        break
                    
                    elif n_conds < 3:
                        good_clust = False
                        continue
                    
                    else:
                        good_clust = False
                        break
                    
        ###############################################

        if good_clust and n_conds != 3:

            logger.error(f"Bad conductor number")
            response_vano['FLAG'] = "bad_cond_number" 
            
            return response_vano, -1
        
        elif good_clust:
            
            logger.success(f"Good clustering with n conductors: {n_conds}")
            logger.info(f"Fitting and evaluating")
        
            # pols, params, metrics = fit_and_evaluate_conds(clusters, scaled_vertices, vano_length)
            
            pols, params, metrics = fit_and_evaluate_conds(scaled_vertices, scaled_vertices, vano_length)
            
            pols = unscale_fits(pols, scaler_x, scaler_y, scaler_z)
            
            fit1, fit2, fit3 = stack_unrotate_fits(pols, mat)
            
            response_vano = puntuate_and_save(response_vano, fit1, fit2, fit3, params, metrics, n_conds)
            
            logger.critical(f"Puntuaciones aposteriori: {response_vano['PUNTUACION_APOSTERIORI']}")
            
            fits = [fit1,fit2,fit3]
            
            plot_data("good fit", cond_values, apoyo_values, fits, extremos_values)
            plt.show()
            
            # fits = [np.stack([fit1[0], fit2[0], fit3[0]]), np.stack([fit1[1], fit2[1], fit3[1]]), np.stack([fit1[2], fit2[2], fit3[2]])]

            # plot_fit_2("good fit", cond_values, apoyo_values, vert_values, fits)
            # fits = [fit3]
            # # print(np.stack(fits).shape)
            # # plot_data("good fit", cond_values, apoyo_values, fits, extremos_values)
            # # plt.show()
            # fits = [fit2]
            # print(np.stack(fits).shape)
            # plot_data("good fit", cond_values, apoyo_values, fits, extremos_values)
            # plt.show()
            
            # logger.critical(f"{np.array(pols[0]).mean(), np.array(pols[1]).mean(), np.array(pols[2]).mean()}")
            
            return response_vano, metrics
            
        else:
            
            logger.error(f"Bad clustering, next vano")
            response_vano['FLAG'] = 'bad_cluster'
            return response_vano, -1
                
        
    except Exception as e:
        logger.error(f"Vano {vano['ID_VANO']} failed preprocess: {e}")
        raise ValueError(f"Vano {vano['ID_VANO']} failed preprocess: {e}")
    

def make_summary(data):
    
    summary = pd.DataFrame(columns=["ID_VANO","RECONSTRUCCION", "FLAG", "NUM_CONDUCTORES", "NUM_CONDUCTORES_FIABLE", "CONFIG_CONDUCTORES", "COMPLETITUD", "PUNTUACION_APRIORI", "PUNTUACION_APOSTERIORI"])
    
    for i,vano in enumerate(data):
        
        try:
            row = pd.DataFrame({"ID_VANO" : vano["ID_VANO"], "RECONSTRUCCION" : vano["RECONSTRUCCION"], "FLAG" : vano["FLAG"], "NUM_CONDUCTORES" : vano["NUM_CONDUCTORES"], 
                                "NUM_CONDUCTORES_FIABLE": vano["NUM_CONDUCTORES_FIABLE"], "CONFIG_CONDUCTORES" : vano["CONFIG_CONDUCTORES"], "COMPLETITUD" : vano["COMPLETITUD"],
                                "PUNTUACION_APRIORI" : vano["PUNTUACION_APRIORI"]["NOTA"], "PUNTUACION_APOSTERIORI": vano["PUNTUACION_APOSTERIORI"]["NOTA"]}, index = [i])
                
            summary = pd.concat([summary, row])
            
        except Exception as e:
            try:
                print(e)
        
            except Exception as ex:
                pass
        
    return summary

        
def main_pipeline(pathdata0, n_vanos):
    
    import copy
    # pathdata0 = "./data/lineas_completas/XIN803.json"

    # bad_ids0 = get_bad_ids(pathdata0.split("json")[0]+"txt")

    with open(pathdata0, 'r') as archivo:
            data = json.load(archivo)

    # print(len(data), bad_ids0)

    set_logger("INFO")

    # Declare lists for metrics: rmse, correlations...
    rmses = []
    maxes = []
    correlations = []

    bad_cases = 0
    good_cases = 0

    # Define a dictionary to store the results of all fit evaluation to extract Puntuaciones
    # logger.error(f"\n NUMBER OF BAD IDS {len(bad_ids0)}")

    # Loop over data
    for i in range(len(data[:n_vanos])):
            
            logger.critical(f"\nProcessing Vano {i}")
            
            vano = copy.deepcopy(data[i])
            idv, vano_length, cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(vano)
            data[i], metrics = process_vano(data[i])
            
            if metrics != -1:
                    
                    metrics = np.array(metrics)
                    
                    rmses.append(np.mean(metrics[:,0]))
                    maxes.append(np.mean(metrics[:,1]))
                    correlations.append([np.mean(metrics[:,2]), np.mean(metrics[:,3])])
                    good_cases += 1
                    # plot_data(str(idv),cond_values, list(apoyo_values), vert_values, extremos_values)
                    # plt.show()
            else:
                    # logger.error("Bad clustering")
                    plot_data(str(idv),cond_values, list(apoyo_values), vert_values, extremos_values)
                    plt.show()
                    bad_cases += 1
                    
    logger.info(f"\nMETRICS: ")
    if len(rmses) != 0:
        
        logger.success(f"Bad cases vs good cases: {bad_cases}, {good_cases}")
        logger.success(f"Mean RMSE and mean RmaxSE: {np.array(rmses).mean().mean()}, {np.array(maxes).mean().mean()}")
        logger.success(f"Mean correlation R Pearson and Spearman: {np.array(correlations)[:,0].mean().mean()}, {np.array(correlations)[:,1].mean().mean()}")
        
        plt.figure(figsize=(12,8))

        plt.subplot(121)
        plt.hist(np.array(rmses).flatten())
        plt.subplot(122)
        plt.hist(np.array(maxes).flatten())

        plt.title("RMSE and MaxError distribution")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12,8))

        plt.subplot(121)
        plt.hist(np.array(correlations)[:,0].flatten())
        plt.subplot(122)
        plt.hist(np.array(correlations)[:,1].flatten())

        plt.title("SPEARMAN R and PEARSON R distribution")
        plt.tight_layout()
        plt.show()
        
    else:
        logger.warning("Empty metrics, no good cases...")
  
    summary = make_summary(data[:n_vanos])
    logger.success(summary.to_string())
        
    return data[:n_vanos], summary
