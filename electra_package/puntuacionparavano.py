#LO QUE NECESITAN LAS FUNCIONES A CAMBIAR PARA FUNCIONAR:

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.cluster import SpectralClustering

from electra_package.modules_utils import *
from electra_package.modules_preprocess import *


def rmse(x, y):
    """
    RMSE entre las polilíneas y los puntos LIDAR
    """
    if len(x) >= len(y):
        intervalo = len(x) // (len(y)-1)
        nn = [x[i * intervalo] for i in range(len(y)-1)] + [x[-1]]
        return np.sqrt(mean_squared_error(nn, y))
    else:
        intervalo = len(y) // (len(x)-1)
        nn = [y[i * intervalo] for i in range(len(x)-1)] + [y[-1]]
        return np.sqrt(mean_squared_error(nn, x))
    
    # hay que rotar antes de calcular las puntuaciones apriori

def puntuacion_apriori(cond_values, extremos_values, apoyo_values, vert_values, vano_length):
    
    _, _, _, rotated_vertices, _ = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
    
    notas_apriori = dict()
    
    if np.array(extremos_values).shape[1] != 4:
        
        logger.critical(np.array(extremos_values).shape[1])
        logger.warning(f"No tiene 2 apoyos")
        
        notas_apriori["NOTA"] = 0
        notas_apriori["P_HUECO"] = 0
        notas_apriori["DIFF2D"] = 0
        notas_apriori["NOTA"] = 0
        
        return notas_apriori
        

    nota = 0
    huecos = []
    diffs = []
    
    for cond in len(rotated_vertices):
        
        long_poli= np.sqrt((cond[1][0]-cond[1][-1])**2 + (cond[2][0]-cond[2][-1])**2)
        diff_2d = abs(long_poli - vano_length)/vano_length
        
        distancias = [cond[1][i]-cond[1][i+1] for i in range(1, len(cond[1])-2)]
        
        mean_dist = np.mean(distancias)
        std_dist = np.std(distancias)
        
        thresh = mean_dist + 2*std_dist
        total_huecos = np.sum(distancias[np.where(distancias >= thresh)])
        
        p_huecos = abs(total_huecos - vano_length)/vano_length
        
        if p_huecos + diff_2d >= 1:
            logger.warning(f"Bad error asigned (p_huecos, diff_2d): {p_huecos, diff_2d}")
        
        nota_cond = 3.33*(1 - p_huecos - diff_2d)
        nota += nota_cond
        
        huecos.append(p_huecos)
        huecos.append(diff_2d)

    notas_apriori["P_HUECO"] = p_huecos
    notas_apriori["DIFF2D"] = diffs
    notas_apriori["NOTA"] = nota
    
    return notas_apriori

def puntuacion_posteriori(metrics, n_conds):
    
    notas_posteriori = dict()

    nota = 0
    
    p_values, correlations, rmses = [], [], []
    
    for i,cond in enumerate(range(n_conds)):
        
        spearman_z, p_value = metrics[i][3], metrics[i][4]
        
        nota_cond = 3.33*spearman_z*(1-p_value)
        nota += nota_cond
        
        correlations.append(spearman_z)
        p_values.append(p_value)
        rmses.append(metrics[i][0])

    notas_posteriori["P_VALUE"] = p_values
    notas_posteriori["CORRELATION"] = correlations
    notas_posteriori["RMSE"] = rmses
    notas_posteriori["NOTA"] = nota
    
    return notas_posteriori


    # plt.figure(figsize=(10, 6))
    # # Pintamos los puntos de cada cable
    # plt.scatter(y1, z1, color='coral', s=30)
    # plt.scatter(y2, z2, color='lightblue', s=30)
    # plt.scatter(y3, z3, color='lightgreen', s=30)

    # ############################################33

    # # Pintamos las polilíneas que hemos generado
    # plt.plot(x_pol1, y_pol1, color='red', label='P1')
    # plt.plot(x_pol2, y_pol2, color='blue', label='P2')
    # plt.plot(x_pol3, y_pol3, color='green', label='P3')

    # # Pintamos las polilíneas que nos dan con los datos
    # plt.scatter(rotated_vertices1[1], rotated_vertices1[2], color='red', label='Polilínea 1', s=30)
    # plt.scatter(rotated_vertices2[1], rotated_vertices2[2], color='blue', label='Polilínea 2', s=30)
    # plt.scatter(rotated_vertices3[1], rotated_vertices3[2], color='green', label='Polilínea 3', s=30)

    # plt.legend()
    # plt.title(vano)


def evaluar_ajuste(x_pols, y_pols, rotated_vertices, longitud_vano, clusters):
    
    # logger.info(f"{len(clusters), clusters}")
    y1, z1 = clusters[0][1,:], clusters[0][2,:]
    y2, z2 = clusters[1][1,:], clusters[1][2,:]
    y3, z3 = clusters[2][1,:], clusters[2][2,:]
    
    x_pol1, x_pol2, x_pol3 = x_pols[0], x_pols[1], x_pols[2]
    y_pol1, y_pol2, y_pol3 = y_pols[0], y_pols[1], y_pols[2]

    # Error nuestra polilínea
    errorx1 = rmse(x_pol1, y1)
    errory1 = rmse(y_pol1, z1)
    errorx2 = rmse(x_pol2, y2)
    errory2 = rmse(y_pol2, z2)
    errorx3 = rmse(x_pol3, y3)
    errory3 = rmse(y_pol3, z3)

    errorp1_nuestra = np.sqrt(errorx1**2 + errory1**2)
    errorp2_nuestra = np.sqrt(errorx2**2 + errory2**2)
    errorp3_nuestra = np.sqrt(errorx3**2 + errory3**2)

    error_nuestra = (errorp1_nuestra + errorp2_nuestra + errorp3_nuestra) / 3

    # Porcentaje huecos intermedios

    y1, y2, y3 = np.sort(y1, axis=0), np.sort(y2, axis=0), np.sort(y3, axis=0)

    distancias1y = [y1[i]-y1[i+1] for i in range(len(y1)-1)]
    distancias2y = [y2[i]-y2[i+1] for i in range(len(y2)-1)]
    distancias3y = [y3[i]-y3[i+1] for i in range(len(y3)-1)]

    distancias1y = np.abs(distancias1y)
    distancias2y = np.abs(distancias2y)
    distancias3y = np.abs(distancias3y)

    huecos1 = np.where(distancias1y > 5)
    huecos2 = np.where(distancias2y > 5)
    huecos3 = np.where(distancias3y > 5)

    p_huecos_1 = 0
    for i in range(len(huecos1[0])):
        p_huecos_1 = p_huecos_1 + (distancias1y[huecos1[0][i]]/longitud_vano)*100
    p_huecos_2 = 0
    for i in range(len(huecos2[0])):
        p_huecos_2 = p_huecos_2 + (distancias2y[huecos2[0][i]]/longitud_vano)*100
    p_huecos_3 = 0
    for i in range(len(huecos3[0])):
        p_huecos_3 = p_huecos_3 + (distancias3y[huecos3[0][i]]/longitud_vano)*100

    p_huecos = (p_huecos_1 + p_huecos_2 + p_huecos_3)/3
    p_huecos = float(p_huecos)
    
    if len(rotated_vertices) != 3:
        logger.warning("Empty vertices...")
        return (0, 0, 0, 0, error_nuestra, p_huecos)

    for i in range(len(rotated_vertices)):
        if len(rotated_vertices[i]) == 0:
            logger.warning("Empty vertices...")
            return (0, 0, 0, 0, error_nuestra, p_huecos)

    # Error su polilínea
    errorx1_suya = rmse(rotated_vertices[0][1], y1)
    errory1_suya = rmse(rotated_vertices[0][2], z1)
    errorx2_suya = rmse(rotated_vertices[1][1], y2)
    errory2_suya = rmse(rotated_vertices[1][2], z2)
    errorx3_suya = rmse(rotated_vertices[2][1], y3)
    errory3_suya = rmse(rotated_vertices[2][2], z3)

    errorp1_suya = np.sqrt(errorx1_suya**2 + errory1_suya**2)
    errorp2_suya = np.sqrt(errorx2_suya**2 + errory2_suya**2)
    errorp3_suya = np.sqrt(errorx3_suya**2 + errory3_suya**2)

    error_suya = (errorp1_suya + errorp2_suya + errorp3_suya) / 3
    
    # Huecos
    long_pol1 = np.sqrt((rotated_vertices[0][1][0]-rotated_vertices[0][1][-1])**2 + (rotated_vertices[0][2][0]-rotated_vertices[0][2][-1])**2)
    long_pol2 = np.sqrt((rotated_vertices[1][1][0]-rotated_vertices[1][1][-1])**2 + (rotated_vertices[1][2][0]-rotated_vertices[1][2][-1])**2)
    long_pol3 = np.sqrt((rotated_vertices[2][1][0]-rotated_vertices[2][1][-1])**2 + (rotated_vertices[2][2][0]-rotated_vertices[2][2][-1])**2)

    return (long_pol1, long_pol2, long_pol3, error_suya, error_nuestra, p_huecos)


def puntuacionconparametros(vano, error_suya, error_nuestra, n_cluster,completeteness): 

    clasificacion1 = {
        'Apoyos': [],
        'Apoyos Lidar':[],
        'Conductores': [],
        'Completitud':[],
        'Numero de clusters':[],
        'Apoyos encontrados vs lidar':[],
        'Puntuación': [],
    }
  
    apoyos = vano['APOYOS']
    lidar = vano['LIDAR']
    conductores = vano['CONDUCTORES']
    len_apoyos = len(apoyos)
    len_lidar = len(lidar)
    len_conductores = len(conductores)
    clasificacion1['Apoyos'].append(len_apoyos)
    clasificacion1['Conductores'].append(len_conductores)
    clasificacion1['Completitud'].append(completeteness)
    nota = 10

    if len_apoyos == 2:
        if len_lidar == 2:
            if n_cluster == len_conductores:
                if len_conductores >= 3:
                    nota = nota
                elif len_conductores == 2:
                    nota = nota - 2
                elif len_conductores == 1:
                    nota = nota - 4
                else:
                    nota = 1
                clasificacion1['Puntuación'].append(nota) 
                if completeteness == 'full':
                    nota = nota
                elif completeteness == 'partially incomplete': 
                    nota = nota/2    
                else:
                    nota = 0 
                clasificacion1['Puntuación'].append(nota)
                #Si error nuestra esta en porcentaje      
                nota = nota - nota*(error_nuestra/100) 

                clasificacion1['Puntuación'].append(nota) 
            else: 
                clasificacion1['Puntuación'].append(0)
        else:
            clasificacion1['Puntuación'].append(0)
    else:
        clasificacion1['Puntuación'].append(0)
        

    clasificacion1 = pd.DataFrame(clasificacion1)
    return(clasificacion1)