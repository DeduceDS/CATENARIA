import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error 

def rmse(x, y):
    """
    RMSE entre las polilíneas y los puntos LIDAR
    """
    
    if x.shape[0] >= y.shape[0]:
        intervalo = x.shape[0] // (y.shape[0]-1)
        nn = [x[i * intervalo] for i in range(y.shape[0]-1)] + [x[-1]]
        return np.sqrt(mean_squared_error(nn, y))
    else:
        intervalo = y.shape[0] // (x.shape[0]-1)
        nn = [y[i * intervalo] for i in range(x.shape[0]-1)] + [y[-1]]
        return np.sqrt(mean_squared_error(nn, x))
    
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


def puntuación_por_vano(response_vano, evaluaciones, longitud):

    logger.success(f"Setting vano score with old puntuation function")    
    
    # longitud = response_vano['LONGITUD_2D']
    len_conductores = response_vano["NUM_CONDUCTORES"]

    if len_conductores == 3:
        
        resultados_evaluacion = evaluaciones
        
        long_pol1, long_pol2, long_pol3 = resultados_evaluacion[0], resultados_evaluacion[1], resultados_evaluacion[2]
        error_suya, error_nuestra = resultados_evaluacion[3], resultados_evaluacion[4]
        p_huecos_intermedios = resultados_evaluacion[5]
        
        p_hueco_1 = ((longitud - long_pol1)/longitud)*100
        p_hueco_2 = ((longitud - long_pol2)/longitud)*100
        p_hueco_3 = ((longitud - long_pol3)/longitud)*100
        p_hueco = (p_hueco_1 + p_hueco_2 + p_hueco_3) / 3
        p_hueco = (p_hueco + p_huecos_intermedios) / 2
        
        response_vano["RECONSTRUCCION"] = "Posible"
        response_vano["PORCENTAJE_HUECOS"] = p_hueco
        response_vano["ERROR_POLILINEA"] = error_suya
        response_vano["ERROR_CATENARIA"] = error_nuestra-error_nuestra*p_hueco/100
        
    else:
        
        response_vano["RECONSTRUCCION"] = "No posible (n_cond != 3)"
        response_vano["PORCENTAJE_HUECOS"] = 0
        response_vano["ERROR_POLILINEA"] = 0
        response_vano["ERROR_CATENARIA"] = 0

    return response_vano

############################ NOT IN RELEASE #########################################

    
# def rmse_suya_2(vano):
#     """
#     RMSE y huecos intermedios de los datos
#     """
#     puntos_conductores = vano['LIDAR']['CONDUCTORES']
#     puntos_vertices = vano['CONDUCTORES'][0]['VERTICES']
#     puntos_vertices2 = vano['CONDUCTORES'][1]['VERTICES']
#     puntos_extremos = vano['APOYOS']

#     x_vals_conductores, y_vals_conductores, z_vals_conductores = get_coord(puntos_conductores)
#     x_vals_extremos, y_vals_extremos, z_vals_extremos = get_coord2(puntos_extremos)
#     x_vert1, y_vert1, z_vert1 = get_coord(puntos_vertices)
#     x_vert2, y_vert2, z_vert2 = get_coord(puntos_vertices2)

#     cond_values = [x_vals_conductores, y_vals_conductores, z_vals_conductores]
#     extremos_values = [x_vals_extremos, y_vals_extremos, z_vals_extremos]
#     vert_values1 = [x_vert1, y_vert1, z_vert1]
#     vert_values2 = [x_vert2, y_vert2, z_vert2]

#     # Matriz de rotación
#     mat, rotated_conds = rotate_points(cond_values, extremos_values)
#     extremos_values = mat.dot(extremos_values)
#     rotated_vertices1 = mat.dot(vert_values1)
#     rotated_vertices2 = mat.dot(vert_values2)

#     X_extremos = extremos_values[0]
#     Y_extremos = extremos_values[1]
#     Z_extremos = extremos_values[2]

#     X_cond = rotated_conds[0]
#     Y_cond = rotated_conds[1]
#     Z_cond = rotated_conds[2]

#     # Filtramos los puntos de los conductores que están entre los extremos
#     x = []
#     y = []
#     z = []

#     for i in range(len(X_cond)):
#         if Y_cond[i] > np.min(Y_extremos) and Y_cond[i] < np.max(Y_extremos):
#             x.append(X_cond[i])
#             y.append(Y_cond[i])
#             z.append(Z_cond[i])

#     x_cond = np.array(x)
#     y_cond = np.array(y)
#     z_cond = np.array(z)

#     # Clustering
#     [X, y] = [x_cond.reshape(-1, 1), y_cond.reshape(-1, 1)]

#     model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)

#     y_spectral = model.fit_predict(X)

#     x1, x2 = [], []
#     y1, y2 = [], []
#     z1, z2 = [], []


#     for i in range(0, len(y_spectral)):
#         if y_spectral[i] == 0:
#             x1.append(X[i])
#             y1.append(y[i])
#             z1.append(z[i])
#         if y_spectral[i] == 1:
#             x2.append(X[i])
#             y2.append(y[i])
#             z2.append(z[i])

#     x1, y1, z1 = np.array(x1), np.array(y1), np.array(z1)
#     x2, y2, z2 = np.array(x2), np.array(y2), np.array(z2)

#     errorx1_suya = rmse(rotated_vertices1[1], y1)
#     errory1_suya = rmse(rotated_vertices1[2], z1)
#     errorx2_suya = rmse(rotated_vertices2[1], y2)
#     errory2_suya = rmse(rotated_vertices2[2], z2)

#     errorp1_suya = np.sqrt(errorx1_suya**2 + errory1_suya**2)
#     errorp2_suya = np.sqrt(errorx2_suya**2 + errory2_suya**2)

#     error_suya = (errorp1_suya + errorp2_suya) / 2

#     y1, y2 = np.sort(y1, axis=0), np.sort(y2, axis=0)

#     distancias1y = [y1[i]-y1[i+1] for i in range(len(y1)-1)]
#     distancias2y = [y2[i]-y2[i+1] for i in range(len(y2)-1)]

#     distancias1y = np.abs(distancias1y)
#     distancias2y = np.abs(distancias2y)

#     distancias1y = np.array(distancias1y)
#     distancias2y = np.array(distancias2y)

#     huecos1 = np.where(distancias1y > 5)
#     huecos2 = np.where(distancias2y > 5)

#     longitud = vano['LONGITUD_2D']
#     p_huecos_1 = 0
#     for i in range(len(huecos1[0])):
#         p_huecos_1 = p_huecos_1 + (distancias1y[huecos1[0][i]]/longitud)*100
#     p_huecos_2 = 0
#     for i in range(len(huecos2[0])):
#         p_huecos_2 = p_huecos_2 + (distancias2y[huecos2[0][i]]/longitud)*100
#     p_huecos = (p_huecos_1 + p_huecos_2) / 2
#     p_huecos = float(p_huecos)

#     return error_suya, p_huecos