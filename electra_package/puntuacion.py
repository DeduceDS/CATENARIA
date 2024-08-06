
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# pathdata0 = "./data/lineas_completas/VDI711.json"
# pathdata1 = "./data/lineas_completas/REG804.json"
# pathdata2 = "./data/lineas_completas/XIN803.json"

def get_coord(points):
    
    x_vals = [punto[0] for punto in points]
    y_vals = [punto[1] for punto in points]
    z_vals = [punto[2] for punto in points]
    
    return np.stack(x_vals), np.stack(y_vals), np.stack(z_vals)

def get_coord2(extremos_apoyos):
    
    x_vals = [extremos_apoyos[0]["COORDENADA_X"], extremos_apoyos[0]["COORDENADA_X"], extremos_apoyos[1]["COORDENADA_X"], extremos_apoyos[1]["COORDENADA_X"]]
    y_vals = [extremos_apoyos[0]["COORDEANDA_Y"], extremos_apoyos[0]["COORDEANDA_Y"], extremos_apoyos[1]["COORDEANDA_Y"], extremos_apoyos[1]["COORDEANDA_Y"]]
    z_vals = extremos_apoyos[0]["COORDENADAS_Z"] + extremos_apoyos[1]["COORDENADAS_Z"]

    return np.stack(x_vals), np.stack(y_vals), np.stack(z_vals)

def get_coord3(extremos_apoyos):
    
    x_vals = [extremos_apoyos[0]["COORDENADA_X"], extremos_apoyos[0]["COORDENADA_X"]]
    y_vals = [extremos_apoyos[0]["COORDEANDA_Y"], extremos_apoyos[0]["COORDEANDA_Y"]]
    z_vals = extremos_apoyos[0]["COORDENADAS_Z"]

    return np.stack(x_vals), np.stack(y_vals), np.stack(z_vals)

def unravel_data_element(element):
    
    for key in element.keys():
    
        if type(element[key]) in [list, dict]:
            
            print(f"\n{key}: ")
            
            if type(element[key]) == list:
            
                element2 = element[key][0]
                print(f"- Length of list: {len(element[key])}")
                
            else:
                
                element2 = element[key]
            
            for key2 in element2.keys():
                
                print(f"    {key2}: {element2[key2]}")
                
                if type(element2[key2]) == list:
                    print(f"    - Length of list: {len(element2[key2])}")
        
        else:
            print(f"\n{key}: {element[key]}")

def rotate_points(points, extremos_values):
    
    points = np.array(points).T

    extremo1 = np.array(extremos_values).T[0]  # Extremo superior del primer poste
    extremo2 = np.array(extremos_values).T[2]  # Extremo inferior del primer poste
    
    # Calcular la distancia en el plano XY y la dirección de la diagonal
    distancia_xy = np.linalg.norm(extremo2[:2] - extremo1[:2])
    direccion_diagonal = (extremo2[:2] - extremo1[:2]) / distancia_xy # Normalizada para la distancia
    
    # Calcular el ángulo de rotación necesario para alinear la diagonal con el eje Y
    angulo = np.arctan2(direccion_diagonal[1], direccion_diagonal[0])
    
    # Ajustar el ángulo para la rotación correcta
    angulo += np.pi / 2
    c, s = np.cos(angulo), np.sin(angulo)
    
    # Crear la matriz de rotación para alinear la diagonal con el eje Y
    matriz_rotacion = np.array([[c, s, 0],
                                [-s, c, 0],
                                [0, 0, 1]])
    
    rotated_points = matriz_rotacion.dot(points.T)
    # print(rotated.shape)
    
    return matriz_rotacion, np.array(rotated_points)

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
    
def num_apoyos_LIDAR(data, vano):
    puntos_apoyos = data[vano]['LIDAR']['APOYOS']

    x_vals_apoyos, y_vals_apoyos, z_vals_apoyos = get_coord(puntos_apoyos)

    X = np.column_stack((y_vals_apoyos, z_vals_apoyos))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    centros = kmeans.cluster_centers_
    dif = 2
    if centros[0][0] - dif <= centros[1][0] <= centros[0][0] + dif:
        return 1
    else:
        return 2
    
def ajuste(data, vano):

    puntos_conductores = data[vano]['LIDAR']['CONDUCTORES']
    puntos_vertices = data[vano]['CONDUCTORES'][0]['VERTICES']
    puntos_vertices2 = data[vano]['CONDUCTORES'][1]['VERTICES']
    puntos_vertices3 = data[vano]['CONDUCTORES'][2]['VERTICES']
    puntos_extremos = data[vano]['APOYOS']

    x_vals_conductores, y_vals_conductores, z_vals_conductores = get_coord(puntos_conductores)
    x_vals_extremos, y_vals_extremos, z_vals_extremos = get_coord2(puntos_extremos)
    x_vert1, y_vert1, z_vert1 = get_coord(puntos_vertices)
    x_vert2, y_vert2, z_vert2 = get_coord(puntos_vertices2)
    x_vert3, y_vert3, z_vert3 = get_coord(puntos_vertices3)

    cond_values = [x_vals_conductores, y_vals_conductores, z_vals_conductores]
    extremos_values = [x_vals_extremos, y_vals_extremos, z_vals_extremos]
    vert_values1 = [x_vert1, y_vert1, z_vert1]
    vert_values2 = [x_vert2, y_vert2, z_vert2]
    vert_values3 = [x_vert3, y_vert3, z_vert3]

    # Matriz de rotación
    mat, rotated_conds = rotate_points(cond_values, extremos_values)
    extremos_values = mat.dot(extremos_values)
    rotated_vertices1 = mat.dot(vert_values1)
    rotated_vertices2 = mat.dot(vert_values2)
    rotated_vertices3 = mat.dot(vert_values3)

    X_extremos = extremos_values[0]
    Y_extremos = extremos_values[1]
    Z_extremos = extremos_values[2]

    X_cond = rotated_conds[0]
    Y_cond = rotated_conds[1]
    Z_cond = rotated_conds[2]

    # Filtramos los puntos de los conductores que están entre los extremos
    x = []
    y = []
    z = []

    for i in range(len(X_cond)):
        if Y_cond[i] > np.min(Y_extremos) and Y_cond[i] < np.max(Y_extremos):
            x.append(X_cond[i])
            y.append(Y_cond[i])
            z.append(Z_cond[i])

    x_cond = np.array(x)
    y_cond = np.array(y)
    z_cond = np.array(z)

    # Clustering
    [X, y] = [x_cond.reshape(-1, 1), y_cond.reshape(-1, 1)]

    model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0)

    y_spectral = model.fit_predict(X)

    # Separamos los 3 cables
    x1, x2, x3 = [], [], []
    y1, y2, y3 = [], [], []
    z1, z2, z3 = [], [], []

    for i in range(0, len(y_spectral)):
        if y_spectral[i] == 0:
            x1.append(X[i])
            y1.append(y[i])
            z1.append(z[i])
        if y_spectral[i] == 1:
            x2.append(X[i])
            y2.append(y[i])
            z2.append(z[i])
        if y_spectral[i] == 2:
            x3.append(X[i])
            y3.append(y[i])
            z3.append(z[i])

    x1, y1, z1 = np.array(x1), np.array(y1), np.array(z1)
    x2, y2, z2 = np.array(x2), np.array(y2), np.array(z2)
    x3, y3, z3 = np.array(x3), np.array(y3), np.array(z3)

    # Ajuste de la catenaria
    data_2d_cond1 = np.column_stack((y1, z1))
    data_2d_cond2 = np.column_stack((y2, z2))
    data_2d_cond3 = np.column_stack((y3, z3))

    pca = PCA(n_components=2)
    data_2d_pca_cond1 = pca.fit_transform(data_2d_cond1)
    data_2d_pca_cond2 = pca.fit_transform(data_2d_cond2)
    data_2d_pca_cond3 = pca.fit_transform(data_2d_cond3)

    y_min_cond1, y_max_cond1 = data_2d_pca_cond1[:, 1].min(), data_2d_pca_cond1[:,1].max()
    y_min_cond2, y_max_cond2 = data_2d_pca_cond2[:, 1].min(), data_2d_pca_cond2[:,1].max()
    y_min_cond3, y_max_cond3 = data_2d_pca_cond3[:, 1].min(), data_2d_pca_cond3[:,1].max()

    f_ind1 = (data_2d_pca_cond1[:,1] > y_min_cond1) & (data_2d_pca_cond1[:,1] < y_max_cond1)
    f_ind2 = (data_2d_pca_cond2[:,1] > y_min_cond2) & (data_2d_pca_cond2[:,1] < y_max_cond2)
    f_ind3 = (data_2d_pca_cond3[:,1] > y_min_cond3) & (data_2d_pca_cond3[:,1] < y_max_cond3)
    x_filt_cond1, y_filt_cond1, z_filt_cond1 = x1[f_ind1], y1[f_ind1], z1[f_ind1]
    x_filt_cond2, y_filt_cond2, z_filt_cond2 = x2[f_ind2], y2[f_ind2], z2[f_ind2]
    x_filt_cond3, y_filt_cond3, z_filt_cond3 = x3[f_ind3], y3[f_ind3], z3[f_ind3]

    # Función de la catenaria
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import curve_fit
    def catenaria(x, a, h, k):
        return a*np.cosh((x-h)/a)+k

    y_vals1 = y_filt_cond1.reshape(-1, 1)
    z_vals1 = z_filt_cond1.reshape(-1, 1)
    y_vals2 = y_filt_cond2.reshape(-1, 1)
    z_vals2 = z_filt_cond2.reshape(-1, 1)
    y_vals3 = y_filt_cond3.reshape(-1, 1)
    z_vals3 = z_filt_cond3.reshape(-1, 1)


    scaler_y1 = StandardScaler()
    scaler_z1 = StandardScaler()
    scaler_y2 = StandardScaler()
    scaler_z2 = StandardScaler()
    scaler_y3 = StandardScaler()
    scaler_z3 = StandardScaler()

    y_vals_scaled1 = scaler_y1.fit_transform(y_vals1).flatten()
    z_vals_scaled1 = scaler_z1.fit_transform(z_vals1).flatten()
    y_vals_scaled2 = scaler_y2.fit_transform(y_vals2).flatten()
    z_vals_scaled2 = scaler_z2.fit_transform(z_vals2).flatten()
    y_vals_scaled3 = scaler_y3.fit_transform(y_vals3).flatten()
    z_vals_scaled3 = scaler_z3.fit_transform(z_vals3).flatten()

    p0 = [1, 0, 0]

    parametros1, _ = curve_fit(catenaria, y_vals_scaled1.flatten(), z_vals_scaled1)
    parametros2, _ = curve_fit(catenaria, y_vals_scaled2.flatten(), z_vals_scaled2)
    parametros3, _ = curve_fit(catenaria, y_vals_scaled3.flatten(), z_vals_scaled3)

    # Ajuste de los puntos de los datos a una catenaria
    fitted_z_vals_scaled1 = catenaria(y_vals_scaled1.flatten(), *parametros1)
    fitted_z_vals1 = scaler_z1.inverse_transform(fitted_z_vals_scaled1.reshape(-1, 1)).flatten()
    fitted_z_vals_scaled2 = catenaria(y_vals_scaled2.flatten(), *parametros2)
    fitted_z_vals2 = scaler_z2.inverse_transform(fitted_z_vals_scaled2.reshape(-1, 1)).flatten()
    fitted_z_vals_scaled3 = catenaria(y_vals_scaled3.flatten(), *parametros3)
    fitted_z_vals3 = scaler_z3.inverse_transform(fitted_z_vals_scaled3.reshape(-1, 1)).flatten()

    # Interpolación de la polilínea
    minimo = np.min(Y_extremos)
    maximo = np.max(Y_extremos)
    x_pol1 = np.linspace(minimo, maximo, 1000)

    x_pol2 = np.linspace(minimo, maximo, 1000)

    x_pol3 = np.linspace(minimo, maximo, 1000)

    scaler_x1 = StandardScaler()
    scaler_x2 = StandardScaler()
    scaler_x3 = StandardScaler()

    x_scaled1 = scaler_x1.fit_transform(x_pol1.reshape(-1, 1)).flatten()
    x_scaled2 = scaler_x2.fit_transform(x_pol2.reshape(-1, 1)).flatten()
    x_scaled3 = scaler_x3.fit_transform(x_pol3.reshape(-1, 1)).flatten()

    fitted_y_scaled1 = catenaria(x_scaled1.flatten(), *parametros1)
    fitted_y1 = scaler_z1.inverse_transform(fitted_y_scaled1.reshape(-1, 1)).flatten()
    fitted_y_scaled2 = catenaria(x_scaled2.flatten(), *parametros2)
    fitted_y2 = scaler_z2.inverse_transform(fitted_y_scaled2.reshape(-1, 1)).flatten()
    fitted_y_scaled3 = catenaria(x_scaled3.flatten(), *parametros3)
    fitted_y3 = scaler_z3.inverse_transform(fitted_y_scaled3.reshape(-1, 1)).flatten()

    y_pol1 = np.interp(x_pol1, scaler_y1.inverse_transform(y_vals_scaled1.reshape(-1, 1)).flatten(), fitted_z_vals1, period=1000)
    y_pol2 = np.interp(x_pol2, scaler_y2.inverse_transform(y_vals_scaled2.reshape(-1, 1)).flatten(), fitted_z_vals2, period=1000)
    y_pol3 = np.interp(x_pol3, scaler_y3.inverse_transform(y_vals_scaled3.reshape(-1, 1)).flatten(), fitted_z_vals3, period=1000)

    # Huecos

    long_pol1 = np.sqrt((rotated_vertices1[1][0]-rotated_vertices1[1][-1])**2 + (rotated_vertices1[2][0]-rotated_vertices1[2][-1])**2)
    long_pol2 = np.sqrt((rotated_vertices2[1][0]-rotated_vertices2[1][-1])**2 + (rotated_vertices2[2][0]-rotated_vertices2[2][-1])**2)
    long_pol3 = np.sqrt((rotated_vertices3[1][0]-rotated_vertices3[1][-1])**2 + (rotated_vertices3[2][0]-rotated_vertices3[2][-1])**2)

    # Error su polilínea
    errorx1_suya = rmse(rotated_vertices1[1], y1)
    errory1_suya = rmse(rotated_vertices1[2], z1)
    errorx2_suya = rmse(rotated_vertices2[1], y2)
    errory2_suya = rmse(rotated_vertices2[2], z2)
    errorx3_suya = rmse(rotated_vertices3[1], y3)
    errory3_suya = rmse(rotated_vertices3[2], z3)

    errorp1_suya = np.sqrt(errorx1_suya**2 + errory1_suya**2)
    errorp2_suya = np.sqrt(errorx2_suya**2 + errory2_suya**2)
    errorp3_suya = np.sqrt(errorx3_suya**2 + errory3_suya**2)

    error_suya = (errorp1_suya + errorp2_suya + errorp3_suya) / 3

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

    longitud = data[vano]['LONGITUD_2D']
    p_huecos_1 = 0
    for i in range(len(huecos1[0])):
        p_huecos_1 = p_huecos_1 + (distancias1y[huecos1[0][i]]/longitud)*100
    p_huecos_2 = 0
    for i in range(len(huecos2[0])):
        p_huecos_2 = p_huecos_2 + (distancias2y[huecos2[0][i]]/longitud)*100
    p_huecos_3 = 0
    for i in range(len(huecos3[0])):
        p_huecos_3 = p_huecos_3 + (distancias3y[huecos3[0][i]]/longitud)*100

    p_huecos = (p_huecos_1 + p_huecos_2 + p_huecos_3)/3
    p_huecos = float(p_huecos)

    return (long_pol1, long_pol2, long_pol3, error_suya, error_nuestra, p_huecos)

def evaluar_ajuste(x_pols, y_pols, rotated_vertices, longitud_vano, clusters):
    
    # logger.info(f"{len(clusters), clusters}")
    y1, z1 = clusters[0][1,:], clusters[0][2,:]
    y2, z2 = clusters[1][1,:], clusters[1][2,:]
    y3, z3 = clusters[2][1,:], clusters[2][2,:]
    
    x_pol1, x_pol2, x_pol3 = x_pols[0], x_pols[1], x_pols[2]
    y_pol1, y_pol2, y_pol3 = y_pols[0], y_pols[1], y_pols[2]
    
    # Huecos
    long_pol1 = np.sqrt((rotated_vertices[0][1][0]-rotated_vertices[0][1][-1])**2 + (rotated_vertices[0][2][0]-rotated_vertices[0][2][-1])**2)
    long_pol2 = np.sqrt((rotated_vertices[1][1][0]-rotated_vertices[1][1][-1])**2 + (rotated_vertices[1][2][0]-rotated_vertices[1][2][-1])**2)
    long_pol3 = np.sqrt((rotated_vertices[2][1][0]-rotated_vertices[2][1][-1])**2 + (rotated_vertices[2][2][0]-rotated_vertices[2][2][-1])**2)

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

    return (long_pol1, long_pol2, long_pol3, error_suya, error_nuestra, p_huecos)
    
def rmse_suya_2(vano):
    """
    RMSE y huecos intermedios de los datos
    """
    puntos_conductores = vano['LIDAR']['CONDUCTORES']
    puntos_vertices = vano['CONDUCTORES'][0]['VERTICES']
    puntos_vertices2 = vano['CONDUCTORES'][1]['VERTICES']
    puntos_extremos = vano['APOYOS']

    x_vals_conductores, y_vals_conductores, z_vals_conductores = get_coord(puntos_conductores)
    x_vals_extremos, y_vals_extremos, z_vals_extremos = get_coord2(puntos_extremos)
    x_vert1, y_vert1, z_vert1 = get_coord(puntos_vertices)
    x_vert2, y_vert2, z_vert2 = get_coord(puntos_vertices2)

    cond_values = [x_vals_conductores, y_vals_conductores, z_vals_conductores]
    extremos_values = [x_vals_extremos, y_vals_extremos, z_vals_extremos]
    vert_values1 = [x_vert1, y_vert1, z_vert1]
    vert_values2 = [x_vert2, y_vert2, z_vert2]

    # Matriz de rotación
    mat, rotated_conds = rotate_points(cond_values, extremos_values)
    extremos_values = mat.dot(extremos_values)
    rotated_vertices1 = mat.dot(vert_values1)
    rotated_vertices2 = mat.dot(vert_values2)

    X_extremos = extremos_values[0]
    Y_extremos = extremos_values[1]
    Z_extremos = extremos_values[2]

    X_cond = rotated_conds[0]
    Y_cond = rotated_conds[1]
    Z_cond = rotated_conds[2]

    # Filtramos los puntos de los conductores que están entre los extremos
    x = []
    y = []
    z = []

    for i in range(len(X_cond)):
        if Y_cond[i] > np.min(Y_extremos) and Y_cond[i] < np.max(Y_extremos):
            x.append(X_cond[i])
            y.append(Y_cond[i])
            z.append(Z_cond[i])

    x_cond = np.array(x)
    y_cond = np.array(y)
    z_cond = np.array(z)

    # Clustering
    [X, y] = [x_cond.reshape(-1, 1), y_cond.reshape(-1, 1)]

    model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)

    y_spectral = model.fit_predict(X)

    x1, x2 = [], []
    y1, y2 = [], []
    z1, z2 = [], []


    for i in range(0, len(y_spectral)):
        if y_spectral[i] == 0:
            x1.append(X[i])
            y1.append(y[i])
            z1.append(z[i])
        if y_spectral[i] == 1:
            x2.append(X[i])
            y2.append(y[i])
            z2.append(z[i])

    x1, y1, z1 = np.array(x1), np.array(y1), np.array(z1)
    x2, y2, z2 = np.array(x2), np.array(y2), np.array(z2)

    errorx1_suya = rmse(rotated_vertices1[1], y1)
    errory1_suya = rmse(rotated_vertices1[2], z1)
    errorx2_suya = rmse(rotated_vertices2[1], y2)
    errory2_suya = rmse(rotated_vertices2[2], z2)

    errorp1_suya = np.sqrt(errorx1_suya**2 + errory1_suya**2)
    errorp2_suya = np.sqrt(errorx2_suya**2 + errory2_suya**2)

    error_suya = (errorp1_suya + errorp2_suya) / 2

    y1, y2 = np.sort(y1, axis=0), np.sort(y2, axis=0)

    distancias1y = [y1[i]-y1[i+1] for i in range(len(y1)-1)]
    distancias2y = [y2[i]-y2[i+1] for i in range(len(y2)-1)]

    distancias1y = np.abs(distancias1y)
    distancias2y = np.abs(distancias2y)

    distancias1y = np.array(distancias1y)
    distancias2y = np.array(distancias2y)

    huecos1 = np.where(distancias1y > 5)
    huecos2 = np.where(distancias2y > 5)

    longitud = vano['LONGITUD_2D']
    p_huecos_1 = 0
    for i in range(len(huecos1[0])):
        p_huecos_1 = p_huecos_1 + (distancias1y[huecos1[0][i]]/longitud)*100
    p_huecos_2 = 0
    for i in range(len(huecos2[0])):
        p_huecos_2 = p_huecos_2 + (distancias2y[huecos2[0][i]]/longitud)*100
    p_huecos = (p_huecos_1 + p_huecos_2) / 2
    p_huecos = float(p_huecos)

    return error_suya, p_huecos

def rmse_suya_1(vano):
    """
    RMSE y huecos intermedios de los datos
    """
    puntos_conductores = vano['LIDAR']['CONDUCTORES']
    puntos_vertices = vano['CONDUCTORES'][0]['VERTICES']
    puntos_vertices2 = vano['CONDUCTORES'][1]['VERTICES']
    puntos_vertices3 = vano['CONDUCTORES'][2]['VERTICES']
    puntos_extremos = vano['APOYOS']

    x_vals_conductores, y_vals_conductores, z_vals_conductores = get_coord(puntos_conductores)
    x_vals_extremos, y_vals_extremos, z_vals_extremos = get_coord2(puntos_extremos)
    x_vert1, y_vert1, z_vert1 = get_coord(puntos_vertices)
    x_vert2, y_vert2, z_vert2 = get_coord(puntos_vertices2)
    x_vert3, y_vert3, z_vert3 = get_coord(puntos_vertices3)

    cond_values = [x_vals_conductores, y_vals_conductores, z_vals_conductores]
    extremos_values = [x_vals_extremos, y_vals_extremos, z_vals_extremos]
    vert_values1 = [x_vert1, y_vert1, z_vert1]
    vert_values2 = [x_vert2, y_vert2, z_vert2]
    vert_values3 = [x_vert3, y_vert3, z_vert3]

    # Matriz de rotación
    mat, rotated_conds = rotate_points(cond_values, extremos_values)
    extremos_values = mat.dot(extremos_values)
    rotated_vertices1 = mat.dot(vert_values1)
    rotated_vertices2 = mat.dot(vert_values2)
    rotated_vertices3 = mat.dot(vert_values3)

    X_extremos = extremos_values[0]
    Y_extremos = extremos_values[1]
    Z_extremos = extremos_values[2]

    X_cond = rotated_conds[0]
    Y_cond = rotated_conds[1]
    Z_cond = rotated_conds[2]

    # Filtramos los puntos de los conductores que están entre los extremos
    x = []
    y = []
    z = []

    for i in range(len(X_cond)):
        if Y_cond[i] > np.min(Y_extremos) and Y_cond[i] < np.max(Y_extremos):
            x.append(X_cond[i])
            y.append(Y_cond[i])
            z.append(Z_cond[i])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    x, y, z = np.array(x), np.array(y), np.array(z)

    errorx1_suya = rmse(rotated_vertices1[1], y)
    errory1_suya = rmse(rotated_vertices1[2], z)

    errorp1_suya = np.sqrt(errorx1_suya**2 + errory1_suya**2)

    y = np.sort(y, axis=0)

    distaciasy = [y[i]-y[i+1] for i in range(len(y)-1)]

    distanciasy = np.abs(distanciasy)

    distanciasy = np.array(distanciasy)

    huecos = np.where(distanciasy > 5)

    longitud = vano['LONGITUD_2D']

    p_huecos = 0

    for i in range(len(huecos[0])):
        p_huecos = p_huecos + (distanciasy[huecos[0][i]]/longitud)*100

    p_huecos = float(p_huecos)

    return errorp1_suya, p_huecos

def puntuación_por_vanos(data, id_vano):
    # with open(pathdata, 'r') as archivo:
    #     data = json.load(archivo)

    clasificacion1 = {
        'Vano': [],
        'Reconstrucción': [],
        'Flag': [],
        'Error polilínea': [],
        'Error nuestro ajuste': [],
    }
    for iel, el in enumerate(data):
        if id_vano == el['ID_VANO']:
            num_el = el
            num_iel = iel
            break

    apoyos = num_el['APOYOS']
    conductores = num_el['CONDUCTORES']
    len_apoyos = len(apoyos)
    len_conductores = len(conductores)
    clasificacion1['Vano'].append(num_el['ID_VANO'])
    longitud = num_el['LONGITUD_2D']
    if len_apoyos == 2:
        if len_conductores >= 3:
            clasificacion1['Reconstrucción'].append("Posible")
            long_pol1, long_pol2, long_pol3, error_suya, error_nuestra, p_huecos_intermedios = ajuste(data, num_iel)
            p_hueco_1 = ((longitud - long_pol1)/longitud)*100
            p_hueco_2 = ((longitud - long_pol2)/longitud)*100
            p_hueco_3 = ((longitud - long_pol3)/longitud)*100
            p_hueco = (p_hueco_1 + p_hueco_2 + p_hueco_3) / 3
            p_hueco = (p_hueco + p_huecos_intermedios) / 2
            clasificacion1['Flag'].append(f"Tiene 3 o más conductores. Porcentaje de huecos: {p_hueco:.2f}%")
            clasificacion1['Error polilínea'].append(error_suya)
            clasificacion1['Error nuestro ajuste'].append(error_nuestra-error_nuestra*p_hueco/100)
        elif len_conductores == 2:
            clasificacion1['Reconstrucción'].append("No posible")
            x1, y1, z1 = get_coord(conductores[0]['VERTICES'])
            x2, y2, z2 = get_coord(conductores[1]['VERTICES'])
            long_pol1 = np.sqrt((x1[0]-x1[-1])**2 + (y1[0]-y1[-1])**2)
            long_pol2 = np.sqrt((x2[0]-x2[-1])**2 + (y2[0]-y2[-1])**2)
            p_hueco_1 = ((longitud - long_pol1)/longitud)*100
            p_hueco_2 = ((longitud - long_pol2)/longitud)*100
            p_hueco = (p_hueco_1 + p_hueco_2) / 2
            error_suya, p_huecos_intermedios = rmse_suya_2(data, num_iel)
            p_hueco = (p_hueco + p_huecos_intermedios) / 2
            clasificacion1['Flag'].append(f"Tiene 2 conductores. Porcentaje de huecos: {p_hueco:.2f}%")
            clasificacion1['Error polilínea'].append(error_suya)
            clasificacion1['Error nuestro ajuste'].append(0)
        elif len_conductores == 1:
            clasificacion1['Reconstrucción'].append("No posible")
            x1, y1, z1 = get_coord(conductores[0]['VERTICES'])
            long_pol1 = np.sqrt((x1[0]-x1[-1])**2 + (y1[0]-y1[-1])**2)
            p_hueco_1 = ((longitud - long_pol1)/longitud)*100
            error_suya, p_huecos_intermedios = rmse_suya_1(data, num_iel)
            p_hueco_1 = (p_hueco_1 + p_huecos_intermedios) / 2
            clasificacion1['Flag'].append(f"Tiene 1 conductor. Porcentaje de huecos: {p_hueco_1:.2f}%")
            clasificacion1['Error polilínea'].append(error_suya)
            clasificacion1['Error nuestro ajuste'].append(0)
        else:
            clasificacion1['Reconstrucción'].append("No posible")
            clasificacion1['Flag'].append('No tiene conductores')
            clasificacion1['Error polilínea'].append(0)
            clasificacion1['Error nuestro ajuste'].append(0)
    else:
        clasificacion1['Reconstrucción'].append("No posible")
        apoyos_LIDAR = num_apoyos_LIDAR(data, num_iel)
        clasificacion1['Error polilínea'].append(0)
        clasificacion1['Error nuestro ajuste'].append(0)
        if apoyos_LIDAR == 2:
            clasificacion1['Flag'].append('No tiene 2 apoyos cartografiados pero tiene 2 apoyos LIDAR.')
        else:
            clasificacion1['Flag'].append('No tiene 2 apoyos cartografiados ni 2 apoyos LIDAR.')
    clasificacion1 = pd.DataFrame(clasificacion1)
    return(clasificacion1)


def puntuación_por_vanos_sin_ajuste(data, id_vano, evaluaciones):
    # with open(pathdata, 'r') as archivo:
    #     data = json.load(archivo)
    
    logger.success(f"Setting vano score with old puntuation function")    
    clasificacion1 = {
        'Vano': [],
        'Reconstrucción': [],
        'Flag': [],
        'Error polilínea': [],
        'Error nuestro ajuste': [],
    }
    
    
    for iel, el in enumerate(data):
        if id_vano == el['ID_VANO']:
            num_el = el
            num_iel = iel
            break

    apoyos = num_el['APOYOS']
    conductores = num_el['CONDUCTORES']
    len_apoyos = len(apoyos)
    len_conductores = len(conductores)
    clasificacion1['Vano'].append(num_el['ID_VANO'])
    longitud = num_el['LONGITUD_2D']
    
    if len_apoyos == 2:
        if len_conductores >= 3:
            
            resultados_evaluacion = evaluaciones
            
            clasificacion1['Reconstrucción'].append("Posible")
            
            long_pol1, long_pol2, long_pol3 = resultados_evaluacion[0], resultados_evaluacion[1], resultados_evaluacion[2]
            error_suya, error_nuestra = resultados_evaluacion[3], resultados_evaluacion[4]
            p_huecos_intermedios = resultados_evaluacion[5]
            
            p_hueco_1 = ((longitud - long_pol1)/longitud)*100
            p_hueco_2 = ((longitud - long_pol2)/longitud)*100
            p_hueco_3 = ((longitud - long_pol3)/longitud)*100
            p_hueco = (p_hueco_1 + p_hueco_2 + p_hueco_3) / 3
            p_hueco = (p_hueco + p_huecos_intermedios) / 2
            
            clasificacion1['Flag'].append(f"Tiene 3 o más conductores. Porcentaje de huecos: {p_hueco:.2f}%")
            clasificacion1['Error polilínea'].append(error_suya)
            clasificacion1['Error nuestro ajuste'].append(error_nuestra-error_nuestra*p_hueco/100)
        elif len_conductores == 2:
            clasificacion1['Reconstrucción'].append("No posible")
            x1, y1, z1 = get_coord(conductores[0]['VERTICES'])
            x2, y2, z2 = get_coord(conductores[1]['VERTICES'])
            long_pol1 = np.sqrt((x1[0]-x1[-1])**2 + (y1[0]-y1[-1])**2)
            long_pol2 = np.sqrt((x2[0]-x2[-1])**2 + (y2[0]-y2[-1])**2)
            p_hueco_1 = ((longitud - long_pol1)/longitud)*100
            p_hueco_2 = ((longitud - long_pol2)/longitud)*100
            p_hueco = (p_hueco_1 + p_hueco_2) / 2
            error_suya, p_huecos_intermedios = rmse_suya_2(data, num_iel)
            p_hueco = (p_hueco + p_huecos_intermedios) / 2
            clasificacion1['Flag'].append(f"Tiene 2 conductores. Porcentaje de huecos: {p_hueco:.2f}%")
            clasificacion1['Error polilínea'].append(error_suya)
            clasificacion1['Error nuestro ajuste'].append(0)
        elif len_conductores == 1:
            clasificacion1['Reconstrucción'].append("No posible")
            x1, y1, z1 = get_coord(conductores[0]['VERTICES'])
            long_pol1 = np.sqrt((x1[0]-x1[-1])**2 + (y1[0]-y1[-1])**2)
            p_hueco_1 = ((longitud - long_pol1)/longitud)*100
            error_suya, p_huecos_intermedios = rmse_suya_1(data, num_iel)
            p_hueco_1 = (p_hueco_1 + p_huecos_intermedios) / 2
            clasificacion1['Flag'].append(f"Tiene 1 conductor. Porcentaje de huecos: {p_hueco_1:.2f}%")
            clasificacion1['Error polilínea'].append(error_suya)
            clasificacion1['Error nuestro ajuste'].append(0)
        else:
            clasificacion1['Reconstrucción'].append("No posible")
            clasificacion1['Flag'].append('No tiene conductores')
            clasificacion1['Error polilínea'].append(0)
            clasificacion1['Error nuestro ajuste'].append(0)
    else:
        clasificacion1['Reconstrucción'].append("No posible")
        apoyos_LIDAR = 2
        clasificacion1['Error polilínea'].append(0)
        clasificacion1['Error nuestro ajuste'].append(0)
        if apoyos_LIDAR == 2:
            clasificacion1['Flag'].append('No tiene 2 apoyos cartografiados pero tiene 2 apoyos LIDAR.')
        else:
            clasificacion1['Flag'].append('No tiene 2 apoyos cartografiados ni 2 apoyos LIDAR.')
    clasificacion1 = pd.DataFrame(clasificacion1)
    return(clasificacion1)


def puntuación_por_vano(response_vano, evaluaciones, longitud):

    logger.success(f"Setting vano score with old puntuation function")    
    
    # longitud = response_vano['LONGITUD_2D']
    len_conductores = response_vano["line_number"]

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
        p_hueco = p_hueco[0]
        
        response_vano["reconstruccion"] = "Posible"
        response_vano["huecos"] = f"{p_hueco:.2f}%"
        response_vano["Error_polilinea"] = error_suya
        response_vano["Error_catenaria"] = error_nuestra-error_nuestra*p_hueco/100
        
    # elif len_conductores == 2:
        
    #     clasificacion1['Reconstrucción'].append("No posible")
    #     x1, y1, z1 = get_coord(conductores[0]['VERTICES'])
    #     x2, y2, z2 = get_coord(conductores[1]['VERTICES'])
    #     long_pol1 = np.sqrt((x1[0]-x1[-1])**2 + (y1[0]-y1[-1])**2)
    #     long_pol2 = np.sqrt((x2[0]-x2[-1])**2 + (y2[0]-y2[-1])**2)
    #     p_hueco_1 = ((longitud - long_pol1)/longitud)*100
    #     p_hueco_2 = ((longitud - long_pol2)/longitud)*100
    #     p_hueco = (p_hueco_1 + p_hueco_2) / 2
        
    #     error_suya, p_huecos_intermedios = rmse_suya_2(vano)
        
    #     p_hueco = (p_hueco + p_huecos_intermedios) / 2
    #     clasificacion1['Flag'].append(f"Tiene 2 conductores. Porcentaje de huecos: {p_hueco:.2f}%")
    #     clasificacion1['Error polilínea'].append(error_suya)
    #     clasificacion1['Error nuestro ajuste'].append(0)
        
    # elif len_conductores == 1:
    #     clasificacion1['Reconstrucción'].append("No posible")
    #     x1, y1, z1 = get_coord(conductores[0]['VERTICES'])
    #     long_pol1 = np.sqrt((x1[0]-x1[-1])**2 + (y1[0]-y1[-1])**2)
    #     p_hueco_1 = ((longitud - long_pol1)/longitud)*100
    #     error_suya, p_huecos_intermedios = rmse_suya_1(vano)
    #     p_hueco_1 = (p_hueco_1 + p_huecos_intermedios) / 2
    #     clasificacion1['Flag'].append(f"Tiene 1 conductor. Porcentaje de huecos: {p_hueco_1:.2f}%")
    #     clasificacion1['Error polilínea'].append(error_suya)
    #     clasificacion1['Error nuestro ajuste'].append(0)
    else:
        
        response_vano["reconstruccion"] = "No posible (n_cond != 3)"
        response_vano["huecos"] = "0 %"
        response_vano["Error_polilinea"] = 0
        response_vano["Error_catenaria"] = 0

    return response_vano

