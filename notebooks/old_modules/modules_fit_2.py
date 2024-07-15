
import json
import plotly.graph_objects as go
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as ma
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from statistics import mode

from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

from scipy.optimize import curve_fit

import math
import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import open3d as o3d

from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import SpectralClustering
from puntuacion import *


def print_element(element):

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


def get_coord(points):

    x_vals = [punto[0] for punto in points]
    y_vals = [punto[1] for punto in points]
    z_vals = [punto[2] for punto in points]

    return np.stack(x_vals), np.stack(y_vals), np.stack(z_vals)

def get_coord2(extremos_apoyos):

    x_vals = []
    y_vals = []
    z_vals = []

    for i in range(len(extremos_apoyos)):
        # z_vals.append(extremos_apoyos[i]["COORDENADAS_Z"])
        z_vals = z_vals + extremos_apoyos[i]["COORDENADAS_Z"]

    for i in range(len(extremos_apoyos)):

        x_vals = x_vals + [extremos_apoyos[i]["COORDENADA_X"], extremos_apoyos[i]["COORDENADA_X"]]
        y_vals = y_vals + [extremos_apoyos[i]["COORDEANDA_Y"], extremos_apoyos[i]["COORDEANDA_Y"]]

    return np.stack(x_vals), np.stack(y_vals), np.stack(z_vals)


def extract_vano_values(data, vano):

    puntos_conductores = data[vano]['LIDAR']['CONDUCTORES']
    puntos_apoyos = data[vano]['LIDAR']['APOYOS']
    extremos_apoyos = data[vano]['APOYOS']

    # Extrae las coordenadas x, y, z de los conductores
    x_vals_conductores, y_vals_conductores, z_vals_conductores = get_coord(puntos_conductores)
    x_vals_apoyos, y_vals_apoyos, z_vals_apoyos = get_coord(puntos_apoyos)
    x_vals_extremos, y_vals_extremos, z_vals_extremos = get_coord2(extremos_apoyos)

    cond_values = [x_vals_conductores, y_vals_conductores, z_vals_conductores]
    apoyo_values = [x_vals_apoyos, y_vals_apoyos, z_vals_apoyos]
    extremos_values = [x_vals_extremos, y_vals_extremos, z_vals_extremos]

    vert_values = []

    for element in data[vano]['CONDUCTORES']:
        vert_values.append(get_coord(element['VERTICES']))

    return cond_values, apoyo_values, vert_values, extremos_values

def add_plot(fig, data, color, size, name, mode):

    fig.add_trace(go.Scatter3d(
        x=data[0],
        y=data[1],
        z=data[2],
        mode=mode,
        marker=dict(
            size=size,
            color=color,  # Color de los apoyos
        ),
        name=name  # Nombre de la traza de los apoyos
    ))

def plot_fit(title,cond_values, apoyo_values, vert_values,fit,crossesa,crossesb):

    # Crea el gráfico para los conductores
    fig = go.Figure(data=[go.Scatter3d(
        x=fit[0],
        y=fit[1],
        z=fit[2],
        mode='markers',
        marker=dict(
            size=2.5,
            color='green',  # Color de los conductores
        ),
        name='Conductores'  # Nombre de la traza de los conductores
    )])

    # Agrega el gráfico para el fit
    add_plot(fig, cond_values, "blue", 2.5, "Apoyos", "markers")
    # Agrega el gráfico para los apoyos
    add_plot(fig, apoyo_values, "orange", 2.5, "Apoyos", "markers")
    add_plot(fig, crossesa, "purple", 5, "Apoyos", "markers")
    add_plot(fig, crossesb, "purple", 5, "Apoyos", "markers")

    for vert in vert_values:

        # Agrega el gráfico para los vertices
        add_plot(fig, vert , "red", 5, "Vertices", "lines")

    # Agrega títulos a los ejes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=0, b=0, l=0, t=0)  # Reduce el margen alrededor del gráfico
    )

    # Muestra el gráfico
    fig.show()

def plot_fit_2(title,cond_values, apoyo_values, vert_values,fit):

    # Crea el gráfico para los conductores
    fig = go.Figure(data=[go.Scatter3d(
        x=fit[0],
        y=fit[1],
        z=fit[2],
        mode='markers',
        marker=dict(
            size=2.5,
            color='green',  # Color de los conductores
        ),
        name='Conductores'  # Nombre de la traza de los conductores
    )])

    # Agrega el gráfico para el fit
    add_plot(fig, cond_values, "blue", 2.5, "Apoyos", "markers")
    # Agrega el gráfico para los apoyos
    add_plot(fig, apoyo_values, "orange", 2.5, "Apoyos", "markers")

    for vert in vert_values:

        # Agrega el gráfico para los vertices
        add_plot(fig, vert , "red", 5, "Vertices", "lines")

    # Agrega títulos a los ejes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=0, b=0, l=0, t=0)  # Reduce el margen alrededor del gráfico
    )

    # Muestra el gráfico
    fig.show()

def plot_data(title,cond_values, apoyo_values, vert_values, extremos_values):

    # Crea el gráfico para los conductores
    fig = go.Figure(data=[go.Scatter3d(
        x=cond_values[0],
        y=cond_values[1],
        z=cond_values[2],
        mode='markers',
        marker=dict(
            size=2.5,
            color='blue',  # Color de los conductores
        ),
        name='Conductores'  # Nombre de la traza de los conductores
    )])

    # Agrega el gráfico para los apoyos
    add_plot(fig, apoyo_values, "orange", 2.5, "Apoyos", "markers")

    # Agrega el gráfico para los extremos
    add_plot(fig, extremos_values, "black", 5, "Extremos", "markers")


    for vert in vert_values:

        # Agrega el gráfico para los vertices
        add_plot(fig, vert , "red", 5, "Vertices", "lines")

    # Agrega títulos a los ejes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=0, b=0, l=0, t=0)  # Reduce el margen alrededor del gráfico
    )

    # Muestra el gráfico
    fig.show()


def get_distances(extremos_values):

    x_apoyos_distance = abs(extremos_values[0][0] - extremos_values[0][2])
    y_apoyos_distance = abs(extremos_values[1][0] - extremos_values[1][2])
    xz_distance = np.sqrt((extremos_values[0][0] - extremos_values[0][2])**2 + (extremos_values[2][0] - extremos_values[2][2])**2)
    xy_distance = np.sqrt((extremos_values[0][0] - extremos_values[0][2])**2 + (extremos_values[1][0] - extremos_values[1][2])**2)
    yz_distance = np.sqrt((extremos_values[1][0] - extremos_values[1][2])**2 + (extremos_values[2][0] - extremos_values[2][2])**2)

    D3_apoyos_distance = np.sqrt((extremos_values[0][0] - extremos_values[0][2])**2 + (extremos_values[1][0] - extremos_values[1][2])**2 + (extremos_values[2][0] - extremos_values[2][2])**2)

    return x_apoyos_distance, y_apoyos_distance, xz_distance, xy_distance, yz_distance, D3_apoyos_distance

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

def un_rotate_points(points, matriz_rotacion):

    points = np.array(points).T
    # Crear la matriz de rotación para alinear la diagonal con el eje Y
    matriz_rotacion = matriz_rotacion.T

    rotated_points = matriz_rotacion.dot(points.T)
    # print(rotated.shape)

    return matriz_rotacion, np.array(rotated_points)

def rotate_vano(cond_values, extremos_values, apoyo_values, vert_values):

    # Rotate and compare
    mat, rotated_conds = rotate_points(cond_values, extremos_values)

    rotated_apoyos = mat.dot(apoyo_values)
    rotated_extremos = mat.dot(extremos_values)
    rotated_vertices = [mat.dot(vert) for vert in vert_values]

    return mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos


def clean_outliers(rotated_conds, rotated_extremos):

    # Get top and bottom extreme values
    # top = np.max([rotated_extremos.T[1][2],rotated_extremos.T[3][2]])
    # bottom = np.max([rotated_extremos.T[0][2],rotated_extremos.T[2][2]])

    #Get left and right extreme values
    left = np.max([rotated_extremos.T[2][1],rotated_extremos.T[3][1]])
    right = np.min([rotated_extremos.T[0][1],rotated_extremos.T[1][1]])

    # Filter points within the specified boundaries
    cropped_conds = rotated_conds[:, (right > rotated_conds[1,:]) & (rotated_conds[1,:] > left)]
    # cropped_conds = cropped_conds[:, (top > cropped_conds[2,:]) & (cropped_conds[2,:] > bottom)]

    # Paso 1: Calcular el histograma de las coordenadas Y
    hist, bin_edges = np.histogram(cropped_conds[1, :], bins=200)

    # Paso 2: Identificar los picos significativos en ambos extremos del histograma
    # Definir un umbral para considerar un pico significativo
    threshold_density = np.mean(hist) + 2 * np.std(hist)

    # Encontrar el bin con la mayor cantidad de puntos en la parte superior
    peak_bin_upper = np.argmax(hist[:len(hist)//2])
    # Encontrar el bin con la mayor cantidad de puntos en la parte inferior
    peak_bin_lower = np.argmax(hist[len(hist)//2:]) + len(hist)//2

    # Inicializar los umbrales
    threshold_y_upper = None
    threshold_y_lower = None

    # print(hist[peak_bin_upper], hist[peak_bin_lower])
    # plt.hist(cropped_conds[1, :])

    # Verificar si hay una línea horizontal significativa en la parte superior
    if hist[peak_bin_upper] > threshold_density:
        threshold_y_upper = bin_edges[peak_bin_upper + 1]  # El +1 es para obtener el borde superior del bin
        print(f"Umbral de corte inferior detectado: {threshold_y_upper}")

    # Verificar si hay una línea horizontal significativa en la parte inferior
    if hist[peak_bin_lower] > threshold_density:
        threshold_y_lower = bin_edges[peak_bin_lower]  # No se necesita ajustar más
        print(f"Umbral de corte superior detectado: {threshold_y_lower}")

    # Paso 3: Filtrar los puntos usando los umbrales detectados
    if threshold_y_upper is not None:
        cropped_conds = cropped_conds[:, cropped_conds[1, :] > threshold_y_upper]

    if threshold_y_lower is not None:
        cropped_conds = cropped_conds[:, cropped_conds[1, :] < threshold_y_lower]

    # Calcular percentiles 1 y 99
    p1 = np.percentile(cropped_conds[1, :], 2)
    p99 = np.percentile(cropped_conds[1, :], 98)

    # Filtrar los datos para eliminar el 5% de los puntos con menor y mayor coordenada Y
    cropped_conds = cropped_conds[:,(cropped_conds[1, :] > p1) & (cropped_conds[1, :] < p99)]

    # Erase X axis outliers
    p1 = np.percentile(cropped_conds[0, :], 2)
    p99 = np.percentile(cropped_conds[0, :], 98)

    cropped_conds = cropped_conds[:,(cropped_conds[0, :] > p1) & (cropped_conds[0, :] < p99)]

    return cropped_conds


def clean_outliers_2(rotated_conds):

    lx=pd.Series(rotated_conds[0,:]).quantile(0.25)
    ux=pd.Series(rotated_conds[0,:]).quantile(0.75)
    ly=pd.Series(rotated_conds[1,:]).quantile(0.25)
    uy=pd.Series(rotated_conds[1,:]).quantile(0.75)
    lz=pd.Series(rotated_conds[2,:]).quantile(0.25)
    uz=pd.Series(rotated_conds[2,:]).quantile(0.75)
    rotated_conds=rotated_conds[:,(rotated_conds[0,:]>lx-1.5*(ux-lx))&(rotated_conds[0,:]<ux+1.5*(ux-lx))]
    rotated_conds=rotated_conds[:,(rotated_conds[1,:]>ly-1.5*(uy-ly))&(rotated_conds[1,:]<uy+1.5*(uy-ly))]
    rotated_conds=rotated_conds[:,(rotated_conds[2,:]>lz-1.5*(uz-lz))&(rotated_conds[2,:]<uz+1.5*(uz-lz))]

    return rotated_conds


def clean_outliers_3(cropped_conds):
    nn = 10 # Local search
    std_multip = 1 # Not very sensitive

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.array(cropped_conds).T)
    cl, ind = pcd_o3d.remove_statistical_outlier(nb_neighbors=nn, std_ratio=std_multip)
    inlier_cloud = pcd_o3d.select_by_index(ind)

    cropped_conds = np.asarray(inlier_cloud.points).T
    return cropped_conds


def clean_outliers_4(cropped_conds):
    nn = 10 # Local search
    radius = 1 # Not very sensitive

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.array(cropped_conds).T)
    cl, ind = pcd_o3d.remove_radius_outlier(nb_points=nn, radius=radius)
    inlier_cloud = pcd_o3d.select_by_index(ind)

    cropped_conds = np.asarray(inlier_cloud.points).T
    return cropped_conds


def scale_conductor(X):

    # Normalizzazione dei valori di x e y
    scaler_y = StandardScaler()
    scaler_x = StandardScaler()
    scaler_z = StandardScaler()

    y_vals_scaled = scaler_y.fit_transform(X[1,:].reshape(-1, 1)).flatten()
    x_vals_scaled = scaler_x.fit_transform(X[0,:].reshape(-1, 1)).flatten()  # Flatten per curve_fit
    z_vals_scaled = scaler_z.fit_transform(X[2,:].reshape(-1, 1)).flatten()

    X_scaled = np.array([x_vals_scaled, y_vals_scaled, z_vals_scaled])

    return X_scaled,scaler_x,scaler_y,scaler_z



def un_scale_conductor(X,scaler_x,scaler_y,scaler_z):

    y_vals_unscaled = scaler_y.inverse_transform(X[1,:].reshape(-1, 1)).flatten()
    x_vals_unscaled = scaler_x.inverse_transform(X[0,:].reshape(-1, 1)).flatten()  # Flatten per curve_fit
    z_vals_unscaled = scaler_z.inverse_transform(X[2,:].reshape(-1, 1)).flatten()

    X_unscaled = np.array([x_vals_unscaled, y_vals_unscaled, z_vals_unscaled])

    return X_unscaled


def distance(pt1, pt2):

    x1, y1, z1 = pt1
    x2, y2, z2 = pt2

    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    return distancia



def initialize_centroids(points, n_clusters):

    centroids = np.zeros((n_clusters))

    centroids[0] = np.min(points[0,:])
    centroids[1] = np.mean(points[0,:])
    centroids[2] = np.max(points[0,:])

    return centroids

def assign_clusters(points, centroids):
    distances = np.abs(points[0][:, None] - centroids)
    return np.argmin(distances, axis=1)

def update_centroids(points, labels, n_clusters):
    new_centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = points[0, labels == i]
        if cluster_points.size > 0:
            new_centroids[i] = np.mean(cluster_points)
    return new_centroids

def kmeans_clustering(points, n_clusters, max_iterations):
    centroids = initialize_centroids(points, n_clusters)
    for iteration in range(max_iterations):
        labels = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, labels, n_clusters)
        if np.allclose(new_centroids, centroids):
            # print(f"Convergence reached at iteration {iteration}")
            break
        centroids = new_centroids

    print(labels.shape)
    print(np.unique(labels))
    print(centroids.shape)
    return labels, centroids


def spectral_clustering(points, n_clusters, n_init):

    n_samples = points.shape[1]
    n_neighbors = max(2, n_samples // 2)
    n_neighbors = min(n_neighbors, n_samples)
    affinity_matrix = kneighbors_graph(points.T, n_neighbors=n_neighbors, include_self=True)

    # model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=n_init, random_state=0)
    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    labels = model.fit_predict(points.T)
    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_points = points.T[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    centroids=[point.tolist() for point in centroids]
    return labels, centroids

def catenaria(x, a, h, k):
    x = np.asarray(x).flatten()
    r=a * np.cosh((x - h) / a) + k
    return r

def invert_linear_model(y_val, slope, intercept):
    return (y_val - intercept) / slope

def flatten_sublist(sublist):
    flat_list = [sublist[0]]
    for array in sublist[1:]:
        print(array)
        flat_list.extend(array.tolist())
    return flat_list

def flatten_sublist_2(sublist):
    flat_list = [sublist[0]]
    for array in sublist[1:-2]:
        flat_list.extend(array.tolist())
    flat_list.extend([sublist[-2]])
    flat_list.extend([sublist[-1]])
    return flat_list

def fit_data_parameters(data,sublist=[]):

    parameters=[]
    labelsw=[]
    idvw=[]
    non_fitting = []
    for i in range(len(data)):
        print(f"\nProcessing Vano {i}")

        idv=data[i]['ID_VANO']
        if idv in sublist:

            cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)

            cond_values=np.vstack(cond_values)
            apoyo_values=np.vstack(apoyo_values)

            if np.array(extremos_values).shape[1]==4:

                rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)

                cropped_conds = clean_outliers(rotated_conds, rotated_extremos)

                X_scaled = scale_conductor(cropped_conds)

                labels, centroids = kmeans_clustering(X_scaled, n_clusters=3, max_iterations=1000)

                total_points = X_scaled.shape[1]

                parameters_vano=[]
                labelsc=[]
                for lab in np.unique(labels):
                    idl=idv+'_'+str(lab)
                    labelsc.append(lab)

                    clust = X_scaled[:,labels == lab]
                    proportion = clust.shape[1]/total_points

                    if proportion< 0.15:
                        if idl not in non_fitting:
                            non_fitting.append(str(idl))
                        print(f"Error en el cluster {idl}")
                        print(f"This cluster represents: {round(100*proportion,2)}%")
                    else:
                        y_vals = np.append(clust[1].reshape(-1, 1),data[i]['APOYOS'][0]['COORDEANDA_Y'])
                        z_vals = np.append(clust[2].reshape(-1, 1),data[i]['APOYOS'][0]['COORDEANDA_Y'])
                        initial_params = [1, 0, 0]  # a, h, k
                        try:
                            optim_params, _ = curve_fit(catenaria, y_vals.flatten(), z_vals.flatten(), p0=initial_params, method = 'lm')
                            fitted_z = catenaria(y_vals.flatten(), *optim_params)
                            parameters_vano.append(optim_params)
                        except:
                            if str(idl) not in non_fitting:
                                non_fitting.append(str(idl))

            else:
                for el in [0,1,2]:
                    non_fitting.append(idv+'_'+str(el))

            if idv not in non_fitting:
                 labelsw=labelsw+labelsc
                 parameters=parameters+parameters_vano

    columns = ['ID','a', 'h', 'k']
    parameters = pd.DataFrame(parameters, columns=columns)

    return parameters,non_fitting


def fit_vano_group(data,sublist=[]):

    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    for i in range(len(data)):

        print(f"\nProcessing Vano {i}")

        idv=data[i]['ID_VANO']

        if idv in sublist:

            cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)

            cond_values=np.vstack(cond_values)
            apoyo_values=np.vstack(apoyo_values)

            if np.array(extremos_values).shape[1]==4:
                y=((data[i]['APOYOS'][0]['COORDEANDA_Y'] + data[i]['APOYOS'][1]['COORDEANDA_Y']) / 2)
                x=((data[i]['APOYOS'][0]['COORDENADA_X'] + data[i]['APOYOS'][1]['COORDENADA_X']) / 2)
                rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)

                cropped_conds = clean_outliers(rotated_conds, rotated_extremos)

                X_scaled = scale_conductor(cropped_conds)

                labels, centroids = kmeans_clustering(X_scaled, n_clusters=3, max_iterations=1000)

                total_points = X_scaled.shape[1]

                parameters_vano=[]
                for lab in np.unique(labels):

                    idl=idv+'_'+str(lab)
                    clust = X_scaled[:,labels == lab]
                    proportion = clust.shape[1]/total_points

                    if proportion< 0.15:
                        if idv not in incomplete_vanos:
                            incomplete_vanos.append(idv)
                        incomplete_lines.append(idl)
                        print(f"Error en el cluster {idv}")
                        print(f"This cluster represents: {round(100*proportion,2)}%")
                    else:
                        y_vals = clust[1].reshape(-1, 1)
                        z_vals = clust[2].reshape(-1, 1)
                        initial_params = [1, 0, 0]  # a, h, k
                        try:
                            optim_params, _ = curve_fit(catenaria, y_vals.flatten(), z_vals.flatten(), p0=initial_params, method = 'lm')
                            fitted_z = catenaria(y_vals.flatten(), *optim_params)
                            if idv not in incomplete_vanos:
                                parameters_vano.append(optim_params)
                        except:
                            if idv not in incomplete_vanos:
                                incomplete_vanos.append(idv)
                            incomplete_lines.append(idl)
            else:
                incomplete_vanos.append(idv)
                for el in [0,1,2]:
                    incomplete_lines.append(idv+'_'+str(el))

            if idv not in incomplete_vanos:
                parameters.append([idv]+parameters_vano+[x,y])

    return parameters,incomplete_vanos,incomplete_lines


def group_dbscan_4(k,X_scaled):

    neighbors = NearestNeighbors(n_neighbors=int(k))
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, int(k)-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1

    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=int(k), algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_points = X_scaled[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    centroids=[point.tolist() for point in centroids]
    return centroids, labels

def dbscan_find_clusters_4(X_scaled):

    ar=[3,5,7,10,15,20,35,50]
    matching_clust=[]
    for k in ar:
        
        centroids,labels=group_dbscan_4(k,X_scaled.T)
        if len(np.unique(labels))!=3:
            score=-1

        elif len(np.unique(labels))==3:
            score=metrics.silhouette_score(X_scaled.T,labels)

        if k==np.array(ar).min():
            best_score=score
            best_k=k

        elif score>best_score:
            best_score=score
            best_k=k

    centroids,labels=group_dbscan_3(best_k,X_scaled.T)

    return centroids,labels

def group_dbscan_3(k,X_scaled):

    neighbors = NearestNeighbors(n_neighbors=int(k))
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, int(k)-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1

    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=int(k), algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_points = X_scaled[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    centroids=[point.tolist() for point in centroids]

    # centroids=flatten_sublist(centroids)
    return centroids, labels


def dbscan_find_clusters_3(X_scaled):

    ar=[5,10,20]
    for k in ar:
        
        centroids,labels=group_dbscan_3(k,X_scaled.T)
        if len(np.unique(labels))==1:
            score=-1
        else:
            score=metrics.silhouette_score(X_scaled.T,labels)

        if k==np.array(ar).min():
            best_score=score
            best_k=k
        elif score>best_score:
            best_score=score
            best_k=k

    centroids,labels=group_dbscan_3(best_k,X_scaled.T)

    return centroids,labels


def plot_vano(title,X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values):

    if len(labels)!=0:
        plt.scatter(X_scaled.T[:, 0], X_scaled.T[:, 1], c=labels, cmap='viridis', label = labels)
        plt.title('Clustering con kmeans')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.tight_layout()
        plt.title(title)
        plt.show()

        plt.scatter( X_scaled.T[:, 1], X_scaled.T[:, 2],c=labels, cmap='viridis', label = labels)
        plt.title('Clustering con kmeans')
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.tight_layout()
        plt.title(title)
        plt.show()

    plot_data(title,cond_values, apoyo_values, vert_values, extremos_values)

def define_backings(vano_length,apoyo_values):
    #args: vano_length, and apoyo_values
    #returns: a coordinate list like [array([119842.5432, 119842.5432, 119934.9426, 119934.9426]), array([4695380.2077, 4695380.2077, 4695375.6154, 4695375.6154]), array([949.0614, 958.8735, 987.3561, 997.7533])]
    points = np.array(apoyo_values)
    print(points.shape)

    kmeans = KMeans(n_clusters=2, max_iter=500, n_init="auto").fit(points.T)
    labels=kmeans.labels_
    extremos = []
    # print(f"Distance between centroids: {abs(centroids[0] - centroids[1])}")

    for lab in np.unique(labels):

        apoyo = points[:, labels == lab]

        mean_x = np.mean(apoyo[0,:])
        mean_y = np.mean(apoyo[1,:])
        mean_z = np.mean(apoyo[2,:])

        c_mass = np.array([mean_x, mean_y, mean_z])
        extremos.append(c_mass)

    dist = np.linalg.norm(np.array(extremos)[0,:] - np.array(extremos)[1,:])
    extremos = np.array(extremos).T

    extremos_values=[]
    for ael in extremos:
        l=[]
        for bel in ael:
            l=l+[bel,bel]
        res=np.array(l)
        extremos_values.append(res)

    if 100*abs(dist - vano_length)/vano_length > 10.0:
        extremos_values=-1

    return extremos_values


def clpt_to_array(cl_pt):
    rfx=[]
    rfy=[]
    rfz=[]
    print(cl_pt)
    for el in cl_pt:
        rfx.append(el[0])
        rfy.append(el[1])
        rfz.append(el[2])
    return np.array([rfx,rfy,rfz])



def fit_plot_vano_group(data,sublist=[],plot_filter=None,init=0,end=None,save=False,label=''):
    #plot_filter= "bad_backing", bad_cluster, bad_line_number, bad_line_orientation, bad_fit, good_fit, empty

    if len(sublist)==0:
        sublist=[data[i]['ID_VANO'] for i in range(len(data))]
    end=len(sublist) if end!=None else end
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    dataf = {'id': [], 'flag': [],'line_number':[]}
    for i in range(len(data)):

        if all([i>=init,i<=end]):

            idv=data[i]['ID_VANO']
            vano_length=data[i]["LONGITUD_2D"]
            dataf['id'].append(idv)
            
            if idv in sublist:

                print(f"\nProcessing Vano {i}")
                print(f"\nReference {idv}")
                cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
                cond_values=np.vstack(cond_values)
                apoyo_values=np.vstack(apoyo_values)

                # tree = KDTree(cond_values.T)
                # distances, indices = tree.query(apoyo_values.T)
                # min_index = np.argmin(distances)
                # print(indices[min_index])
                # cl_pt=cond_values[:,indices[min_index]]
                # print(cl_pt)

                X_scaled=np.array([])
                labels=np.array([])

                data[i]['CONDUCTORES_CORREGIDOS']={}
                data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)']={}

                if np.array(extremos_values).shape[1]!=4:

                    dataf['flag'].append('bad_backing')
                    dataf['line_number'].append(0)
                    if any([plot_filter=='all',plot_filter=='bad_backing']):
                        plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    # print(i+1)
                    # print(len(dataf['flag']))
                    # print(dataf['flag'][-1])
                    continue

                # print('bad_backing2')

                    # print('bad_backing')
                    # extremos_values=define_backings(vano_length,apoyo_values)
                    # print(extremos_values)

                    # if extremos_values==-1:
                    #     if any([plot_filter=='all',plot_filter=='bad_backing']):
                    #         plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    #     continue

                # y=((data[i]['APOYOS'][0]['COORDEANDA_Y'] + data[i]['APOYOS'][1]['COORDEANDA_Y']) / 2)
                # x=((data[i]['APOYOS'][0]['COORDENADA_X'] + data[i]['APOYOS'][1]['COORDENADA_X']) / 2)

                mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
                rotated_ymin=min(rotated_conds[1])
                rotated_ymax=max(rotated_conds[1])
                cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
                cropped_conds= clean_outliers_2(rotated_conds)
                cropped_conds =clean_outliers_3(cropped_conds)

                # outliers = pcd_o3d.select_by_index(filtered_points[1], invert=True)

                X_scaled,scaler_x,scaler_y,scaler_z = scale_conductor(cropped_conds)
                
                rotated_ymax=scaler_y.transform(np.array([rotated_ymax]).reshape(-1, 1))[0]
                rotated_ymin=scaler_y.transform(np.array([rotated_ymin]).reshape(-1, 1))[0]
    
                # Find clusters in 10 cloud of points corresponding to 3 positions in y axis

                maxy=max(rotated_extremos[1,1],rotated_extremos[1,2])
                miny=min(rotated_extremos[1,1],rotated_extremos[1,2])
                leny=maxy-miny

                l=[]
                filt=[]
                k=10

                for g in range(0,k):

                    filt0=(cropped_conds[1,:]>(miny+g*(1/k)*leny))&(cropped_conds[1,:]<(miny+(g+1)*(1/k)*leny))
                    l0=cropped_conds[:,filt0].shape[1]
                    l.append(l0)
                    filt.append(filt0)

                c=[]
                greater_var_z_than_x=[]
                ncl=[]

                for g in range(0,k):

                    l0=X_scaled[:,filt[g]].shape[1]
                    fl=pd.Series(X_scaled[2,filt[g]]).var()-pd.Series(X_scaled[0,filt[g]]).var()
                    centroids0,labels0=dbscan_find_clusters_3(X_scaled[:,filt[g]]) if l0>20 else ([],[])
                    greater_var_z_than_x.append(True if fl>=0 else False)
                    c.append(centroids0)
                    ncl.append(len(centroids0))

                md=mode(ncl)
                var_z_x=mode(greater_var_z_than_x)
                num_empty=np.array([l0<20 for l0 in l]).sum()
                completeness=np.array(['incomplete','partially incomplete','full'])
                completeness_conds=np.array([num_empty>5,all([num_empty<=5,num_empty>=2]),num_empty<2])
                finc= completeness[completeness_conds]

                fits=[]
                crossesa= []
                crossesb= []

                dataf['line_number'].append(md)
                print(f'Number of lines: {md}')

                if finc[0]=='incomplete':
                    dataf['flag'].append('empty')
                    if any([plot_filter=='all',plot_filter=='empty']):
                        plot_vano('{} Empty{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)

                elif md!=3:
                    dataf['flag'].append('bad_line_number')
                    if any([plot_filter=='all',plot_filter=='bad_line_number']):
                        plot_vano('{} Bad_Line_Number{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)

                else:
                    # X_scaled=X_scaled[:,np.logical_or.reduce(filt[1:-1])]
                    # centroids,labels=dbscan_find_clusters_4(X_scaled)
                    # labels,centroids=spectral_clustering(X_scaled,n_clusters=3,n_init=100)
                    labels, centroids = kmeans_clustering(X_scaled, n_clusters=3, max_iterations=1000)

                    total_points = X_scaled.shape[1]

                    parameters_vano=[]
                    count_good=0
                    bad_cluster=0
                    bad_fit=0
                    crl=[]
                    
                    if len(np.unique(labels))<3:
                        bad_cluster=1

                    for lab in np.unique(labels):

                        idl=idv+'_'+str(lab)
                        clust = X_scaled[:,labels == lab]
                        # corr=np.corrcoef(clust[:1,:])
                        # crl.append(corr)
                        proportion = clust.shape[1]/total_points
                        
                        if proportion< 0.15:
                            bad_cluster=1

                        else:
                            y_vals = clust[1,:].reshape(-1, 1)
                            z_vals = clust[2,:].reshape(-1, 1)
                            y_mean = np.mean(y_vals)
                            y_range = np.max(y_vals) - np.min(y_vals)
                            meanp=min(rotated_apoyos[1])+(max(rotated_apoyos[1])-min(rotated_apoyos[1]))/2
                            
                            initial_params = [1,0,0] # a, h, k
                            
                            try:
                                    
                                optim_params, _ = curve_fit(catenaria, y_vals.flatten(), z_vals.flatten())#,p0=initial_params,method = 'trf'

                                y_fit=np.linspace(rotated_ymin,rotated_ymax,1000).flatten()

                                z_fit = catenaria(y_fit, *optim_params)


                                # coefficients = np.polyfit(x_vals.flatten(), y_vals.flatten(), 1)
                                # linear_fit = np.poly1d(coefficients)
                                # slope, intercept=coefficients
                                # x_fit = invert_linear_model(y_fit, slope, intercept)
                                # ****model = LinearRegression()
                                # ****model.fit(x_vals, y_vals)
                                # ****slope = model.coef_[0][0]
                                # ****intercept = model.intercept_[0]
                                # ****x_fit = invert_linear_model(y_fit, slope, intercept)

                                x_fit = np.repeat(pd.Series(clust[0,:]).quantile(0.5),1000)
                                fit=np.vstack((x_fit, y_fit,z_fit))

                                fit=un_scale_conductor(fit,scaler_x,scaler_y,scaler_z)
                                # optim_params, _ = curve_fit(catenaria, fit[1,:].flatten(), fit[2,:].flatten(), p0=initial_params, method = 'trf')

                                apoyo_values_a=rotated_apoyos[:,rotated_apoyos[1,:]<(meanp)]
                                apoyo_values_b=rotated_apoyos[:,rotated_apoyos[1,:]>(meanp)]
                                
                                tree = KDTree(fit.T)
                                distancesa, indicesa = tree.query(apoyo_values_a.T)
                                min_index = np.argmin(distancesa)
                                cl_pta=fit[:,indicesa[min_index]]
                                
                                distancesb, indicesb = tree.query(apoyo_values_b.T)
                                min_index = np.argmin(distancesb)
                                cl_ptb=fit[:,indicesb[min_index]]
            
                                fit=fit[:,(fit[1,:]>cl_pta[1])&(fit[1,:]<cl_ptb[1])]
                                
                                # cl_pta=clpt_to_array(cl_pta)
                                # cl_ptb=clpt_to_array(cl_ptb)

                                y_fit=np.linspace(cl_pta[1],cl_ptb[1],1000).flatten()
                                y_fit=scaler_y.transform(y_fit.reshape(-1,1))
                                

                                # coefficients = np.polyfit(x_vals.flatten(), y_vals.flatten(), 1)
                                # linear_fit = np.poly1d(coefficients)
                                # slope, intercept=coefficients
                                # x_fit = invert_linear_model(y_fit.flatten(), slope, intercept)
                                # ****x_fit = invert_linear_model(y_fit, slope, intercept)

                                z_fit = catenaria(y_fit, *optim_params)
                                x_fit = np.repeat(pd.Series(clust[0,:]).quantile(0.5),1000)
                                fit=np.vstack((x_fit.flatten(), y_fit.flatten(),z_fit.flatten()))
                                
                                fit=un_scale_conductor(fit,scaler_x,scaler_y,scaler_z)
                                
                                mat_neg,cl_pta=un_rotate_points(cl_pta,mat)
                                mat_neg,cl_ptb=un_rotate_points(cl_ptb,mat)
                                mat_neg,fit=un_rotate_points(fit,mat)
                                
                                fits.append(fit)
                                crossesa.append(cl_pta)
                                crossesb.append(cl_ptb)
                                data[i]['CONDUCTORES_CORREGIDOS'][str(lab)]=fit.T.tolist()

                                # data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'][str(lab)]=optim_params.tolist()
                                
                                # print(data[i]['CONDUCTOR_CORREGIDO'][lab])
                                # print(data[i]['CONDUCTOR_CORREGIDO_PARAMETROS_(a,h,k)'][lab])
                                # print(indices[min_index])
                                # if corr>0.8:
                                count_good=count_good+1
                                # else:
                                #     bad_cluster=1
                                
                            except:
                                bad_fit=1

                    fits = np.hstack(fits)
                    # print(fits)
                    # print(fits.shape)
                    # print(cond_values)
                    crossesa=np.vstack(crossesa).T#(crossesa[0],crossesa[1],crossesa[2])
                    # print(crossesa)
                    crossesb=np.vstack(crossesb).T#(crossesb[0],crossesb[1],crossesb[2])
                    # print(crossesb)

                    # print(f"{bad_cluster =}")
                    # print(f"{bad_fit =}")
                    # print(f"{count_good =}")

                    if bad_cluster==1:
                        if all([md==3,var_z_x]):
                            dataf['flag'].append('bad_line_orientation')
                            if any([plot_filter=='all',plot_filter=='bad_line_orientation']):
                                plot_vano('{} Bad_Orientation{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        else:
                            dataf['flag'].append('bad_cluster')
                            if any([plot_filter=='bad_cluster',plot_filter=='all']):
                                plot_vano('{} Incomplete_cluster{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    elif bad_fit==1:
                        dataf['flag'].append('bad_fit')
                        if any([plot_filter=='bad_fit',plot_filter=='all']):
                            plot_vano('{} Bad_fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    elif count_good==3:
                        dataf['flag'].append('good_fit')
                        if any([plot_filter=='good_fit',plot_filter=='all']):
                            # plot_vano('{} Good_Fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                            plot_fit('{} Good_Fit{}'.format(idv,' '+finc[0]),cond_values, apoyo_values, vert_values,fits,crossesa,crossesb)

                            # plot_fit(f'{idv}',cond_values, apoyo_values, vert_values,fit)
    #             print(i+1)
    #             print(len(dataf['flag']))
    #             print(dataf['flag'][-1])
    # print(len(dataf['line_number']))
    # print(len(dataf['id']))
    # print(len(dataf['flag']))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if save==True:
        with open(timestamp+'_'+label+'_resultado.json', 'w') as file:
            json.dump(data, file)

    datafr=pd.DataFrame(dataf)
    return datafr


def fit_plot_vano_group_2(data,sublist=[],plot_filter=None,init=0,end=None,save=False,label=''):
    #filter= "bad_backing", bad_cluster, bad_line_number, bad_line_orientation, bad_fit, good_fit, empty

    if len(sublist)==0:
        sublist=[data[i]['ID_VANO'] for i in range(len(data))]
    end=len(sublist) if end!=None else end
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    dataf = {'id': [], 'flag': [],'line_number':[]}
    for i in range(len(data)):

        if all([i>=init,i<=end]):

            idv=data[i]['ID_VANO']
            vano_length=data[i]["LONGITUD_2D"]
            dataf['id'].append(idv)
            
            if idv in sublist:

                print(f"\nProcessing Vano {i}")
                print(f"\nReference {idv}")
                cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
                cond_values=np.vstack(cond_values)
                apoyo_values=np.vstack(apoyo_values)

                # tree = KDTree(cond_values.T)
                # distances, indices = tree.query(apoyo_values.T)
                # min_index = np.argmin(distances)
                # print(indices[min_index])
                # cl_pt=cond_values[:,indices[min_index]]
                # print(cl_pt)

                X_scaled=np.array([])
                labels=np.array([])

                data[i]['CONDUCTORES_CORREGIDOS']={}
                data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)']={}
                data[i]['PUNTUACIONES']={}

                if np.array(extremos_values).shape[1]!=4:

                    dataf['flag'].append('bad_backing')
                    dataf['line_number'].append(0)
                    if any([plot_filter=='all',plot_filter=='bad_backing']):
                        plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    # print(i+1)
                    # print(len(dataf['flag']))
                    # print(dataf['flag'][-1])
                    continue

                # print('bad_backing2')

                    # print('bad_backing')
                    # extremos_values=define_backings(vano_length,apoyo_values)
                    # print(extremos_values)

                    # if extremos_values==-1:
                    #     if any([plot_filter=='all',plot_filter=='bad_backing']):
                    #         plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    #     continue

                # y=((data[i]['APOYOS'][0]['COORDEANDA_Y'] + data[i]['APOYOS'][1]['COORDEANDA_Y']) / 2)
                # x=((data[i]['APOYOS'][0]['COORDENADA_X'] + data[i]['APOYOS'][1]['COORDENADA_X']) / 2)

                mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
                rotated_ymin=min(rotated_conds[1])
                rotated_ymax=max(rotated_conds[1])
                cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
                cropped_conds= clean_outliers_2(rotated_conds)
                cropped_conds =clean_outliers_3(cropped_conds)

                # outliers = pcd_o3d.select_by_index(filtered_points[1], invert=True)

                X_scaled,scaler_x,scaler_y,scaler_z = scale_conductor(cropped_conds)
                
                rotated_ymax=scaler_y.transform(np.array([rotated_ymax]).reshape(-1, 1))[0]
                rotated_ymin=scaler_y.transform(np.array([rotated_ymin]).reshape(-1, 1))[0]
    
                # Find clusters in 10 cloud of points corresponding to 3 positions in y axis

                maxy=max(rotated_extremos[1,1],rotated_extremos[1,2])
                miny=min(rotated_extremos[1,1],rotated_extremos[1,2])
                leny=maxy-miny

                l=[]
                filt=[]
                k=10

                for g in range(0,k):

                    filt0=(cropped_conds[1,:]>(miny+g*(1/k)*leny))&(cropped_conds[1,:]<(miny+(g+1)*(1/k)*leny))
                    l0=cropped_conds[:,filt0].shape[1]
                    l.append(l0)
                    filt.append(filt0)

                c=[]
                greater_var_z_than_x=[]
                ncl=[]

                for g in range(0,k):

                    l0=X_scaled[:,filt[g]].shape[1]
                    fl=pd.Series(X_scaled[2,filt[g]]).var()-pd.Series(X_scaled[0,filt[g]]).var()
                    centroids0,labels0=dbscan_find_clusters_3(X_scaled[:,filt[g]]) if l0>20 else ([],[])
                    greater_var_z_than_x.append(True if fl>=0 else False)
                    c.append(centroids0)
                    ncl.append(len(centroids0))

                md=mode(ncl)
                var_z_x=mode(greater_var_z_than_x)
                num_empty=np.array([l0<20 for l0 in l]).sum()
                completeness=np.array(['incomplete','partially incomplete','full'])
                completeness_conds=np.array([num_empty>5,all([num_empty<=5,num_empty>=2]),num_empty<2])
                finc= completeness[completeness_conds]

                fits=[]
                crossesa= []
                crossesb= []
                bad_fit=0
                bad_cluster=0
                good=0

                dataf['line_number'].append(md)
                print(f'Number of lines: {md}')

                if finc[0]=='incomplete':
                    dataf['flag'].append('empty')
                    if any([plot_filter=='all',plot_filter=='empty']):
                        plot_vano('{} Empty{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)

                elif md!=3:
                    dataf['flag'].append('bad_line_number')
                    if any([plot_filter=='all',plot_filter=='bad_line_number']):
                        plot_vano('{} Bad_Line_Number{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)

                else:

                    try:
                        # Matriz de rotación
                        mat, rotated_conds = rotate_points(cond_values, extremos_values)
                        extremos_values = mat.dot(extremos_values)


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

                        for j in range(len(X_cond)):
                            if Y_cond[j] > np.min(Y_extremos) and Y_cond[j] < np.max(Y_extremos):
                                x.append(X_cond[j])
                                y.append(Y_cond[j])
                                z.append(Z_cond[j])

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

                        for j in range(0, len(y_spectral)):
                            if y_spectral[j] == 0:
                                x1.append(X[j])
                                y1.append(y[j])
                                z1.append(z[j])
                            if y_spectral[j] == 1:
                                x2.append(X[j])
                                y2.append(y[j])
                                z2.append(z[j])
                            if y_spectral[j] == 2:
                                x3.append(X[j])
                                y3.append(y[j])
                                z3.append(z[j])

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
                        minimo1 = np.min(scaler_y1.inverse_transform(y_vals_scaled1.reshape(-1, 1)).flatten())
                        maximo1 = np.max(scaler_y1.inverse_transform(y_vals_scaled1.reshape(-1, 1)).flatten())
                        x_pol1 = np.linspace(minimo1, maximo1, 1000)

                        minimo2 = np.min(scaler_y2.inverse_transform(y_vals_scaled2.reshape(-1, 1)).flatten())
                        maximo2 = np.max(scaler_y2.inverse_transform(y_vals_scaled2.reshape(-1, 1)).flatten())
                        x_pol2 = np.linspace(minimo2, maximo2, 1000)

                        minimo3 = np.min(scaler_y3.inverse_transform(y_vals_scaled3.reshape(-1, 1)).flatten())
                        maximo3 = np.max(scaler_y3.inverse_transform(y_vals_scaled3.reshape(-1, 1)).flatten())
                        x_pol3 = np.linspace(minimo3, maximo3, 1000)

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

                        y_pol1 = np.interp(x_pol1, scaler_y1.inverse_transform(y_vals_scaled1.reshape(-1, 1)).flatten(), fitted_z_vals1, period=len(fitted_z_vals3))
                        y_pol2 = np.interp(x_pol2, scaler_y2.inverse_transform(y_vals_scaled2.reshape(-1, 1)).flatten(), fitted_z_vals2, period=len(fitted_z_vals3))
                        y_pol3 = np.interp(x_pol3, scaler_y3.inverse_transform(y_vals_scaled3.reshape(-1, 1)).flatten(), fitted_z_vals3, period=len(fitted_z_vals3))

                        plt.figure(figsize=(10, 6))
                        # Pintamos los puntos de cada cable
                        plt.scatter(y1, z1, color='coral', s=30)
                        plt.scatter(y2, z2, color='lightblue', s=30)
                        plt.scatter(y3, z3, color='lightgreen', s=30)

                        # Pintamos las polilíneas que hemos generado
                        plt.plot(x_pol1, y_pol1, color='red', label='P1')
                        plt.plot(x_pol2, y_pol2, color='blue', label='P2')
                        plt.plot(x_pol3, y_pol3, color='green', label='P3')

                        plt.legend()
                        plt.title(idv)
                        plt.show()
                        # X_scaled=X_scaled[:,np.logical_or.reduce(filt[1:-1])]

                        # print(len(x_pol1))
                        # print(len(y_pol1))
                        # print(x1)
                        x_fit1 = np.repeat(pd.Series(x1.flatten()).quantile(0.5),1000)
                        x_fit2 = np.repeat(pd.Series(x2.flatten()).quantile(0.5),1000)
                        x_fit3 = np.repeat(pd.Series(x3.flatten()).quantile(0.5),1000)
                        # print(x_pol1)
                        # print(y_pol1)
                        # print(x_fit1)
                        fit1=np.vstack((x_fit1, x_pol1, y_pol1))
                        fit2=np.vstack((x_fit2, x_pol2, y_pol2))
                        fit3=np.vstack((x_fit3, x_pol3, y_pol3))

                        # fit1=un_scale_conductor(fit1,scaler_x,scaler_y,scaler_z)
                        # fit2=un_scale_conductor(fit2,scaler_x,scaler_y,scaler_z)
                        # fit3=un_scale_conductor(fit3,scaler_x,scaler_y,scaler_z)
                        
                        # mat_neg,cl_pta=un_rotate_points(cl_pta,mat)
                        # mat_neg,cl_ptb=un_rotate_points(cl_ptb,mat)
                        mat_neg,fit1=un_rotate_points(fit1,mat)
                        mat_neg,fit2=un_rotate_points(fit2,mat)
                        mat_neg,fit3=un_rotate_points(fit3,mat)
                        data[i]['CONDUCTORES_CORREGIDOS'][str(0)]=fit1.T.tolist()
                        data[i]['CONDUCTORES_CORREGIDOS'][str(1)]=fit2.T.tolist()
                        data[i]['CONDUCTORES_CORREGIDOS'][str(2)]=fit3.T.tolist()
                        data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'][str(0)]=parametros1
                        data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'][str(1)]=parametros2
                        data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'][str(2)]=parametros3
                        
                        fits = np.hstack((fit1,fit2,fit3))
                        good=1
                        # print(fits)
                        # print(fits.shape)
                        # print(cond_values)
                        # crossesa=np.vstack(crossesa).T#(crossesa[0],crossesa[1],crossesa[2])
                        # print(crossesa)
                        # crossesb=np.vstack(crossesb).T
                    except:
                        bad_fit=1

                    if bad_cluster==1:
                        if all([md==3,var_z_x]):
                            dataf['flag'].append('bad_line_orientation')
                            if any([plot_filter=='all',plot_filter=='bad_line_orientation']):
                                plot_vano('{} Bad_Orientation{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        else:
                            dataf['flag'].append('bad_cluster')
                            if any([plot_filter=='bad_cluster',plot_filter=='all']):
                                plot_vano('{} Incomplete_cluster{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    elif bad_fit==1:
                        dataf['flag'].append('bad_fit')
                        if any([plot_filter=='bad_fit',plot_filter=='all']):
                            plot_vano('{} Bad_fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    elif good==1:
                        dataf['flag'].append('good_fit')
                        if any([plot_filter=='good_fit',plot_filter=='all']):
                            # plot_vano('{} Good_Fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                            # print(cond_values[0])
                            # print(cond_values[1])
                            # print(cond_values[2])
                            plot_fit_2('{} Good_Fit{}'.format(idv,' '+finc[0]),cond_values, apoyo_values, vert_values,fits)

                puntuacion=puntuación_por_vanos(data, idv).to_json()
                puntuacion_dict = json.loads(puntuacion)
                for n in puntuacion_dict:
                    puntuacion_dict[n]=puntuacion_dict[n]["0"]
                puntuacion_dict['Continuidad']=finc[0]
                puntuacion_dict['Conductores identificados']=dataf['line_number'][-1]
                puntuacion_dict['Output']=dataf['flag'][-1]
                del puntuacion_dict['Vano']
                data[i]['PUNTUACIONES']=puntuacion_dict
                print(puntuacion_dict)
    return data
                
                
def group_dbscan(k,X_scaled):

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, k-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1

    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=k, algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X_scaled)

    return labels

def data_middlepoints(data):
    x=[]
    y=[]
    ids_bad_backing = []
    ids = []
    for iel, el in enumerate(data):
        if len(data[iel]['APOYOS']) >= 2:
            ids.append(data[iel]['ID_VANO'])
            y.append((data[iel]['APOYOS'][0]['COORDEANDA_Y'] + data[iel]['APOYOS'][1]['COORDEANDA_Y']) / 2)
            x.append((data[iel]['APOYOS'][0]['COORDENADA_X'] + data[iel]['APOYOS'][1]['COORDENADA_X']) / 2)
        elif len(data[iel]['APOYOS']) == 1:
            ids.append(data[iel]['ID_VANO'])
            y.append(data[iel]['APOYOS'][0]['COORDEANDA_Y'] )
            x.append(data[iel]['APOYOS'][0]['COORDENADA_X'] )
        else:
            ids_bad_backing.append(data[iel]['ID_VANO'])
            print(f"Error: No se encontraron apoyos válidos para el elemento {iel}.")
    scaler_x=StandardScaler()
    scaler_y=StandardScaler()
    x=scaler_x.fit_transform(np.array(x).reshape(-1,1))
    y=scaler_y.fit_transform(np.array(y).reshape(-1,1))
    X=pd.DataFrame({'ids':ids,'x':x.flatten(),'y':y.flatten()})

    return ids_bad_backing,X

def pretreatment_linegroup(parameters):
    flattened_data = [flatten_sublist(sublist) for sublist in parameters]
    columns = ['ID', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    df = pd.DataFrame(flattened_data, columns=columns)
    dfd=df.dropna().copy()
    for i in  [ 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']:

        IQR=dfd[i].quantile(0.75)-dfd[i].quantile(0.25)
        dfd=dfd.loc[(dfd[i]>dfd[i].quantile(0.25)-1.5*IQR)&(dfd[i]<dfd[i].quantile(0.75)+1.5*IQR),:]
    dfd=dfd.reset_index()
    return dfd


def pretreatment_linegroup_from_json(df):

    dfd=df.dropna().copy()
    for i in  [  'a0', 'a1', 'a2']:
        IQR=dfd[i].quantile(0.75)-dfd[i].quantile(0.25)
        dfd=dfd.loc[(dfd[i]>dfd[i].quantile(0.25)-1.5*IQR)&(dfd[i]<dfd[i].quantile(0.75)+1.5*IQR),:]
    dfd=dfd.reset_index()
    return dfd

def plot_linegroup_parameters(dfd,lbl):
    total=pd.concat([dfd['A1'],dfd['B1'],dfd['C1']],axis=0)

    for ai in  ['A1','B1','C1']:
        mn=dfd[ai].mean()
        plt.hist(dfd[ai],label=ai,alpha=0.5,density=True)
        plt.axvline(mn, color='red', linestyle='--', linewidth=1)
    plt.xlim(total.min(),total.max())
    plt.legend()
    plt.title(f'3 Lines Distribution, cluster {lbl}')
    plt.show()

    mn=total.mean()
    plt.hist(total)
    plt.xlim(total.min(),total.max())
    plt.axvline(mn, color='red', linestyle='--', linewidth=1)
    plt.title(f'All lines, cluster {lbl}')
    plt.show()


def group_net(data,k=10):

    ids_single_backing,X=data_middlepoints(data)

    scaler=StandardScaler()
    X=scaler.fit_transform(X.loc[:,['x','y']])
    X=pd.DataFrame(X,columns=['x','y'])

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    distances = np.sort(distances[:, k-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1
    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=k, algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X)

    return labels


def plot_net(data,labels,k=10):

    ids_single_backing,X=data_middlepoints(data)
    plt.figure(figsize=(8, 6))

    # Plot the points and connect them with a line
    scatter =plt.scatter(X['x'], X['y'], marker='o', c=labels, cmap='viridis', label = labels)

    # for i, label in enumerate(labels):
    #     plt.annotate(i, (X.iloc[i,0], y.iloc[i,1]), textcoords="offset points", xytext=(0,10), ha='center')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sequential Points Connected by a Line')
    # Show the plot
    handles, _ = scatter.legend_elements()

    plt.legend(handles, np.unique(labels), title="Labels")
    plt.grid(True)
    plt.show()

def plot_full_net(data,labels):

    ids_single_backing,X=data_middlepoints(data)

    fulldata_plot=[]
    for lbl in np.unique(labels):

        idval_subg=X.loc[labels==lbl,'ids'].to_list()

        parameters,incomplete_vanos=fit_vano_group(data,sublist=idval_subg)

        dfd=pretreatment_linegroup(parameters)

        print(f'\nVanos con un sólo apoyo: {len(ids_single_backing)}')
        print(f'Vanos incompletos: {len(incomplete_vanos)}')
        print(f'Incompletos con apoyos: {len([el for el in incomplete_vanos if el not in ids_single_backing])}')
        print(f'Sin apoyos y completos: {len([el for el in ids_single_backing if el not in incomplete_vanos])}')
        print(f'Vanos analizados:{dfd.shape[0]}')
        print(f'Vanos perdidos:{len(parameters)-dfd.shape[0]}\n')

        plot_linegroup_parameters(dfd,str(lbl))
        total=pd.concat([dfd['A1'],dfd['B1'],dfd['C1']],axis=0)
        fulldata_plot.append(total)

    mins=[]
    maxs=[]
    for ils,lbl in enumerate(np.unique(labels)):
        plt.hist(fulldata_plot[ils],label=lbl,alpha=0.5,density=True)
        mins.append(fulldata_plot[ils].min())
        maxs.append(fulldata_plot[ils].max())

    plt.xlim(min(mins)-0.2,max(maxs)+0.2)
    plt.legend()
    plt.title('All Lines Distribution')
    plt.show()


def data_catenaryparameters_to_df(data):

    parameters={'id':[],'a0':[],'a1':[],'a2':[]}
    for i in range(len(data)):
        parameters['id'].append(data[i]['ID_VANO'])
        
        for k in range(3):
            if str(k) in data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'].keys():
                parameters['a'+str(k)].append(data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'][str(k)][0])
            else:
                parameters['a'+str(k)].append(np.nan)
    
    return pd.DataFrame(parameters)
    

# if __name__ == "__main__":
#     fit_plot_vano_group(data,sublist=[],plot_filter="all",init=0,end=20,save=False,label='')


