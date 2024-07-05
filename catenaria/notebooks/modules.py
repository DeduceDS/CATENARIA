# Obtenemos datos json con 25 elementos = vanos

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

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

from scipy.optimize import curve_fit

import math

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def print_element(element):
    """
    Print the contents of a nested dictionary in a formatted manner.

    Parameters:
    element (dict): The dictionary to be printed. It can contain nested dictionaries and lists.
    """
    
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
    """
    Extract and stack x, y, and z coordinates from a list of points.

    Parameters:
    points (list of tuples): A list where each tuple represents a point (x, y, z).

    Returns:
    tuple: Three numpy arrays containing the x, y, and z coordinates, respectively.
    """
    
    x_vals = [punto[0] for punto in points]
    y_vals = [punto[1] for punto in points]
    z_vals = [punto[2] for punto in points]
    
    return np.stack(x_vals), np.stack(y_vals), np.stack(z_vals)

def get_coord2(extremos_apoyos):
    """
    Extract and stack x, y, and z coordinates from a list of support points.

    Parameters:
    extremos_apoyos (list of dicts): A list where each dictionary contains the keys 
                    "COORDENADAS_Z", "COORDENADA_X", and "COORDEANDA_Y".
    Returns:
    tuple: Three numpy arrays containing the x, y, and z coordinates, respectively.
    """
    
    x_vals = []
    y_vals = []
    z_vals = []
    
    for i in range(len(extremos_apoyos)):        # z_vals.append(extremos_apoyos[i]["COORDENADAS_Z"])
        z_vals = z_vals + extremos_apoyos[i]["COORDENADAS_Z"]     
    
    for i in range(len(extremos_apoyos)):
    
        x_vals = x_vals + [extremos_apoyos[i]["COORDENADA_X"], extremos_apoyos[i]["COORDENADA_X"]]
        y_vals = y_vals + [extremos_apoyos[i]["COORDEANDA_Y"], extremos_apoyos[i]["COORDEANDA_Y"]]

    return np.stack(x_vals), np.stack(y_vals), np.stack(z_vals)

        
def extract_vano_values(data, vano):
    """
    Extract coordinate values for conductors, supports, and their endpoints from a data dictionary.

    Parameters:
    data (dict): The dictionary containing all the data.
    vano (str): The key corresponding to the specific span of interest in the data dictionary.

    Returns:
    tuple: Four lists containing the x, y, and z coordinate arrays for conductors, supports, vertices, and support endpoints.
    """
    
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
    """
    Add a 3D scatter plot to a given figure.

    Parameters:
    fig (plotly.graph_objs.Figure): The figure to add the plot to.
    data (list of arrays): A list containing the x, y, and z coordinates.
    color (str or array-like): The color of the markers.
    size (int): The size of the markers.
    name (str): The name of the trace.
    mode (str): The mode of the scatter plot (e.g., 'markers', 'lines', 'lines+markers').
    """
    
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


def plot_data(title,cond_values, apoyo_values, vert_values, extremos_values):
    """
    Create and display a 3D scatter plot with conductors, supports, vertices, and endpoints.

    Parameters:
    title (str): The title of the plot.
    cond_values (list of arrays): The x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The x, y, and z coordinates for vertices.
    extremos_values (list of arrays): The x, y, and z coordinates for endpoints.
    """

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
    """
    Calculate various distances between the endpoints.

    Parameters:
    extremos_values (list of arrays): The x, y, and z coordinates for endpoints.

    Returns:
    tuple: Distances between endpoints in the x, y, xz, xy, yz planes and the 3D distance.
    """
    
    x_apoyos_distance = abs(extremos_values[0][0] - extremos_values[0][2]) 
    y_apoyos_distance = abs(extremos_values[1][0] - extremos_values[1][2])
    xz_distance = np.sqrt((extremos_values[0][0] - extremos_values[0][2])**2 + (extremos_values[2][0] - extremos_values[2][2])**2)
    xy_distance = np.sqrt((extremos_values[0][0] - extremos_values[0][2])**2 + (extremos_values[1][0] - extremos_values[1][2])**2)
    yz_distance = np.sqrt((extremos_values[1][0] - extremos_values[1][2])**2 + (extremos_values[2][0] - extremos_values[2][2])**2)

    D3_apoyos_distance = np.sqrt((extremos_values[0][0] - extremos_values[0][2])**2 + (extremos_values[1][0] - extremos_values[1][2])**2 + (extremos_values[2][0] - extremos_values[2][2])**2)

    return x_apoyos_distance, y_apoyos_distance, xz_distance, xy_distance, yz_distance, D3_apoyos_distance

def rotate_points(points, extremos_values):
    """
    Rotate a set of points to align the diagonal between two endpoints with the y-axis.

    Parameters:
    points (list of arrays): The x, y, and z coordinates of the points to be rotated.
    extremos_values (list of arrays): The x, y, and z coordinates for endpoints.

    Returns:
    tuple: The rotation matrix and the rotated points.
    """
    
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
    
    return matriz_rotacion, np.array(rotated_points)

def rotate_vano(cond_values, extremos_values, apoyo_values, vert_values):
    """
    Rotate conductors, supports, vertices, and endpoints to align with the y-axis.

    Parameters:
    cond_values (list of arrays): The x, y, and z coordinates for conductors.
    extremos_values (list of arrays): The x, y, and z coordinates for endpoints.
    apoyo_values (list of arrays): The x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The x, y, and z coordinates for vertices.

    Returns:
    tuple: Rotated coordinates for conductors, supports, vertices, and endpoints.
    """

    # Rotate and compare
    mat, rotated_conds = rotate_points(cond_values, extremos_values)

    rotated_apoyos = mat.dot(apoyo_values)
    rotated_extremos = mat.dot(extremos_values)
    rotated_vertices = [mat.dot(vert) for vert in vert_values]
    
    return rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos


def clean_outliers(rotated_conds, rotated_extremos):
    """
    Clean outliers from the rotated conductor points based on endpoint boundaries and histogram analysis.

    Parameters:
    rotated_conds (numpy.ndarray): The rotated x, y, and z coordinates for conductors.
    rotated_extremos (numpy.ndarray): The rotated x, y, and z coordinates for endpoints.

    Returns:
    numpy.ndarray: The cleaned conductor points with outliers removed.
    """

    #Get left and right extreme values
    left = np.max([rotated_extremos.T[2][1],rotated_extremos.T[3][1]])
    right = np.min([rotated_extremos.T[0][1],rotated_extremos.T[1][1]])

    # Filter points within the specified boundaries
    cropped_conds = rotated_conds[:, (right > rotated_conds[1,:]) & (rotated_conds[1,:] > left)]
        
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

def scale_conductor(X):
    """
    Scale the x, y, and z coordinates of the conductor points using standard scaling.

    Parameters:
    X (numpy.ndarray): The x, y, and z coordinates of the conductor points.

    Returns:
    numpy.ndarray: The scaled x, y, and z coordinates.
    """

    # Normalizzazione dei valori di x e y
    scaler_y = StandardScaler()
    scaler_x = StandardScaler()
    scaler_z = StandardScaler()

    y_vals_scaled = scaler_y.fit_transform(X[1,:].reshape(-1, 1)).flatten()
    x_vals_scaled = scaler_x.fit_transform(X[0,:].reshape(-1, 1)).flatten()  # Flatten per curve_fit
    z_vals_scaled = scaler_z.fit_transform(X[2,:].reshape(-1, 1)).flatten()
    
    X_scaled = np.array([x_vals_scaled, y_vals_scaled, z_vals_scaled])
    
    return X_scaled


def distance(pt1, pt2):
    """
    Calculate the Euclidean distance between two 3D points.

    Parameters:
    pt1 (tuple): The (x, y, z) coordinates of the first point.
    pt2 (tuple): The (x, y, z) coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return distancia

def initialize_centroids(points, n_clusters):
    """
    Initialize centroids for clustering based on the minimum, maximum, and optionally mean of the x-coordinates.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    n_clusters (int): The number of clusters.

    Returns:
    numpy.ndarray: The initialized centroids.
    """
    
    centroids = np.zeros((n_clusters))
    
    centroids[0] = np.min(points[0,:])
    centroids[1] = np.max(points[0,:])
    
    if n_clusters > 2:
        centroids[2] = np.mean(points[0,:])
    
    return centroids

def assign_clusters(points, centroids):
    """
    Assign points to the nearest centroid based on x-coordinate distance.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    centroids (numpy.ndarray): The coordinates of the centroids.

    Returns:
    numpy.ndarray: The index of the nearest centroid for each point.
    """
    
    distances = np.abs(points[0][:, None] - centroids)
    return np.argmin(distances, axis=1)

def update_centroids(points, labels, n_clusters):
    """
    Update the centroids based on the mean of the points assigned to each cluster.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    labels (numpy.ndarray): The cluster assignment for each point.
    n_clusters (int): The number of clusters.

    Returns:
    numpy.ndarray: The updated centroids.
    """
    new_centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = points[0, labels == i]
        if cluster_points.size > 0:
            new_centroids[i] = np.mean(cluster_points)
    return new_centroids

def kmeans_clustering(points, n_clusters, max_iterations):
    """
    Perform k-means clustering on a set of points.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    n_clusters (int): The number of clusters.
    max_iterations (int): The maximum number of iterations.

    Returns:
    tuple: A tuple containing the labels for each point and the final centroids.
    """
    centroids = initialize_centroids(points, n_clusters)
    for iteration in range(max_iterations):
        labels = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, labels, n_clusters)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids

def catenaria(x, a, h, k):
    """
    Calculate the catenary curve for given parameters.

    Parameters:
    x (array-like): The x-coordinates.
    a (float): The parameter that defines the shape of the catenary.
    h (float): The horizontal offset.
    k (float): The vertical offset.

    Returns:
    numpy.ndarray: The y-coordinates of the catenary curve.
    """
    x = np.asarray(x).flatten()
    r=a * np.cosh((x - h) / a) + k
    return r

def flatten_sublist(sublist):
    """
    Flatten a list of arrays into a single list.

    Parameters:
    sublist (list of arrays): The list of arrays to be flattened.

    Returns:
    list: A single list containing all the elements of the input arrays.
    """
    flat_list = [sublist[0]] 
    for array in sublist[1:]:
        flat_list.extend(array.tolist()) 
    return flat_list

def flatten_sublist_2(sublist):
    """
    Flatten a list of arrays into a single list, ensuring the last two elements are individually added at the end.

    Parameters:
    sublist (list of arrays): The list of arrays to be flattened.

    Returns:
    list: A single list containing all the elements of the input arrays, with the last two arrays added individually.
    """
    flat_list = [sublist[0]] 
    for array in sublist[1:-2]:
        flat_list.extend(array.tolist()) 
    flat_list.extend([sublist[-2]])
    flat_list.extend([sublist[-1]])
    return flat_list

def fit_data_parameters(data,sublist=[]):
    """
    Fit catenary parameters for the given data and specified sublist of span IDs. (span = vano)

    This function processes each span in the provided data, extracts relevant values for conductors,
    supports, and endpoints, and then performs a series of transformations and clustering to fit
    catenary parameters to the data. It identifies clusters with insufficient points and skips
    them from the fitting process. Finally, it returns a DataFrame containing the fitted parameters
    for the valid clusters and a list of non-fitting span IDs.

    Parameters:
    data (list of dicts): The data containing information about different spans (vanos).
    sublist (list): The list of IDs to be processed. Only spans with these IDs will be considered for fitting.

    Returns:
    tuple: A tuple containing:
        - parameters (pd.DataFrame): A DataFrame with columns ['ID', 'a', 'h', 'k'] representing the fitted
                                    parameters for each valid span and cluster.
        - non_fitting (list): A list of span IDs (and cluster labels) that could not be fitted due to insufficient points
                            or other issues.
    """

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
    
    """
    Fit catenary parameters for a group of spans (vanos) and identify incomplete spans (vanos) and clusters (conductors).

    This function processes each span in the provided data, extracts relevant values for conductors,
    supports, and endpoints, and then performs a series of transformations and clustering to fit
    catenary parameters to the data. It identifies clusters with insufficient points and spans with
    incomplete data, and skips them from the fitting process. Finally, it returns a list of fitted
    parameters for the valid spans, and lists of incomplete spans and clusters.

    Parameters:
    data (list of dicts): The data containing information about different spans (vanos).
    sublist (list): The list of IDs to be processed. Only spans with these IDs will be considered for fitting.

    Returns:
    tuple: A tuple containing:
        - parameters (list): A list of fitted parameters for each valid span, including the span ID and coordinates.
        - incomplete_vanos (list): A list of span IDs that could not be fitted due to incomplete data or other issues.
        - incomplete_lines (list): A list of span IDs with cluster labels that could not be fitted due to insufficient points
                                or other issues.

    """

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

def plot_vano(title,X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values):
    
    """
    This function generates scatter plots to visualize the clustering results on the x-y and y-z planes.
    It also generates a 3D plot of the conductor values, support values, vertices, and endpoints.

    Parameters:
    title (str): The title of the plots.
    X_scaled (numpy.ndarray): The scaled x, y, and z coordinates of the conductor points.
    labels (numpy.ndarray): The cluster assignment for each point.
    cond_values (list of arrays): The original x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The original x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The original x, y, and z coordinates for vertices.
    extremos_values (list of arrays): The original x, y, and z coordinates for endpoints.

    """

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
    

def plot_vano_group(data,sublist=[],filter="all"):
    """
    Plot and analyze a group of spans based on their fitting quality.

    This function processes each span (vano) in the provided data, extracts relevant values for conductors,
    supports, and endpoints, and then performs a series of transformations and clustering to analyze
    the data. It generates plots based on the specified filter criteria, which can be "all", "length",
    "good_fit", "bad_fit", or "incomplete". The plots help visualize the clustering and fitting results
    for each span.

    Parameters:
    data (list of dicts): The data containing information about different spans (vanos).
    sublist (list): The list of IDs to be processed. Only spans with these IDs will be considered for analysis.
    filter (str): The filter criteria for plotting. Can be one of "all", "length", "good_fit", "bad_fit", or "incomplete".

    Returns:
    None
    """
    
    #filter= all, length,good_fit,bad_fit,incomplete
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    for i in range(len(data)):
        

        idv=data[i]['ID_VANO']
        
        if idv in sublist:
                
            print(f"\nProcessing Vano {i}")
            print(f"\nReference {idv}")
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
                count_good=0
                for lab in np.unique(labels):
                    
                    idl=idv+'_'+str(lab)
                    clust = X_scaled[:,labels == lab]
                    proportion = clust.shape[1]/total_points

                    if proportion< 0.15:
                        if idv not in incomplete_vanos:
                            incomplete_vanos.append(idv)
                        incomplete_lines.append(idl)
                        if any([filter=='incomplete',filter=='all']):
                            plot_vano('Incomplete',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                
                        print(f"Error en el cluster {idv}")
                        print(f"This cluster represents: {round(100*proportion,2)}%")
                    else:
                        y_vals = clust[1].reshape(-1, 1)
                        z_vals = clust[2].reshape(-1, 1)
                        initial_params = [1, 0, 0]  # a, h, k
                        try:
                            optim_params, _ = curve_fit(catenaria, y_vals.flatten(), z_vals.flatten(), p0=initial_params, method = 'lm')
                            fitted_z = catenaria(y_vals.flatten(), *optim_params)

                            count_good=count_good+1
                            if all([any([filter=='good_fit',filter=='all']),count_good==2]):
                                plot_vano('Good_Fit',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                
                        except:
                            if any([filter=='bad_fit',filter=='all']):
                                plot_vano('Bad_Fit',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                
            else:
                if any([filter=='length',filter=='all']):
                    plot_vano('Bad_Length',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)


def group_dbscan(k,X_scaled):
    """
    Perform DBSCAN clustering on scaled data, determining the optimal epsilon using the k-nearest neighbors method.

    This function uses the DBSCAN algorithm to cluster the given scaled data. It first fits the k-nearest neighbors 
    to determine the distances and uses the second derivative of the sorted distances to find the inflection point, 
    which is used as the epsilon value for DBSCAN. The function then performs DBSCAN clustering and returns the cluster labels.

    Parameters:
    k (int): The number of neighbors to use for determining the optimal epsilon.
    X_scaled (numpy.ndarray): The scaled x, y, and z coordinates of the data points.

    Returns:
    numpy.ndarray: The cluster labels assigned to each data point.
    """

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, k-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1 

    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=k, algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X_scaled) 

    return labels

def plot_vano_group_2(data,sublist=[],filter="all"):
    """
    Plot and analyze a group of spans (vanos) based on their fitting quality using DBSCAN clustering.

    This function processes each span (vano) in the provided data, extracts relevant values for conductors,
    supports, and endpoints, and then performs a series of transformations and clustering using the
    DBSCAN algorithm to analyze the data. It generates plots based on the specified filter criteria,
    which can be "all", "length", "good_fit", "bad_fit", or "incomplete". The plots help visualize the 
    clustering and fitting results for each span (vano).

    Parameters:
    data (list of dicts): The data containing information about different spans (vanos).
    sublist (list): The list of IDs to be processed. Only spans (vanos) with these IDs will be considered for analysis.
    filter (str): The filter criteria for plotting. Can be one of "all", "length", "good_fit", "bad_fit", or "incomplete".
    """
    #filter= all, length,good_fit,bad_fit,incomplete
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    for i in range(len(data)):
        

        idv=data[i]['ID_VANO']
        
        if idv in sublist:
                
            print(f"\nProcessing Vano {i}")
            print(f"\nReference {idv}")
            cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
            
            cond_values=np.vstack(cond_values)
            apoyo_values=np.vstack(apoyo_values)
            
            if np.array(extremos_values).shape[1]==4:
                y=((data[i]['APOYOS'][0]['COORDEANDA_Y'] + data[i]['APOYOS'][1]['COORDEANDA_Y']) / 2)
                x=((data[i]['APOYOS'][0]['COORDENADA_X'] + data[i]['APOYOS'][1]['COORDENADA_X']) / 2)
                rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
                
                cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
                
                X_scaled = scale_conductor(cropped_conds)
                
                ar=[5,10,20]
                for k in ar:

                    labels=group_dbscan(k,X_scaled.T)
                    score=metrics.silhouette_score(X_scaled.T,labels)

                    if k==np.array(ar).min():
                        best_score=score
                        best_k=k
                    elif score>best_score:
                        best_score=score
                        best_k=k

                labels=group_dbscan(best_k,X_scaled.T)

                # labels, centroids = kmeans_clustering(X_scaled, n_clusters=3, max_iterations=1000)

                total_points = X_scaled.shape[1]

                parameters_vano=[]
                count_good=0
                for lab in np.unique(labels):
                    
                    idl=idv+'_'+str(lab)
                    clust = X_scaled[:,labels == lab]
                    proportion = clust.shape[1]/total_points

                    if proportion< 0.15:
                        if idv not in incomplete_vanos:
                            incomplete_vanos.append(idv)
                        incomplete_lines.append(idl)
                        if any([filter=='incomplete',filter=='all']):
                            plot_vano('Incomplete',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                
                        print(f"Error en el cluster {idv}")
                        print(f"This cluster represents: {round(100*proportion,2)}%")
                    else:
                        y_vals = clust[1].reshape(-1, 1)
                        z_vals = clust[2].reshape(-1, 1)
                        initial_params = [1, 0, 0]  # a, h, k
                        try:
                            optim_params, _ = curve_fit(catenaria, y_vals.flatten(), z_vals.flatten(), p0=initial_params, method = 'lm')
                            fitted_z = catenaria(y_vals.flatten(), *optim_params)

                            count_good=count_good+1
                            if all([any([filter=='good_fit',filter=='all']),count_good==2]):
                                plot_vano('Good_Fit',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                
                        except:
                            if any([filter=='bad_fit',filter=='all']):
                                plot_vano('Bad_Fit',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                
            else:
                if any([filter=='length',filter=='all']):
                    plot_vano('Bad_Length',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)

#         if idv in ['G_13578475_13578503',
#  'G_13578475_13578516',
#  'G_13578376_13578412',
#  'G_13578457_13578461',
#  'G_13578336_13578432',
#  'G_13576786_13576844',
#  'G_13578376_13578516',
#  'G_13578452_13578480',
#  'G_13578337_13578377',
#  'G_13578393_13578412',
#  'G_13578283_13578388',
#  'G_13576829_13578393',
#  'G_13576775_13576844',
#  'G_13578344_13578397',
#  'G_13528004_13528124',
#  'G_13578283_13578344',
#  'G_13578461_13578480',
#  'G_13578320_13578492',
#  'G_13578409_13578432']:
#             print(x)
#             print(y)
#             print(parameters_vano)
#         print(parameters_vano)

# def fill_vano_group(data,sublist=[]):

#     parameters=[]
#     incomplete_vanos = []
#     for i in range(len(data)):
        
#         print(f"\nProcessing Vano {i}")

#         idv=data[i]['ID_VANO']
#         if idv in sublist:
                
#             cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
            
#             cond_values=np.vstack(cond_values)
#             apoyo_values=np.vstack(apoyo_values)

#             if np.array(extremos_values).shape[1]==4:
#                 rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
                
#                 cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
                
#                 X_scaled = scale_conductor(cropped_conds)

#                 labels, centroids = kmeans_clustering(X_scaled, n_clusters=3, max_iterations=1000)
                
#                 total_points = X_scaled.shape[1]

#                 parameters_vano=[]
#                 for lab in np.unique(labels):
                    
#                     clust = X_scaled[:,labels == lab]
#                     proportion = clust.shape[1]/total_points

#                     if proportion< 0.15:
#                         if idv not in incomplete_vanos:
#                             incomplete_vanos.append(idv)
#                         print(f"Error en el cluster {idv}")
#                         print(f"This cluster represents: {round(100*proportion,2)}%")
#                     else:
#                         y_vals = clust[1].reshape(-1, 1)
#                         z_vals = clust[2].reshape(-1, 1)
#                         initial_params = [1, 0, 0]  # a, h, k
#                         try:
#                             optim_params, _ = curve_fit(catenaria, y_vals.flatten(), z_vals.flatten(), p0=initial_params, method = 'lm')
#                             fitted_z = catenaria(y_vals.flatten(), *optim_params)
#                             if idv not in incomplete_vanos:
#                                 parameters_vano.append(optim_params)
#                         except:
#                             if idv not in incomplete_vanos:
#                                 incomplete_vanos.append(idv)
#             else:
#                 incomplete_vanos.append(idv)
            
#             if idv not in incomplete_vanos:
#                 parameters.append([idv]+parameters_vano)

#     return data

def data_middlepoints(data):
    """
    Calculate and scale the middle points of supports for each span in the provided data.

    This function processes the data to extract the middle points of the supports (apoyos) for each span (vano).
    If a span has two supports, the middle point is calculated as the average of the coordinates of the two supports.
    If a span has only one support, its coordinates are used directly. Spans with no valid supports are reported as errors.
    The coordinates are then scaled using standard scaling.

    Parameters:
    data (list of dicts): The data containing information about different spans.

    Returns:
    tuple:
        - ids_bad_backing (list): A list of span IDs with no valid supports.
        - X (pd.DataFrame): A DataFrame containing the span IDs and the scaled x and y coordinates of the middle points.
    """
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
    """
    Preprocess and clean the parameters data for line groups.

    This function flattens the list of parameters, converts it into a DataFrame, and performs
    outlier removal using the interquartile range (IQR) method. The cleaned DataFrame is then
    returned for further analysis.

    Parameters:
    parameters (list): A list of parameters for different line groups, where each sublist contains
                    the parameters for a single line group.

    Returns:
    pd.DataFrame: A cleaned DataFrame containing the parameters for different line groups, with
                outliers removed and indices reset.
    """
    flattened_data = [flatten_sublist(sublist) for sublist in parameters]
    columns = ['ID', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    df = pd.DataFrame(flattened_data, columns=columns)
    dfd=df.dropna().copy()
    for i in  [ 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']:
        
        IQR=dfd[i].quantile(0.75)-dfd[i].quantile(0.25)
        dfd=dfd.loc[(dfd[i]>dfd[i].quantile(0.25)-1.5*IQR)&(dfd[i]<dfd[i].quantile(0.75)+1.5*IQR),:]
    dfd=dfd.reset_index()
    return dfd

def plot_linegroup_parameters(dfd,lbl):
    """
    Plot the distribution of parameters for different line groups within a cluster.

    This function takes a DataFrame containing the parameters for different line groups (A1, B1, C1)
    within a specific cluster, and generates histograms to visualize their distributions. It also plots
    the overall distribution of all parameters combined. Mean values are highlighted with a red dashed line.

    Parameters:
    dfd (pd.DataFrame): A DataFrame containing the parameters for different line groups.
    lbl (str): The label of the cluster being plotted.
    """
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
    """
    Perform DBSCAN clustering on the network of spans based on their middle points.

    This function extracts the middle points of the spans from the provided data, scales the coordinates,
    and performs DBSCAN clustering to group the spans. The optimal epsilon value for DBSCAN is determined
    using the k-nearest neighbors method by finding the inflection point in the sorted distances.

    Parameters:
    data (list of dicts): The data containing information about different spans.
    k (int): The number of neighbors to use for determining the optimal epsilon.

    Returns:
    numpy.ndarray: The cluster labels assigned to each span.
    """
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
    """
    Plot the network of spans based on clustering labels.

    This function extracts the middle points of the spans from the provided data, and then generates 
    a scatter plot of these points colored by their cluster labels. The plot includes labels and a legend 
    to indicate the different clusters.

    Parameters:
    data (list of dicts): The data containing information about different spans.
    labels (numpy.ndarray): The cluster labels assigned to each span.
    k (int): An optional parameter for any future enhancements or clustering requirements (default is 10).
    """
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
    """
    Plot the full network of spans based on clustering labels.

    This function processes each cluster in the provided data, extracts relevant values for each span,
    fits catenary parameters, and generates plots for each cluster. It also provides summary statistics
    about the spans, including counts of spans with single supports, incomplete spans, and spans analyzed.
    Finally, it plots the distribution of parameters for all lines in the network.

    Parameters:
    data (list of dicts): The data containing information about different spans.
    labels (numpy.ndarray): The cluster labels assigned to each span.
    """

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

# if __name__ == "__main__":
#     main()


