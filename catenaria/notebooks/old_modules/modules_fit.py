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

#### FUNCTIONS TO PROCESS JSON DATA ####

from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import SpectralClustering
from puntuacion import *


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

    for i in range(len(extremos_apoyos)):
        # z_vals.append(extremos_apoyos[i]["COORDENADAS_Z"])
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

#### FUNCTIONS TO PLOT DATA AND FITS ####

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

def plot_fit(title,cond_values, apoyo_values, vert_values,fit,crossesa,crossesb):
    """
    Create and display a 3D scatter plot with conductors, supports, vertices, fit and endpoints.

    Parameters:
    title (str): The title of the plot.
    cond_values (list of arrays): The x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The x, y, and z coordinates for vertices.
    fit (list of arrays): The x, y, and z coordinates for the selected fit.
    crossesa (list of arrays): The x, y, and z coordinates for endpoints in apoyo 1.
    crossesb (list of arrays): The x, y, and z coordinates for endpoints in apoyo 2.
    """
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
    """
    Create and display a 3D scatter plot with conductors, supports, vertices, and fit.

    Parameters:
    title (str): The title of the plot.
    cond_values (list of arrays): The x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The x, y, and z coordinates for vertices.
    fit (list of arrays): The x, y, and z coordinates for the selected fit.
    """

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

#### FUNCTIONS TO COMPUTE DISTANCES ####

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

#### FUNCTIONS TO TRANSFORM/PREPROCESS 3D POINTS ####

#### ROTATION FUNCTIONS ####

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
    # print(rotated.shape)

    return matriz_rotacion, np.array(rotated_points)

def un_rotate_points(points, matriz_rotacion):
    """
    Unrotate a set of points to dis-align with the diagonal between two endpoints with the y-axis.

    Parameters:
    points (list of arrays): The x, y, and z coordinates of the points to be rotated.
    matriz_rotacion (list of arrays): 3D inverted rotation matrix.

    Returns:
    tuple: The rotation matrix and the rotated points.
    """

    points = np.array(points).T
    # Crear la matriz de rotación para alinear la diagonal con el eje Y
    matriz_rotacion = matriz_rotacion.T

    rotated_points = matriz_rotacion.dot(points.T)
    # print(rotated.shape)

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

    return mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos

#### OUTLIER FUNCTIONS ####

def clean_outliers(rotated_conds, rotated_extremos):
    """
    Cleans outliers from the rotated conductor points based on endpoint boundaries and histogram analysis.

    Parameters:
    rotated_conds (numpy.ndarray): The rotated x, y, and z coordinates for conductors.
    rotated_extremos (numpy.ndarray): The rotated x, y, and z coordinates for endpoints.

    Returns:
    numpy.ndarray: The cleaned conductor points with outliers removed.

    The function performs the following steps:
    1. Defines the left and right boundaries based on the rotated endpoints.
    2. Filters the conductor points within the specified y-coordinate boundaries.
    3. Computes a histogram of the y-coordinates of the cropped conductor points.
    4. Identifies significant peaks in the histogram to determine upper and lower thresholds for the y-coordinates.
    5. Filters the conductor points using the detected y-coordinate thresholds.
    6. Further refines the conductor points by removing the extreme 2% outliers in both y and x coordinates.

    Note:
    - The function assumes that the input arrays are properly rotated and aligned.
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

def clean_outliers_2(rotated_conds):
    """
    Cleans outliers from the rotated conductor points based on interquartile range (IQR).

    Parameters:
    rotated_conds (numpy.ndarray): The rotated x, y, and z coordinates for conductors.

    Returns:
    numpy.ndarray: The cleaned conductor points with outliers removed based on IQR.
    """

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
    """
    Cleans outliers from the cropped conductor points using statistical outlier removal.

    Parameters:
    cropped_conds (numpy.ndarray): The cropped x, y, and z coordinates for conductors.

    Returns:
    numpy.ndarray: The cleaned conductor points with outliers removed based on statistical outlier removal.

    The function performs the following steps:
    1. Converts the cropped conductor points into an Open3D PointCloud object.
    2. Applies statistical outlier removal with a specified number of neighbors and standard deviation multiplier.
    3. Selects the inlier points and returns them.
    """
    
    nn = 10 # Local search
    std_multip = 1 # Not very sensitive

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.array(cropped_conds).T)
    cl, ind = pcd_o3d.remove_statistical_outlier(nb_neighbors=nn, std_ratio=std_multip)
    inlier_cloud = pcd_o3d.select_by_index(ind)

    cropped_conds = np.asarray(inlier_cloud.points).T
    return cropped_conds

def clean_outliers_4(cropped_conds):
    """
    Cleans outliers from the cropped conductor points using radius outlier removal.

    Parameters:
    cropped_conds (numpy.ndarray): The cropped x, y, and z coordinates for conductors.

    Returns:
    numpy.ndarray: The cleaned conductor points with outliers removed based on radius outlier removal.
    """
    
    nn = 10 # Local search
    radius = 1 # Not very sensitive

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.array(cropped_conds).T)
    cl, ind = pcd_o3d.remove_radius_outlier(nb_points=nn, radius=radius)
    inlier_cloud = pcd_o3d.select_by_index(ind)

    cropped_conds = np.asarray(inlier_cloud.points).T
    return cropped_conds

#### SCALE FUNCTIONS ####

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

    return X_scaled,scaler_x,scaler_y,scaler_z


def un_scale_conductor(X,scaler_x,scaler_y,scaler_z):
    """
    Unscale the x, y, and z scaked coordinates of the conductor points using standard scaling.

    Parameters:
    X (numpy.ndarray): The x, y, and z coordinates of the conductor points.
    scaler_x/y/z (Scaler Objects): The scaler used to scale each coordinate.

    Returns:
    numpy.ndarray: The unscaled x, y, and z coordinates.
    """
    y_vals_unscaled = scaler_y.inverse_transform(X[1,:].reshape(-1, 1)).flatten()
    x_vals_unscaled = scaler_x.inverse_transform(X[0,:].reshape(-1, 1)).flatten()  # Flatten per curve_fit
    z_vals_unscaled = scaler_z.inverse_transform(X[2,:].reshape(-1, 1)).flatten()

    X_unscaled = np.array([x_vals_unscaled, y_vals_unscaled, z_vals_unscaled])

    return X_unscaled

#### FUNCTIONS TO MODIFY/CORRECT ORIGINAL DATA####

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

def define_backings(vano_length,apoyo_values):
    """
    Define the backings (extremos) based on the length of the span and the coordinates of the supports.

    This function clusters the support coordinates into two groups using k-means clustering, calculates
    the center of mass for each group, and determines the coordinates of the backings. If the distance
    between the centroids of the clusters significantly deviates from the provided span length, the
    function returns -1 indicating an error.

    Parameters:
    vano_length (float): The length of the span (vano).
    apoyo_values (list of lists or numpy.ndarray): The x, y, and z coordinates of the supports.

    Returns:
    list: A list containing three numpy arrays representing the x, y, and z coordinates of the backings
        for each support. If the distance between centroids deviates significantly from the span length,
        returns -1.
    """
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

#### FUNCTIONS FOR CLUSTERING ####

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
    centroids[1] = np.mean(points[0,:])
    centroids[2] = np.max(points[0,:])

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
            # print(f"Convergence reached at iteration {iteration}")
            break
        centroids = new_centroids

    print(labels.shape)
    print(np.unique(labels))
    print(centroids.shape)
    return labels, centroids


def spectral_clustering(points, n_clusters, n_init):
    """
    Performs spectral clustering on the given data points with scikit-learn. 
    The affinity matrix is constructed using the k-nearest neighbors graph.

    Parameters:
    points (numpy.ndarray): A 2D array where each column represents a data point.
    n_clusters (int): The number of clusters to form.
    n_init (int): The number of times the algorithm will run with different initializations.

    Returns:
    tuple: A tuple containing:
        - labels (numpy.ndarray): An array of cluster labels assigned to each data point.
        - centroids (list): A list of centroids for each cluster, where each centroid is a list of coordinates.

    The function performs the following steps:
    1. Determines the number of neighbors for the k-nearest neighbors graph.
    2. Constructs the affinity matrix using the k-nearest neighbors graph.
    3. Applies the Spectral Clustering algorithm with the specified number of clusters and initializations.
    4. Computes the centroids for each cluster by averaging the coordinates of points within the cluster.
    5. Returns the cluster labels and centroids.
    """

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
    """
    Inverts a linear model to compute the x-values from given y-values.

    Parameters:
    y_val (float or numpy.ndarray), slope (float), intercept (float)
    
    Returns:
    float or numpy.ndarray: The computed x-values corresponding to the given y-values.
    """
    
    return (y_val - intercept) / slope

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
        print(array)
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

def clpt_to_array(cl_pt):
    """
    This function processes a list of coordinate points, extracts the x, y, and z coordinates,
    and returns them as separate numpy arrays.

    Parameters:
    cl_pt (list of lists or tuples): A list where each element is a list or tuple containing
                                    the x, y, and z coordinates of a point.
    Returns:
    numpy.ndarray: A 2D numpy array where the first row contains the x coordinates, the second row
                contains the y coordinates, and the third row contains the z coordinates.
    """
    rfx=[]
    rfy=[]
    rfz=[]
    print(cl_pt)
    for el in cl_pt:
        rfx.append(el[0])
        rfy.append(el[1])
        rfz.append(el[2])
    return np.array([rfx,rfy,rfz])

#### FUNCTIONS FOR DATA FITS ####

def catenaria(x, a, h, k):
    x = np.asarray(x).flatten()
    r=a * np.cosh((x - h) / a) + k
    return r

def fit_data_parameters(data,sublist=[]):
    """
    Fits curves to the vano data entries, calculates parameters for the fitted curves, and identifies non-fitting vanos and lines.

    Parameters:
    data (list): List of dictionaries containing vano data. Each dictionary should have keys like 'ID_VANO', 'APOYOS', and 'LONGITUD_2D'.
    sublist (list, optional): Sublist of 'ID_VANO' to process. If empty, all vanos in data are processed. Defaults to an empty list.

    Returns:
    tuple: A tuple containing:
        - parameters (pandas.DataFrame): A DataFrame with the fitted parameters for each vano, including columns 'ID', 'a', 'h', and 'k'.
        - non_fitting (list): A list of IDs of vanos or individual lines within vanos that could not be fully processed.

    The function performs the following steps:
    1. Iterates over the data and processes each vano entry.
    2. Extracts coordinate values for the conductors, supports, vertices, and extremos.
    3. Rotates the vano data points and removes outliers.
    4. Scales the data points and applies k-means clustering to identify clusters.
    5. Fits a catenary curve to each cluster and calculates the parameters.
    6. Identifies non-fitting vanos and lines based on the clustering results and curve fitting.
    7. Returns the calculated parameters in a DataFrame, and a list of non-fitting vanos and lines.
    
    Note:
        Functions called:  `extract_vano_values`, `rotate_vano`, `clean_outliers`, `scale_conductor`, `kmeans_clustering`, and `catenaria`.
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
    Fits curves to the vano data entries, calculates parameters for the fitted curves, and identifies incomplete vanos and lines.

    Parameters:
    data (list): List of dictionaries containing vano data. Each dictionary should have keys like 'ID_VANO', 'APOYOS', and 'LONGITUD_2D'.
    sublist (list, optional): Sublist of 'ID_VANO' to process. If empty, all vanos in data are processed. Defaults to an empty list.

    Returns:
    tuple: A tuple containing:
        - parameters (list): A list of parameters for each fitted vano, including the vano ID and the optimized parameters for the curve.
        - incomplete_vanos (list): A list of IDs of vanos that could not be fully processed.
        - incomplete_lines (list): A list of IDs of individual lines within vanos that could not be fully processed.

    The function performs the following steps:
    1. Iterates over the data and processes each vano entry.
    2. Extracts coordinate values for the conductors, supports, vertices, and extremos.
    3. Rotates the vano data points and removes outliers.
    4. Scales the data points and applies k-means clustering to identify clusters.
    5. Fits a catenary curve to each cluster and calculates the parameters.
    6. Identifies incomplete vanos and lines based on the clustering results and curve fitting.
    7. Returns the calculated parameters, and lists of incomplete vanos and lines.

    Note:
    - Functions called: `extract_vano_values`, `rotate_vano`, `clean_outliers`, `scale_conductor`, `kmeans_clustering`, and `catenaria`.
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


def group_dbscan_4(k,X_scaled):
    """
    Groups data points into clusters using the DBSCAN algorithm, with the optimal epsilon value determined from k-nearest neighbors distances.

    Parameters:
    k (int): The number of nearest neighbors to consider for determining the epsilon value.
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (list): A list of centroids for each cluster, where each centroid is a list of coordinates.
        - labels (numpy.ndarray): An array of labels assigned to each data point, indicating the cluster it belongs to. 
        Noise points are labeled as -1.

    The function performs the following steps:
    1. Fits a k-nearest neighbors model to the data to determine distances to the k-th nearest neighbors.
    2. Sorts these distances and computes the second derivative to find the inflection point, which indicates the optimal epsilon value.
    3. Applies the DBSCAN algorithm with the determined epsilon value and the specified min_samples.
    4. Computes the centroids for each cluster by averaging the coordinates of points within the cluster.

    Note:
    - The function determines the optimal epsilon value dynamically based on the distances to the k-th nearest neighbors.
    - The DBSCAN algorithm groups data points into clusters based on density, identifying regions of high density and labeling points in low-density regions as noise.
    """

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
    """
    Finds clusters in the scaled data using the DBSCAN algorithm with different epsilon values,
    and selects the best clustering based on the silhouette score.

    Parameters:
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (numpy.ndarray): The centroids of the clusters found by DBSCAN.
        - labels (numpy.ndarray): The labels assigned to each data point indicating the cluster it belongs to.

    The function performs the following steps:
    1. Iterates over a predefined list of epsilon values.
    2. Applies the `group_dbscan_3` function to cluster the data with each epsilon value.
    3. Computes the silhouette score for each clustering result.
    4. Selects the clustering result with the highest silhouette score.
    5. Returns the centroids and labels of the best clustering result.

    The silhouette score measures how similar an object is to its own cluster compared to other clusters, providing an indication of the quality of the clustering.
    """

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
    """
    Groups data points into clusters using the DBSCAN algorithm, with the optimal epsilon value determined from k-nearest neighbors distances.

    Parameters:
    k (int): The number of nearest neighbors to consider for determining the epsilon value.
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (list): A list of centroids for each cluster, where each centroid is a list of coordinates.
        - labels (numpy.ndarray): An array of labels assigned to each data point, indicating the cluster it belongs to. 
        Noise points are labeled as -1.

    The function performs the following steps:
    1. Fits a k-nearest neighbors model to the data to determine distances to the k-th nearest neighbors.
    2. Sorts these distances and computes the second derivative to find the inflection point, which indicates the optimal epsilon value.
    3. Applies the DBSCAN algorithm with the determined epsilon value and the specified min_samples.
    4. Computes the centroids for each cluster by averaging the coordinates of points within the cluster.

    Note:
    - The function determines the optimal epsilon value dynamically based on the distances to the k-th nearest neighbors.
    - The DBSCAN algorithm groups data points into clusters based on density, identifying regions of high density and labeling points in low-density regions as noise.
    """

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
    """
    Finds clusters in the scaled data using the DBSCAN algorithm with different epsilon values,
    and selects the best clustering based on the silhouette score.

    Parameters:
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (numpy.ndarray): The centroids of the clusters found by DBSCAN.
        - labels (numpy.ndarray): The labels assigned to each data point indicating the cluster it belongs to.

    The function performs the following steps:
    1. Iterates over a predefined list of epsilon values.
    2. Applies the `group_dbscan_3` function to cluster the data with each epsilon value.
    3. Computes the silhouette score for each clustering result.
    4. Selects the clustering result with the highest silhouette score.
    5. Returns the centroids and labels of the best clustering result.

    The silhouette score measures how similar an object is to its own cluster compared to other clusters, providing an indication of the quality of the clustering.
    """

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
    """
    Defines the backing points for a vano based on its length and supporting points.

    Parameters:
    vano_length (float): The length of the vano.
    apoyo_values (list or numpy.ndarray): A list or array of supporting points, where each point is represented by its (x, y, z) coordinates.

    Returns:
    list or int: A list of coordinate arrays representing the backing points, with the format:
                [array([x1, x1, x2, x2]), array([y1, y1, y2, y2]), array([z1, z1, z2, z2])].
                If the distance between the calculated centroids deviates by more than 10% from the vano length, returns -1.

    The function performs the following steps:
    1. Converts the supporting points into a NumPy array.
    2. Uses KMeans clustering to classify the points into two clusters.
    3. Calculates the centroid of each cluster.
    4. Checks the distance between the two centroids.
    5. If the distance is within 10% of the vano length, formats the centroid coordinates into the required structure and returns them.
    Otherwise, returns -1.

    The function ensures that the calculated backings closely match the expected vano length, enhancing the accuracy of the backing points used for further processing.
    """
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
    """
    Converts a list of 3D points (tuples or lists) into a NumPy array with separate arrays for each coordinate.

    Parameters:
    cl_pt (list): List of 3D points, where each point is a tuple or list of three coordinates (x, y, z).

    Returns:
    numpy.ndarray: A 2D NumPy array with shape (3, n) where n is the number of points. The first row contains the x-coordinates, 
                the second row contains the y-coordinates, and the third row contains the z-coordinates.

    This function processes a list of 3D points and separates their coordinates into individual arrays for x, y, and z. 
    It then combines these arrays into a single NumPy array for easy manipulation and analysis.
    """
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
    """
    Processes a group of "vano" data entries, fits curves, and plots results based on various conditions and filters.

    Parameters:
    data (list): List of dictionaries containing vano data. Each dictionary should have keys like 'ID_VANO', 'APOYOS', and 'LONGITUD_2D'.
    sublist (list, optional): Sublist of 'ID_VANO' to process. If empty, all vanos in data are processed. Defaults to an empty list.
    plot_filter (str, optional): Filter for plotting. Options are "bad_backing", "bad_cluster", "bad_line_number", "bad_line_orientation", "bad_fit", "good_fit", "empty", or "all". Defaults to None.
    init (int, optional): Starting index for processing. Defaults to 0.
    end (int, optional): Ending index for processing. If None, processes up to the length of the sublist. Defaults to None.
    save (bool, optional): Flag to save the resulting data to a JSON file. Defaults to False.
    label (str, optional): Label to add to the saved file name. Defaults to an empty string.

    Returns:
    pandas.DataFrame: DataFrame containing results with columns 'id', 'flag', and 'line_number'.

    The function performs the following steps:
    1. Initializes necessary variables and structures.
    2. Iterates over the data, processing each vano entry.
    3. Extracts and scales coordinate data.
    4. Filters and clusters data points.
    5. Fits a curve to the data points and evaluates the fit.
    6. Plots the results based on the specified plot_filter.
    7. Saves the results to a JSON file if save is True.
    8. Returns a DataFrame with the processing results.

    The function handles different conditions such as incomplete data, bad cluster formation, and bad fit, and tags each vano entry accordingly. It also rotates and scales the data points for fitting and plots the results based on the specified filters.
    """
    
    #filter= "bad_backing", bad_cluster, bad_line_number, bad_line_orientation, bad_fit, good_fit, empty

    if len(sublist)==0:
        sublist=[data[i]['ID_VANO'] for i in range(len(data))]
    end=int(len(data)) if end==None else end
    print(end)
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
                                
                                # rmse, corr = evaluate_fit(fit, X_scaled)

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
    """
    Processes a group of "vano" data entries, fits curves, and plots results based on various conditions and filters.

    Parameters:
    data (list): List of dictionaries containing vano data. Each dictionary should have keys like 'ID_VANO', 'APOYOS', and 'LONGITUD_2D'.
    sublist (list, optional): Sublist of 'ID_VANO' to process. If empty, all vanos in data are processed. Defaults to an empty list.
    plot_filter (str, optional): Filter for plotting. Options are "bad_backing", "bad_cluster", "bad_line_number", "bad_line_orientation", "bad_fit", "good_fit", "empty", or "all". Defaults to None.
    init (int, optional): Starting index for processing. Defaults to 0.
    end (int, optional): Ending index for processing. If None, processes up to the length of the sublist. Defaults to None.
    save (bool, optional): Flag to save the resulting data to a JSON file. Defaults to False.
    label (str, optional): Label to add to the saved file name. Defaults to an empty string.

    Returns:
    pandas.DataFrame: DataFrame containing results with columns 'id', 'flag', and 'line_number'.

    The function performs the following steps:
    1. Initializes necessary variables and structures.
    2. Iterates over the data, processing each vano entry.
    3. Extracts and scales coordinate data.
    4. Filters and clusters data points.
    5. Fits a curve to the data points and evaluates the fit.
    6. Plots the results based on the specified plot_filter.
    7. Saves the results to a JSON file if save is True.
    8. Returns a DataFrame with the processing results.

    The function handles different conditions such as incomplete data, bad cluster formation, and bad fit, and tags each vano entry accordingly. It also rotates and scales the data points for fitting and plots the results based on the specified filters.

    Example Usage:
    data = [...]  # List of dictionaries with vano data
    df = fit_plot_vano_group_2(data, sublist=[1, 2, 3], plot_filter='good_fit', save=True, label='test')
    """
    #filter= "bad_backing", bad_cluster, bad_line_number, bad_line_orientation, bad_fit, good_fit, empty

    if len(sublist)==0:
        sublist=[data[i]['ID_VANO'] for i in range(len(data))]
        
    end=int(len(data)) if end==None else end
    
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    dataf = {'id': [], 'flag': [],'line_number':[]}
    
    rmses = []
    maxes = []
    correlations = []
    
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
                print(extremos_values)
                if np.array(extremos_values).shape[1]!=4:

                    dataf['flag'].append('bad_backing')
                    dataf['line_number'].append(0)
                    if any([plot_filter=='all',plot_filter=='bad_backing']):
                        continue
                        # plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        
                    extremos_values = define_backings(vano_length,apoyo_values)
                    
                    if extremos_values == -1:
                        continue
                
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
                        # plot_vano('{} Empty{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        pass

                elif md!=3:
                    dataf['flag'].append('bad_line_number')
                    if any([plot_filter=='all',plot_filter=='bad_line_number']):
                        # plot_vano('{} Bad_Line_Number{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        pass

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
                        
                        # def catenaria(x, a, h, k):
                        #     return a*np.cosh((x-h)/a)+k
                        
                        def catenaria(x, a, b, c, d):
                            return a + b*x + c*x**2 + d*x**3

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

                        # p0 = [1, 0, 0]  # a, h, k
                        
                        p0 = [0, 1, 1, 1]  

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
                        
                        ########################## TONI
                        from scipy.stats import pearsonr, spearmanr
                        
                        RMSE1_z = np.sqrt(np.mean((fitted_z_vals_scaled1 - z_vals_scaled1)**2))
                        max1_z = np.sqrt(np.max((fitted_z_vals_scaled1 - z_vals_scaled1)**2))
                        pearson1_z, sig = pearsonr(fitted_z_vals_scaled1, z_vals_scaled1) 
                        spearman1_z, p_value = spearmanr(fitted_z_vals_scaled1, z_vals_scaled1) 
                        # RMSE1_y = np.sqrt(np.mean((fitted_y_scaled1 - y_vals_scaled1)**2))
                        
                        RMSE2_z = np.sqrt(np.mean((fitted_z_vals_scaled2 - z_vals_scaled2)**2))
                        max2_z = np.sqrt(np.max((fitted_z_vals_scaled2 - z_vals_scaled2)**2))
                        pearson2_z, sig = pearsonr(fitted_z_vals_scaled2, z_vals_scaled2) 
                        spearman2_z, p_value = spearmanr(fitted_z_vals_scaled2, z_vals_scaled2) 
                        # RMSE2_y = np.sqrt(np.mean((fitted_y_scaled2 - y_vals_scaled2)**2))
                        
                        RMSE3_z = np.sqrt(np.mean((fitted_z_vals_scaled3 - z_vals_scaled3)**2))
                        max3_z = np.sqrt(np.max((fitted_z_vals_scaled3 - z_vals_scaled3)**2))
                        pearson3_z, sig = pearsonr(fitted_z_vals_scaled3, z_vals_scaled3) 
                        spearman3_z, p_value = spearmanr(fitted_z_vals_scaled3, z_vals_scaled3) 
                        # RMSE3_y = np.sqrt(np.mean((fitted_y_scaled3 - y_vals_scaled3)**2))
                        
                        print(f"Fit error for z coordinate: {RMSE1_z}, {RMSE2_z}, {RMSE3_z}")
                        print(f"Fit Pearson R for z coordinate: {pearson1_z}, {pearson2_z}, {pearson3_z}")
                        print(f"Fit Spearman R for z coordinate: {spearman1_z}, {spearman2_z}, {spearman3_z}")
                        # print(f"Fit error for y coordinate: {RMSE1_y}, {RMSE2_y}, {RMSE3_y}")
                        
                        rmses.append([RMSE1_z, RMSE2_z, RMSE3_z])
                        maxes.append([max1_z,max2_z,max3_z])
                        correlations.append([spearman1_z, spearman2_z, spearman3_z])
                
                        #########################
                        
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
                
                        x_fit1 = np.repeat(pd.Series(x1.flatten()).quantile(0.5),1000)
                        x_fit2 = np.repeat(pd.Series(x2.flatten()).quantile(0.5),1000)
                        x_fit3 = np.repeat(pd.Series(x3.flatten()).quantile(0.5),1000)
                        # print(x_pol1)
                        # print(y_pol1)
                        # print(x_fit1)
                        fit1=np.vstack((x_fit1, x_pol1, y_pol1))
                        fit2=np.vstack((x_fit2, x_pol2, y_pol2))
                        fit3=np.vstack((x_fit3, x_pol3, y_pol3))
                        
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
                                # plot_vano('{} Bad_Orientation{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                                pass
                        else:
                            dataf['flag'].append('bad_cluster')
                            if any([plot_filter=='bad_cluster',plot_filter=='all']):
                                # plot_vano('{} Incomplete_cluster{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                                pass
                    elif bad_fit==1:
                        dataf['flag'].append('bad_fit')
                        if any([plot_filter=='bad_fit',plot_filter=='all']):
                            # plot_vano('{} Bad_fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                            pass
                    elif good==1:
                        dataf['flag'].append('good_fit')
                        if any([plot_filter=='good_fit',plot_filter=='all']):
                            pass
                            # plot_vano('{} Good_Fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                            # print(cond_values[0])
                            # print(cond_values[1])
                            # print(cond_values[2])
                            # plot_fit_2('{} Good_Fit{}'.format(idv,' '+finc[0]),cond_values, apoyo_values, vert_values,fits)

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
    return data, rmses, maxes, correlations
                
                
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
                outliers removed and indices reset.xº
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

def pretreatment_linegroup_from_json(df):
    """
    Preprocess and clean the parameters data for line groups from a DataFrame.

    This function performs outlier removal on the DataFrame using the interquartile range (IQR) method.
    It processes the columns 'a0', 'a1', and 'a2', and removes rows where the values fall outside of
    1.5 times the IQR from the first and third quartiles. The cleaned DataFrame is then returned for
    further analysis.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the parameters for different line groups, including columns
                    such as 'a0', 'a1', and 'a2'.
    Returns:
    pd.DataFrame: A cleaned DataFrame containing the parameters for different line groups, with
                outliers removed and indices reset.
    """
    dfd=df.dropna().copy()
    for i in  [  'a0', 'a1', 'a2']:
        IQR=dfd[i].quantile(0.75)-dfd[i].quantile(0.25)
        dfd=dfd.loc[(dfd[i]>dfd[i].quantile(0.25)-1.5*IQR)&(dfd[i]<dfd[i].quantile(0.75)+1.5*IQR),:]
    dfd=dfd.reset_index()
    return dfd

def data_catenaryparameters_to_df(data):
    """
    Convert catenary parameters from the data into a DataFrame.

    This function processes the provided data to extract catenary parameters for each span (vano).
    It constructs a DataFrame containing the span ID and the parameters 'a0', 'a1', and 'a2'.
    If a parameter is not present for a specific span, it is filled with NaN.

    Parameters:
    data (list of dicts): The data containing information about different spans. Each dictionary should
                        contain keys like 'ID_VANO' and 'CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'.
    Returns:
    pd.DataFrame: A DataFrame containing the span IDs and the catenary parameters 'a0', 'a1', and 'a2'.
    """
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


