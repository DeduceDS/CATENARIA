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
    
    x_vals = [extremos_apoyos[0]["COORDENADA_X"], extremos_apoyos[0]["COORDENADA_X"], extremos_apoyos[1]["COORDENADA_X"], extremos_apoyos[1]["COORDENADA_X"]]
    y_vals = [extremos_apoyos[0]["COORDEANDA_Y"], extremos_apoyos[0]["COORDEANDA_Y"], extremos_apoyos[1]["COORDEANDA_Y"], extremos_apoyos[1]["COORDEANDA_Y"]]
    z_vals = extremos_apoyos[0]["COORDENADAS_Z"] + extremos_apoyos[1]["COORDENADAS_Z"]

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


def plot_data(cond_values, apoyo_values, vert_values, extremos_values):

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

def rotate_vano(cond_values, extremos_values, apoyo_values, vert_values):

    # Rotate and compare
    mat, rotated_conds = rotate_points(cond_values, extremos_values)

    rotated_apoyos = mat.dot(apoyo_values)
    rotated_extremos = mat.dot(extremos_values)
    rotated_vertices = [mat.dot(vert) for vert in vert_values]
    
    return rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos


def clean_outliers(rotated_conds, rotated_extremos):

    # Get top and bottom extreme values
    top = np.max([rotated_extremos.T[1][2],rotated_extremos.T[3][2]])
    bottom = np.max([rotated_extremos.T[0][2],rotated_extremos.T[2][2]])

    #Get left and right extreme values
    left = np.max([rotated_extremos.T[2][1],rotated_extremos.T[3][1]])
    right = np.min([rotated_extremos.T[0][1],rotated_extremos.T[1][1]])

    # Filter points within the specified boundaries
    cropped_conds = rotated_conds[:, (right > rotated_conds[1,:]) & (rotated_conds[1,:] > left)]
    cropped_conds = cropped_conds[:, (top > cropped_conds[2,:]) & (cropped_conds[2,:] > bottom)]
    # Calcular percentiles 1 y 99

    p1 = np.percentile(cropped_conds[1, :], 1)
    p99 = np.percentile(cropped_conds[1, :], 98)

    # Filtrar los datos para eliminar el 1% de los puntos con menor y mayor coordenada X
    cropped_conds = cropped_conds[:,(cropped_conds[1, :] > p1) & (cropped_conds[1, :] < p99)]

    print(f"Número de puntos después de eliminar los extremos: {(rotated_conds.shape[1])} vs {(cropped_conds.shape[1])}")
    
    return cropped_conds

def scale_conductor(X):

    # Normalizzazione dei valori di x e y
    scaler_y = StandardScaler()
    scaler_x = StandardScaler()

    y_vals_scaled = scaler_y.fit_transform(X[1,:].reshape(-1, 1)).flatten()
    x_vals_scaled = scaler_x.fit_transform(X[0,:].reshape(-1, 1)).flatten()  # Flatten per curve_fit

    X_scaled = np.array([x_vals_scaled, y_vals_scaled])
    
    return X_scaled


# if __name__ == "__main__":
#     main()