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

def scale_conductor(X):

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
    return labels, centroids


def catenaria(x, a, h, k):
    x = np.asarray(x).flatten()
    r=a * np.cosh((x - h) / a) + k
    return r


def fit_vano(data,sublist=[]):

    parameters=[]
    incomplete_vanos = []
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
                for lab in np.unique(labels):
                    
                    clust = X_scaled[:,labels == lab]
                    proportion = clust.shape[1]/total_points

                    if proportion< 0.15:
                        if idv not in incomplete_vanos:
                            incomplete_vanos.append(idv)
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
            else:
                incomplete_vanos.append(idv)
            
            if idv not in incomplete_vanos:
                parameters.append([idv]+parameters_vano)

    return parameters,incomplete_vanos

def data_middlepoints(data):
    x=[]
    y=[]
    ids_single_backing = []
    ids = []
    

    for iel, el in enumerate(data):
        if len(data[iel]['APOYOS']) >= 2:
            ids.append(data[iel]['ID_VANO'])
            y.append((data[iel]['APOYOS'][0]['COORDEANDA_Y'] + data[iel]['APOYOS'][1]['COORDEANDA_Y']) / 2)
            x.append((data[iel]['APOYOS'][0]['COORDENADA_X'] + data[iel]['APOYOS'][1]['COORDENADA_X']) / 2)
        elif len(data[iel]['APOYOS']) == 1:
            ids_single_backing.append(data[iel]['ID_VANO'])
            
        else:
            ids_single_backing.append(data[iel]['ID_VANO'])
            print(f"Error: No se encontraron apoyos válidos para el elemento {iel}.")
    
    scaler_x=StandardScaler()
    scaler_y=StandardScaler()
    x=scaler_x.fit_transform(np.array(x).reshape(-1,1))
    y=scaler_y.fit_transform(np.array(y).reshape(-1,1))
    X=pd.DataFrame({'ids':ids,'x':x.flatten(),'y':y.flatten()})

    return ids_single_backing,X

def flatten_sublist(sublist):
    flat_list = [sublist[0]] 
    for array in sublist[1:]:
        flat_list.extend(array.tolist()) 
    return flat_list

def pretreatment_linegroup(parameters):
    flattened_data = [flatten_sublist(sublist) for sublist in parameters]
    columns = ['ID', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    df = pd.DataFrame(flattened_data, columns=columns)
    dfd=df.dropna().copy()
    for i in  [ 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']:
        
        IQR=dfd[i].quantile(0.75)-dfd[i].quantile(0.25)
        dfd=dfd.loc[(dfd[i]>dfd[i].quantile(0.25)-1.5*IQR)&(dfd[i]<dfd[i].quantile(0.75)+1.5*IQR),:]
    
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
    print(f'LEN {np.unique(labels)}')
    for lbl in np.unique(labels):
        
        idval_subg=X.loc[labels==lbl,'ids'].to_list()
        
        parameters,incomplete_vanos=fit_vano(data,sublist=idval_subg)
        
        dfd=pretreatment_linegroup(parameters)
        
        print(f'\nVanos analizados:{dfd.shape[0]}')
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


