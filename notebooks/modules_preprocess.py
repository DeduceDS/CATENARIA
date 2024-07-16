
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import open3d as o3d
from puntuacion import *
from modules_utils import *
from modules_clustering import *
from loguru import logger

def get_bad_data(pathdata):
    
    bad_ids = get_bad_ids(pathdata.split("json")[0]+"txt")

    with open(pathdata, 'r') as archivo:
        data = json.load(archivo)
        
    # print(len(data), bad_ids)

    bad_data_df = extract_preprocess_errors(data, bad_ids)

    bad_data = [data[i] for i in bad_data_df["num"]]

    return bad_data_df, bad_data

def extract_preprocess_errors(data, bad_ids):

    succeed_preprocess = []
    failed_preprocess = []
    errors = []
    num = []

    for i in range(len(data)):
        
        print(f"\nProcessing Vano {i}")
        
        vano_id = data[i]['ID_VANO']
        
        # if vano_id in bad_ids:
        #     print("Vano skiped ", i)
        #     continue
        
        try:
            
            cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
            rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
            succeed_preprocess.append(vano_id)
            
        except Exception as e:
            
            failed_preprocess.append(vano_id)
            errors.append(e)
            num.append(i)
            print(f"Vano {vano_id} failed preprocess: {e}")
            
def get_new_extreme_values(data):

    nuevos_extremos = []
    nuevos_extremos_ids = []
    un_apoyo_ids = []

    for i in range(len(data)):
        
        print(f"\nProcessing vano {i}")
        
        cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)

        print(f"We lack of extreme values: {len(extremos_values[2]) != 4}")
        
        # Standard scaling
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(np.array(apoyo_values))
        
        labels, centroids = kmeans_clustering(scaled_points, 2, 500)
        
        points = scaler.inverse_transform(scaled_points)
        
        extremos = []

        for lab in np.unique(labels):

            apoyo = points[:, labels == lab]

            mean_x = np.mean(apoyo[0,:])
            mean_y = np.mean(apoyo[1,:])
            mean_z = np.mean(apoyo[2,:])
            
            c_mass = np.array([mean_x, mean_y, mean_z])
            extremos.append(c_mass)
        

        dist = np.linalg.norm(np.array(extremos)[0,:] - np.array(extremos)[1,:])
        extremos = np.array(extremos).T
        
        print(f"Distance between mean points: {dist}")
        # print(f"New extreme values: {extremos}")
        
        if 100*abs(dist - data[i]["LONGITUD_2D"])/data[i]["LONGITUD_2D"] > 10.0:
            
            print(f"Proportional absolut error of distance = {100*abs(dist - data[i]['LONGITUD_2D'])/data[i]['LONGITUD_2D']}")
            print("SOLO HAY 1 APOYO")
            
            plt.scatter(points[0], points[1], c=labels, cmap='viridis', s=1)
            plt.scatter(extremos[0,:], extremos[1,:], s = 10, color = "blue")
            # plt.vlines(centroids, ymin=np.min(points[1]), ymax=np.max(points[1]), color='red')
            plt.title('Custom 1D K-means Clustering')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.show()
                
            # plot_data("test",cond_values, apoyo_values, vert_values, extremos)
            
            un_apoyo_ids.append(data[i]["ID_VANO"])
            continue
        
        nuevos_extremos.append(extremos)
        nuevos_extremos_ids.append(data[i]["ID_VANO"])
        # plot_data("test",cond_values, apoyo_values, vert_values, extremos)
    return np.array(nuevos_extremos), nuevos_extremos_ids, un_apoyo_ids


def mod_extremos(data,new_extremos,ids):
    
    for id in ids:
        
        idx, vano = look_for_vano(data,id)
        new_apoyos = []
        
        for j in range(2):
            
            new_apoyos.append({"COORDENADA_X": list(new_extremos[0,j]), "COORDENADA_Y": list(new_extremos[1,j]), "COORDENADAS_Z": list(new_extremos[2,j])})
        # print(vano["APOYOS"])
        vano['APOYOS'] = new_apoyos
        # print(new_apoyos)
        data[idx] = vano
            
    return data

#### FUNCTIONS TO TRANSFORM/PREPROCESS 3D POINTS ####

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
    
    # print(f"Shape 0: {np.array(rotated_extremos).shape}")
    
    #Get left and right extreme values
    left = np.max([rotated_extremos.T[2][1],rotated_extremos.T[3][1]])
    right = np.min([rotated_extremos.T[0][1],rotated_extremos.T[1][1]])

    # print(right-left)
    # print(rotated_extremos)
    
    # Filter points within the specified boundaries
    cropped_conds = rotated_conds[:, (right > rotated_conds[1,:]) & (rotated_conds[1,:] > left)]
    
    # print(f"Shape 1: {cropped_conds.shape}")

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
        logger.debug(f"Umbral de corte superior detectado: {threshold_y_upper}")

    # Verificar si hay una línea horizontal significativa en la parte inferior
    if hist[peak_bin_lower] > threshold_density:
        threshold_y_lower = bin_edges[peak_bin_lower]  # No se necesita ajustar más
        logger.debug(f"Umbral de corte superior detectado: {threshold_y_lower}")

    # Paso 3: Filtrar los puntos usando los umbrales detectados
    if threshold_y_upper is not None:
        cropped_conds = cropped_conds[:, cropped_conds[1, :] > threshold_y_upper]

    if threshold_y_lower is not None:
        cropped_conds = cropped_conds[:, cropped_conds[1, :] < threshold_y_lower]
        
    # print(f"Shape 2: {cropped_conds.shape}")
    
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
    
    nn = 4 # Local search
    std_multip = 2 # Not very sensitive

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

#### FUNCTIONS TO MODIFY/CORRECT ORIGINAL DATA ####

def num_apoyos_LIDAR(data, vano):
    puntos_apoyos = data[vano]['LIDAR']['APOYOS']

    x_vals_apoyos, y_vals_apoyos, z_vals_apoyos = get_coord(puntos_apoyos)

    X = np.column_stack((y_vals_apoyos, z_vals_apoyos))
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(X)

    centros = kmeans.cluster_centers_
    dif = 2
    if centros[0][0] - dif <= centros[1][0] < centros[0][0] + dif:
        return 1
    else:
        return 2

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

def define_backings(vano_length, apoyo_values):
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
                    
    invertedxy = np.zeros((np.array(apoyo_values).shape))
    invertedxy[1,:] = (np.array(apoyo_values))[0,:]
    invertedxy[0,:] = (np.array(apoyo_values))[1,:]
    invertedxy[2,:] = (np.array(apoyo_values))[2,:]
    
    labels, centroids = kmeans_clustering(invertedxy, 2, 100)

    apoyos = []
    extremos = []
    
    for lab in np.unique(labels):

        apoyo = np.array(apoyo_values)[:, labels == lab]

        mean_x = np.mean(apoyo[1,:])
        mean_y = np.mean(apoyo[0,:])
        max_z = np.max(apoyo[2,:])
        min_z = np.min(apoyo[2,:])
        
        c_mass1 = np.array([mean_x, mean_y, min_z])
        c_mass2 = np.array([mean_x, mean_y, max_z])
        
        extremos.append(c_mass1)
        extremos.append(c_mass2)
        
        apoyos.append(apoyo)

    dist = np.linalg.norm(np.array(extremos)[0,:] - np.array(extremos)[1,:])
    extremos = np.array(extremos)

    logger.debug(f"Distance between mean points: {dist}")

    if 100*abs(dist - vano_length)/vano_length > 10.0: # data[i]["LONGITUD_2D"]
        
        points = np.array(apoyo_values)
        
        labels, centroids = kmeans_clustering(points, 2, 100)
            
        apoyos = []
        extremos = []

        for lab in np.unique(labels):

            apoyo = np.array(apoyo_values)[:, labels == lab]

            mean_x = np.mean(apoyo[0,:])
            mean_y = np.mean(apoyo[1,:])
            max_z = np.max(apoyo[2,:])
            min_z = np.min(apoyo[2,:])
            
            c_mass1 = np.array([mean_x, mean_y, min_z])
            c_mass2 = np.array([mean_x, mean_y, max_z])
            
            extremos.append(c_mass1)
            extremos.append(c_mass2)
            
            apoyos.append(apoyo)
        

        logger.debug(f"Invertir coordenadas")
        
        dist = np.linalg.norm(np.array(extremos)[0,:] - np.array(extremos)[2,:])
        extremos = np.array(extremos)
        
        if 100*abs(dist - vano_length)/vano_length > 10.0:

            logger.debug(f"Proportional absolut error of distance = {100*abs(dist - vano_length)/vano_length}")
            logger.debug("SOLO HAY 1 APOYO")
            
            plt.scatter(points[0], points[1], c=labels, cmap='viridis', s=1)
            plt.vlines(centroids, ymin=np.min(points[1]), ymax=np.max(points[1]), color='red')
            plt.title('Custom 1D K-means Clustering')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.show()
            
            return -1

    print(extremos.shape)
    # z_vals = np.stack([np.array(extremos)[0,2], np.array(extremos)[1,2], np.array(extremos)[0,2], np.array(extremos)[1,2]])
    z_vals = np.stack([np.array(extremos)[2,2], np.array(extremos)[3,2], np.array(extremos)[0,2], np.array(extremos)[1,2]])
    y_vals =  np.stack([np.array(extremos)[2,1], np.array(extremos)[3,1], np.array(extremos)[0,1], np.array(extremos)[1,1]])
    x_vals =  np.stack([np.array(extremos)[2,0], np.array(extremos)[3,0], np.array(extremos)[0,0], np.array(extremos)[1,0]])
    
    extremos_values = [x_vals, y_vals, z_vals]
        
    return extremos_values