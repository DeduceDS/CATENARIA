
import math
import matplotlib.pyplot as plt
import re

from electra_package.puntuacion import *
from electra_package.modules_clustering import *

#### FUNCTIONS TO PROCESS JSON DATA ####

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
    
    vano_length = data[vano]["LONGITUD_2D"]
    idv = data[vano]["ID_VANO"]

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

    return idv, vano_length, cond_values, apoyo_values, vert_values, extremos_values


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

#### FUNCTIONS TO MANIPULATE ARRAYS ####

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

def show_extremos_linea(data):
    
    all_extremos = []

    for i in range(len(data)):
        
        try:
            cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
        except:
            print_element(data[i])
        
        all_extremos.append(extremos_values)
        
        
    first_extremos = []
    last_extremos = []

    plt.figure(figsize=(12,8))

    for i in range(len(all_extremos)):
        first_extremos.append(np.array([all_extremos[i][0][0], all_extremos[i][1][0], all_extremos[i][2][0]]))
        try:
            last_extremos.append(np.array([all_extremos[i][0][2], all_extremos[i][1][2], all_extremos[i][2][0]]))
        except Exception:
            pass
        
    first_extremos = np.array(first_extremos)
    last_extremos = np.array(last_extremos)

    plt.scatter(first_extremos[:,0], first_extremos[:,1], color = "blue", s = 8)
    plt.scatter(last_extremos[:,0], last_extremos[:,1], color = "red", s = 8, marker=5)

    for i in range(len(all_extremos)):
        plt.text(first_extremos[i,0]+20, first_extremos[i,1]+20, s = str(data[i]["OBJECTID_VANO_2D"]), size = 8, color = "brown")

    plt.tight_layout()
    plt.show()
    
    return first_extremos, last_extremos


def extract_apoyos_lidar(data):

    # Separmos los apoyos lidar
    apoyos_separados = dict()

    for i in range(len(data)):

        print(f"\nProcessing vano {i}")

        cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
        
        invertedxy = np.zeros((np.array(apoyo_values).shape))

        invertedxy[1,:] = (np.array(apoyo_values))[0,:]
        invertedxy[0,:] = (np.array(apoyo_values))[1,:]
        invertedxy[2,:] = (np.array(apoyo_values))[2,:]
        
        labels, centroids = kmeans_clustering(invertedxy, 2, 100)

        apoyos = []
        extremos = []
        

        for lab in np.unique(labels):

            apoyo = np.array(apoyo_values)[:, labels == lab]

            mean_x = np.mean(apoyo[0,:])
            mean_y = np.mean(apoyo[1,:])
            mean_z = np.mean(apoyo[2,:])
            
            c_mass = np.array([mean_x, mean_y, mean_z])
            extremos.append(c_mass)
            apoyos.append(apoyo)


        dist = np.linalg.norm(np.array(extremos)[0,:] - np.array(extremos)[1,:])
        extremos = np.array(extremos).T
        apoyos_separados[i] = apoyos

        print(f"Distance between mean points: {dist}")

        if 100*abs(dist - data[i]["LONGITUD_2D"])/data[i]["LONGITUD_2D"] > 10.0:
            
            points = np.array(apoyo_values)
            
            labels, centroids = kmeans_clustering(points, 2, 100)
                
            apoyos = []
            extremos = []

            for lab in np.unique(labels):

                apoyo = np.array(apoyo_values)[:, labels == lab]

                mean_x = np.mean(apoyo[0,:])
                mean_y = np.mean(apoyo[1,:])
                mean_z = np.mean(apoyo[2,:])
                
                c_mass = np.array([mean_x, mean_y, mean_z])
                extremos.append(c_mass)
                apoyos.append(apoyo)
            
            print("Invertir coordenadas")
            
            dist = np.linalg.norm(np.array(extremos)[0,:] - np.array(extremos)[1,:])
            extremos = np.array(extremos).T
            apoyos_separados[i] = apoyos
            
            if 100*abs(dist - data[i]["LONGITUD_2D"])/data[i]["LONGITUD_2D"] > 10.0:
                
                print(f"Proportional absolut error of distance = {100*abs(dist - data[i]['LONGITUD_2D'])/data[i]['LONGITUD_2D']}")
                print("SOLO HAY 1 APOYO")
                
                # apoyos_separados[i] = []
                
                plt.scatter(points[0], points[1], c=labels, cmap='viridis', s=1)
                plt.vlines(centroids, ymin=np.min(points[1]), ymax=np.max(points[1]), color='red')
                plt.title('Custom 1D K-means Clustering')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.show()
                continue
            
    return apoyos_separados

def extract_parejas(data, plot_errors, apoyos_separados):

    parejas = []
    parejas_ids = []

    for j in range(len(data)):
        apoyos_1 = apoyos_separados[j]

        for i in range(j + 1, len(data)):  # Ensure i != j by starting at j + 1
            found = False
            apoyos_2 = apoyos_separados[i]

            if len(apoyos_1) == 1:
                print(f"El vano {j} solo tiene 1 apoyo")

            if len(apoyos_2) == 1:
                print(f"El vano {i} solo tiene 1 apoyo")

            for k in range(len(apoyos_1)):
                for h in range(len(apoyos_2)):
                    points1 = np.array(apoyos_1[k])
                    points2 = np.array(apoyos_2[h])

                    try:
                        cond = (points1 == points2)

                        if cond.any():
                            cond = ~cond

                            if cond.sum() != 0:
                                rmse = np.sqrt(np.mean(np.square(points1 - points2)))
                                    
                                print(f"Some points are the same but not all in {i,j}")
                                print(f"Asociated RMSE: {rmse}")
                                
                                if plot_errors:
                                
                                    print(cond.sum(axis = 0).sum(axis = 0))
                                    print(f"{points1.shape} vs {points1.shape}")
                                
                                    plt.scatter(points1[0], points1[1], c = "red", s=1)
                                    plt.scatter(points2[0], points2[1], c = "green", s=1)
                                    plt.title('Custom 1D K-means Clustering')
                                    plt.xlabel('X Coordinate')
                                    plt.ylabel('Y Coordinate')
                                    plt.show()

                                if rmse < 10:
                                    found = True
                                else:
                                    print(f"Pareja {i,j} not validated, rmse too big!")
                            else:
                                found = True

                        if found:
                            print(f"Apoyo found {i,j}")
                            parejas.append((i, j))
                            parejas_ids.append((data[i]["ID_VANO"], data[j]["ID_VANO"]))
                            break

                    except Exception as e:
                        # print(f"Error processing points {points1} and {points2}: {e}")
                        continue
                
                if found:
                    break

            # if found:
            #     break

        if not found:
            print(f"Apoyo comun not found in {i,j}")
            
    return parejas, parejas_ids

def evaluar_clasificar_parejas(data, parejas):
    
    flat_data = np.array(parejas).flatten()
    conjunto = set(flat_data)
    
    print(f"Total parejas, total vanos: {len(parejas)},  {len(conjunto)}")
    
    # Encontrar los vanos que no conectan = discontinuidades
    aislados = []
    aislados_ids = []

    for i in range(len(data)):
        
        if i not in conjunto:
            # print(f"Aislado: {i}")
            
            aislados.append(i)
            aislados_ids.append(data[i]["ID_VANO"])
            
    # Encontrar los vanos que aparecen 2 veces = tenemos los dos apoyos
    completos = []
    incompletos = []

    completos_ids = []
    incompletos_ids = []

    for i in range(len(data)):
        
        if i in flat_data:
            if sum(i == flat_data) > 1:
                # print(f"Completo: {i}")
                completos.append(i)
                completos_ids.append(data[i]["ID_VANO"])
            else:
                # print(f"Incompleto: {i}")
                incompletos.append(i)
                incompletos_ids.append(data[i]["ID_VANO"])
                
    # Conjuntos totalmente independientes
    print(f"Vanos completos: {completos}")
    print(f"Vanos incompletos: {incompletos}")
    print(f"Vanos aislados: {aislados}")

    print(f"\nParejas que comparten apoyo{parejas}\n")
    
    print(f"{100*len(completos)/len(data)}% de vanos completos")
    print(f"{100*len(incompletos)/len(data)}% de vanos incompletos")
    print(f"{100*len(aislados)/len(data)}% de vanos aislados")
    
    return aislados, incompletos, completos, aislados_ids, incompletos_ids, completos_ids
                
def look_for_vano(data, vano_id):
    
    for i,vano in enumerate(data):
        if vano["ID_VANO"] == vano_id:
            return i,vano
    
    print("Vano not found")
    return 0,0

def get_bad_ids(path):

    # Leer el contenido del archivo de texto
    with open(path, 'r', encoding='utf-8') as file:
        texto = file.read()

    # Expresión regular para encontrar los IDs que empiezan con "G_"
    pattern1 = r'\bG_\d+_\d+\b'
    pattern2 = r'\bC_\d+_\d+\b'

    # Encontrar todos los IDs que coinciden con la expresión regular
    ids = re.findall(pattern1, texto) + re.findall(pattern2, texto)

    # Mostrar los IDs encontrados
    return ids
