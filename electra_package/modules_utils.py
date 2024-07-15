
import math
from puntuacion import *

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