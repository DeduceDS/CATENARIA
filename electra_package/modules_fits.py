import numpy as np
from loguru import logger
import numpy as np
import time

from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as ma
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA


def catenaria(x, a, h, k):
    x = np.asarray(x).flatten()
    r=a * np.cosh((x - h) / a) + k
    return r

def poly3(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def get_metrics(fitted_z_vals_scaled, z_vals_scaled):
                    
    # RMSE_z = np.sqrt(np.mean((fitted_z_vals_scaled - z_vals_scaled)**2))
    
    RMSE_z = rmse(fitted_z_vals_scaled, z_vals_scaled)
    max_z = np.sqrt(np.max((fitted_z_vals_scaled - z_vals_scaled)**2))
    pearson_z, sig = pearsonr(fitted_z_vals_scaled, z_vals_scaled) 
    spearman_z, p_value = spearmanr(fitted_z_vals_scaled, z_vals_scaled) 
    # RMSE1_y = np.sqrt(np.mean((fitted_y_scaled1 - y_vals_scaled1)**2))
    
    # print(f"RmeanSE and RmaxSE error for z coordinate: {RMSE_z}, {max_z}")
    # print(f"Fit Pearson R and Fit Spearman R for z coordinate: {pearson_z}, {spearman_z}")
    # print(f"Fit error for y coordinate: {RMSE1_y}, {RMSE2_y}, {RMSE3_y}")
    
    return RMSE_z, max_z, pearson_z, spearman_z
                        
                                        
def clustering_prefit_1(x,y,z):
    
    x_cond = np.array(x)
    y_cond = np.array(y)
    z_cond = np.array(z)

    # Clustering
    # Combine the two coordinates into a 2D array for clustering
    [X, y] = [x_cond.reshape(-1, 1), y_cond.reshape(-1, 1)]

    model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0)

    y_spectral = model.fit_predict(X)

    # Separate the original data into clusters 1, 2, 3 = 3x3
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

    x1, y1, z1 = np.array(x1), np.array(y1), np.array(z1).reshape(-1,1)
    x2, y2, z2 = np.array(x2), np.array(y2), np.array(z2).reshape(-1,1)
    x3, y3, z3 = np.array(x3), np.array(y3), np.array(z3).reshape(-1,1)

    return [np.array([x1,y1,z1]),np.array([x2,y2,z2]),np.array([x3,y3,z3])]


def clustering_prefit_2(x, y, z):
    
    time1 = time.time()
    
    X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    
    model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0)
    y_spectral = model.fit_predict(X)
    
    time2 = time.time()
    logger.debug(f"clustering time 1 {time2-time1}")
    
    clusters = [[], [], []]
    for i in range(3):
        mask = (y_spectral == i)
        clusters[i] = np.vstack((x[mask], y[mask], z[mask]))
        
    time3 = time.time()
    logger.debug(f"clustering time 2 {time3-time2}")
    
    return clusters

    
def PCA_filtering_prefit_1(x, y, z):
    
    # PCA FILTERING
    data_2d_cond = np.column_stack((y, z))

    # Fit the PCA model and transfor the data to the 2D PCA space
    pca = PCA(n_components=2)
    data_2d_pca_cond = pca.fit_transform(data_2d_cond)
    
    # Extract min and max values of the conductor's PCA second component
    y_min_cond, y_max_cond = data_2d_pca_cond[:, 1].min(), data_2d_pca_cond[:,1].max()

    # Filter the values that fall between the min and max values of the second component
    f_ind = (data_2d_pca_cond[:,1] > y_min_cond) & (data_2d_pca_cond[:,1] < y_max_cond)
    x_filt_cond, y_filt_cond, z_filt_cond = x[f_ind], y[f_ind], z[f_ind]

    return x_filt_cond, y_filt_cond, z_filt_cond

def PCA_filtering_prefit_2(x, y, z):
    
    # Stack y and z coordinates for PCA
    data_2d_cond = np.column_stack((y, z))

    # Fit the PCA model and transform the data
    pca = PCA(n_components=2)
    data_2d_pca_cond = pca.fit_transform(data_2d_cond)
    
    # Extract min and max values of the second PCA component
    y_min_cond, y_max_cond = np.min(data_2d_pca_cond[:, 1]), np.max(data_2d_pca_cond[:, 1])

    # Filter the values that fall between the min and max values of the second component
    mask = (data_2d_pca_cond[:, 1] > y_min_cond) & (data_2d_pca_cond[:, 1] < y_max_cond)
    x_filt_cond = x[mask]
    y_filt_cond = y[mask]
    z_filt_cond = z[mask]

    return x_filt_cond, y_filt_cond, z_filt_cond

def filtering_prefit_1(rotated_conds, extremos_values):

    X_extremos = extremos_values[0]
    Y_extremos = extremos_values[1]
    Z_extremos = extremos_values[2]

    X_cond = rotated_conds[0]
    Y_cond = rotated_conds[1]
    Z_cond = rotated_conds[2]

    # Filtramos los puntos de los conductores cuya coord Y está entre los extremos cartografiados
    
    x = []
    y = []
    z = []

    for j in range(len(X_cond)):
        if Y_cond[j] > np.min(Y_extremos) and Y_cond[j] < np.max(Y_extremos):
            x.append(X_cond[j])
            y.append(Y_cond[j])
            z.append(Z_cond[j])
            
    return x,y,z


def filtering_prefit_2(rotated_conds, extremos_values):
    X_extremos, Y_extremos, Z_extremos = extremos_values
    X_cond, Y_cond, Z_cond = rotated_conds

    mask = (Y_cond > np.min(Y_extremos)) & (Y_cond < np.max(Y_extremos))
    x = X_cond[mask]
    y = Y_cond[mask]
    z = Z_cond[mask]

    return x, y, z


def fit_3D_coordinates(y_values, z_values, fit_function, initial_params):
    
    # Reshape and scale z, y coordinates
    y_vals = y_values.reshape(-1, 1)
    z_vals = z_values.reshape(-1, 1)

    scaler_y = StandardScaler()
    scaler_z = StandardScaler()

    y_vals_scaled = scaler_y.fit_transform(y_vals).flatten()
    z_vals_scaled = scaler_z.fit_transform(z_vals).flatten()
    
    # Fit curve to data and get the fitted parameters
    parametros, _ = curve_fit(fit_function, y_vals_scaled.flatten(), z_vals_scaled, initial_params)

    # With the fitted parameters extract the new corresponding Z coordinate for our Y values.
    fitted_z_vals_scaled = fit_function(y_vals_scaled.flatten(), *parametros)
    fitted_z_vals = scaler_z.inverse_transform(fitted_z_vals_scaled.reshape(-1, 1)).flatten()

    # Interpolación de la polilínea
    # Get min and max Y values in the original scale
    minimo = np.min(scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten())
    maximo = np.max(scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten())
    
    # Uniform array in Y coordinate from min to max
    x_pol = np.linspace(minimo, maximo, 200)

    # Reshape and scale the array
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_pol.reshape(-1, 1)).flatten()

    # With the fitted parameters extract the new corresponding Z coordinate for our new Y array.
    fitted_y_scaled = fit_function(x_scaled.flatten(), *parametros)
    fitted_y = scaler_z.inverse_transform(fitted_y_scaled.reshape(-1, 1)).flatten()

    # Interpolate the fitted values to the same length as x_pol for the 3D representation
    y_pol = np.interp(x_pol, scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten(), fitted_z_vals, period=len(fitted_z_vals))
    
    RMSE_z, max_z, pearson_z, spearman_z = get_metrics(fitted_z_vals_scaled, z_vals_scaled)
    
    return x_pol, y_pol, parametros, [RMSE_z, max_z, pearson_z, spearman_z]

def fit_3D_coordinates_2(y_values, z_values, fit_function, initial_params):
    
    # Reshape and scale z, y coordinates
    y_vals = y_values.reshape(-1, 1)
    z_vals = z_values.reshape(-1, 1)

    scaler_y = StandardScaler()
    scaler_z = StandardScaler()

    y_vals_scaled = scaler_y.fit_transform(y_vals).flatten()
    z_vals_scaled = scaler_z.fit_transform(z_vals).flatten()
    
    # Fit curve to data and get the fitted parameters
    parametros, _ = curve_fit(fit_function, y_vals_scaled.flatten(), z_vals_scaled, initial_params)

    # With the fitted parameters extract the new corresponding Z coordinate for our Y values.
    fitted_z_vals_scaled = fit_function(y_vals_scaled.flatten(), *parametros)
    fitted_z_vals = scaler_z.inverse_transform(fitted_z_vals_scaled.reshape(-1, 1)).flatten()

    # Interpolación de la polilínea
    # Generate interpolated y values for plotting
    # Get min and max Y values in the original scale
    min_y = np.min(scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten())
    max_y = np.max(scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten())
    
    # Uniform array in Y coordinate from min to max
    y_pol = np.linspace(min_y, max_y, 200)

    # Standardize the interpolated y values
    y_pol_scaled = scaler_y.transform(y_pol.reshape(-1, 1)).flatten()

    # Calculate the fitted z values for the interpolated y values
    fitted_z_pol_scaled = fit_function(y_pol_scaled, *parametros)
    fitted_z_pol = scaler_z.inverse_transform(fitted_z_pol_scaled.reshape(-1, 1)).flatten()

    # Interpolate the fitted values to the same length as y_pol for the 3D representation
    # z_pol = np.interp(y_pol, scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten(), fitted_z_vals, period=len(fitted_z_vals))
    
    RMSE_z, max_z, pearson_z, spearman_z = get_metrics(fitted_z_vals_scaled, z_vals_scaled)
    
    return y_pol, fitted_z_pol, parametros, [RMSE_z, max_z, pearson_z, spearman_z]