import numpy as np
from loguru import logger
import numpy as np
import time

from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

from electra_package.modules_preprocess import *


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
    
    return RMSE_z, max_z, pearson_z, spearman_z, p_value
                        

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
    
    RMSE_z, max_z, pearson_z, spearman_z, p_value = get_metrics(fitted_z_vals_scaled, z_vals_scaled)
    
    return y_pol, fitted_z_pol, parametros, [RMSE_z, max_z, pearson_z, spearman_z, p_value]


def stack_unrotate_fits(pols, mat):

    fit1=np.vstack((pols[0][0], pols[1][0], pols[2][0]))
    fit2=np.vstack((pols[0][1], pols[1][1], pols[2][1]))
    fit3=np.vstack((pols[0][2], pols[1][2], pols[2][2]))

    mat_neg,fit1=un_rotate_points(fit1,mat)
    mat_neg,fit2=un_rotate_points(fit2,mat)
    mat_neg,fit3=un_rotate_points(fit3,mat)
    
    return fit1, fit2, fit3

############################ NOT IN RELEASE #########################################

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


def filtering_prefit_2(rotated_conds, extremos_values):
    X_extremos, Y_extremos, Z_extremos = extremos_values
    X_cond, Y_cond, Z_cond = rotated_conds

    mask = (Y_cond > np.min(Y_extremos)) & (Y_cond < np.max(Y_extremos))
    x = X_cond[mask]
    y = Y_cond[mask]
    z = Z_cond[mask]

    return x, y, z
