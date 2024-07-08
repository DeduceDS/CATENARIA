import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as ma
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA


def catenaria(x, a, h, k):
    x = np.asarray(x).flatten()
    r=a * np.cosh((x - h) / a) + k
    return r

def poly3(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def get_metrics(fitted_z_vals_scaled, z_vals_scaled):
                    
    RMSE_z = np.sqrt(np.mean((fitted_z_vals_scaled - z_vals_scaled)**2))
    max_z = np.sqrt(np.max((fitted_z_vals_scaled - z_vals_scaled)**2))
    pearson_z, sig = pearsonr(fitted_z_vals_scaled, z_vals_scaled) 
    spearman_z, p_value = spearmanr(fitted_z_vals_scaled, z_vals_scaled) 
    # RMSE1_y = np.sqrt(np.mean((fitted_y_scaled1 - y_vals_scaled1)**2))
    
    print(f"Fit error for z coordinate: {RMSE_z}")
    print(f"Max error for z coordinate: {max_z}")
    print(f"Fit Pearson R for z coordinate: {pearson_z}")
    print(f"Fit Spearman R for z coordinate: {spearman_z}")
    # print(f"Fit error for y coordinate: {RMSE1_y}, {RMSE2_y}, {RMSE3_y}")
    
    return RMSE_z, max_z, pearson_z
                        
                                        
def clustering_prefit_1(x,y,z):
    
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
    
    return np.array([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]])
    
def PCA_filtering_prefit_1(x, y, z):
    
    # PCA FILTERING
    data_2d_cond = np.column_stack((y, z))

    pca = PCA(n_components=2)
    data_2d_pca_cond = pca.fit_transform(data_2d_cond)

    y_min_cond, y_max_cond = data_2d_pca_cond[:, 1].min(), data_2d_pca_cond[:,1].max()

    f_ind = (data_2d_pca_cond[:,1] > y_min_cond) & (data_2d_pca_cond[:,1] < y_max_cond)

    x_filt_cond, y_filt_cond, z_filt_cond = x[f_ind], y[f_ind], z[f_ind]

    return x_filt_cond, y_filt_cond, z_filt_cond

def filtering_prefit_1(rotated_conds, extremos_values):
    
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
            
    return x,y,z


def fit_3D_coordinates(y_values, z_values, fit_function, initial_params):
    
    y_vals = y_values.reshape(-1, 1)
    z_vals = z_values.reshape(-1, 1)

    scaler_y = StandardScaler()
    scaler_z = StandardScaler()

    y_vals_scaled = scaler_y.fit_transform(y_vals).flatten()
    z_vals_scaled = scaler_z.fit_transform(z_vals).flatten()

    parametros, _ = curve_fit(fit_function, y_vals_scaled.flatten(), z_vals_scaled, initial_params)

    # Ajuste de los puntos de los datos a una catenaria
    fitted_z_vals_scaled = fit_function(y_vals_scaled.flatten(), *parametros)
    fitted_z_vals = scaler_z.inverse_transform(fitted_z_vals_scaled.reshape(-1, 1)).flatten()

    # Interpolación de la polilínea
    minimo = np.min(scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten())
    maximo = np.max(scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten())
    x_pol = np.linspace(minimo, maximo, 1000)

    scaler_x = StandardScaler()

    x_scaled = scaler_x.fit_transform(x_pol.reshape(-1, 1)).flatten()

    fitted_y_scaled = fit_function(x_scaled.flatten(), *parametros)
    fitted_y = scaler_z.inverse_transform(fitted_y_scaled.reshape(-1, 1)).flatten()

    y_pol = np.interp(x_pol, scaler_y.inverse_transform(y_vals_scaled.reshape(-1, 1)).flatten(), fitted_z_vals, period=len(fitted_z_vals))
    
    RMSE_z, max_z, pearson_z = get_metrics(fitted_z_vals_scaled)
    
    return x_pol, y_pol, parametros, [RMSE_z, max_z, pearson_z]