
from loguru import logger
from statistics import mode
from electra_package.modules_utils import *
from electra_package.modules_clustering import *
from electra_package.modules_preprocess import *
from electra_package.modules_plots import *
from electra_package.modules_fits import *
import time

def analyze_backings(vano_length, idv, cond_values, apoyo_values, vert_values, extremos_values, dataf=None):
    
    # Start the timer
    start_time1 = time.time()

    logger.warning(f"Redefining backings")
    
    # Redefine and compute new extreme values
    extremos_values = list(define_backings(vano_length,apoyo_values))
    
    end_time1 = time.time()
    
    logger.debug(f"Second time {end_time1-start_time1}")
    
    # Check for missing LIDAR apoyo points
    # Exception to handle = bad data , correction not possible
    if extremos_values == -1: # any(extremos_values == -1)
        
        # Include flag of bad extreme values
        # Set the line value of this element as 0 ****
        logger.warning("UN APOYO LIDAR")
        # plot_data(f"{idv}",cond_values, apoyo_values, vert_values, extremos_values)
        
        if dataf != None:
            
            dataf['flag'].append('bad_backing')
            dataf['line_number'].append(0)
        
        return -1
    
    # Plot filter to plot bad cases?
    # if any([plot_filter=='all',plot_filter=='bad_backing']):
        # plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
        
    return dataf, extremos_values

def preprocess_conductors(cond_values, extremos_values, apoyo_values, vert_values):
            
    start_time2 = time.time()
    
    logger.success(f"Rotating vano")
    # Preform rotation of all data 3D points over z axis to align the conductor diagonal with the Y axis.
    mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)

    ##########################################
    
    logger.success(f"Cropping conductor")
    # Crop conductor LIDAR points and clean outliers before any transformation
    cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
    cropped_conds = clean_outliers_2(cropped_conds) # More crop?
    cropped_conds = clean_outliers_3(cropped_conds)

    # outliers = pcd_o3d.select_by_index(filtered_points[1], invert=True)
    
    ##########################################
    
    # Scale conductor 3D points and get the scaler models
    X_scaled,scaler_x,scaler_y,scaler_z = scale_conductor(cropped_conds) # Scale now?
    
    end_time2 = time.time()
        
    logger.debug(f"Third time {end_time2-start_time2}")
    
    return X_scaled, cropped_conds

def extract_conductor_config(dataf, X_scaled, scaler_y, rotated_conds, rotated_extremos, cropped_conds):
    
    start_time2 = time.time()
    
    # Extract ymin and ymax values of rotated conductors ***
    rotated_ymin=min(rotated_conds[1])  # More extreme values?
    rotated_ymax=max(rotated_conds[1])
    
    # Scale ymin and ymax values
    rotated_ymax=scaler_y.transform(np.array([rotated_ymax]).reshape(-1, 1))[0]
    rotated_ymin=scaler_y.transform(np.array([rotated_ymin]).reshape(-1, 1))[0]

    # Conductor geometry/configuration analysis
    # Let's study and categorize this vano in terms of it's conductors
    
    logger.success(f"Analyzing conductor configuration")
    
    # Find clusters in 10 cloud of points corresponding to 3 positions in y axis
    # Define boundaries of the conductor to extract min, max values and length
    maxy=max(rotated_extremos[1,1],rotated_extremos[1,2])
    miny=min(rotated_extremos[1,1],rotated_extremos[1,2])
    leny=maxy-miny  # Equal to 2D length?
    
    # Filter and extract 10 10% length segments

    l=[] #Fragment values
    filt=[] # Index for fragments
    k=10
    
    for g in range(0,k):

        filt0=(cropped_conds[1,:]>(miny+g*(1/k)*leny))&(cropped_conds[1,:]<(miny+(g+1)*(1/k)*leny))
        l0=cropped_conds[:,filt0].shape[1]
        l.append(l0)
        filt.append(filt0)
        
    # Calculate: n points of each segment, the variance difference between x and z coordinates
    # Find clusters for segments that have more than 20 points
    # If the variance in z is greater than x then append this result
    
    c=[] # Centroid values resulting from clustering
    greater_var_z_than_x=[] # Bool list with x vs z variance relation
    ncl=[] # Number of centroids from clustering

    for g in range(0,k):

        l0=X_scaled[:,filt[g]].shape[1]
        fl=pd.Series(X_scaled[2,filt[g]]).var()-pd.Series(X_scaled[0,filt[g]]).var()
        centroids0,labels0=dbscan_find_clusters_3(X_scaled[:,filt[g]]) if l0>20 else ([],[])
        greater_var_z_than_x.append(True if fl>=0 else False)
        c.append(centroids0)
        ncl.append(len(centroids0))
        
    ##########################################
    # Obtain the mode of n_clusters along the list of size 10 == number of conductors
    md=mode(ncl)
    
    # Save the final number of conductors detected (number of lines)
    dataf['line_number'].append(md)
    logger.info(f'Number of lines: {md}')
    
    # Do the same for the x vs z variance relation
    var_z_x=mode(greater_var_z_than_x)
    
    # Compute the number of empty fragments
    num_empty=np.array([l0<20 for l0 in l]).sum()
    
    # Define 3 completeness categories
    completeness=np.array(['incomplete','partially incomplete','full'])
    
    # Define the index with the completeness conditions
    completeness_conds=np.array([num_empty>5,all([num_empty<=5,num_empty>=2]),num_empty<2])
    # Extract final value with index
    finc = completeness[completeness_conds]
    
    end_time2 = time.time()
    
    logger.debug(f"Third time {end_time2-start_time2}")

    return dataf, finc


def preprocess_prefit(cond_values):
    end_time3 = time.time()
    
    logger.success(f"Starting fit")
    
    logger.success(f"Rotating vano")
    
    # Get rotated cond points and extreme values again?
    mat, rotated_conds = rotate_points(cond_values, extremos_values)
    extremos_values = mat.dot(extremos_values)
    
    filter_start = time.time()
    
    logger.success(f"Filtering conductor")
    
    # Filter conductor values between extreme values again?
    x, y, z = filtering_prefit_2(rotated_conds, extremos_values)
    
    filter_end = time.time()
    
    logger.debug(f"Filtering time {filter_end-filter_start}")
    
    # x, y, z = np.array(rotated_conds)[0], np.array(rotated_conds)[1], np.array(rotated_conds)[2]
    
    ########################
    
    logger.success(f"Applying spectral clustering")
    
    cluster_start = time.time()

    # Clustering over cond values to extract each conductor individually
    clusters = clustering_prefit_2(x,y,z)
    
    x1, y1, z1 = clusters[0][0,:], clusters[0][1,:], clusters[0][2,:]
    x2, y2, z2 = clusters[1][0,:], clusters[1][1,:], clusters[1][2,:]
    x3, y3, z3 = clusters[2][0,:], clusters[2][1,:], clusters[2][2,:]
    
    cluster_end = time.time()
    
    logger.debug(f"clustering time {cluster_end-cluster_start}")

    ################
    
    logger.success(f"PCA filtering")
    
    pca_start = time.time()
    
    # PCA filtering over the conductor values
    # x_filt_cond1, y_filt_cond1, z_filt_cond1 = PCA_filtering_prefit_2(x1, y1, z1)
    # x_filt_cond2, y_filt_cond2, z_filt_cond2 = PCA_filtering_prefit_2(x2, y2, z2)
    # x_filt_cond3, y_filt_cond3, z_filt_cond3 = PCA_filtering_prefit_2(x3, y3, z3)
    
    x_filt_cond1, y_filt_cond1, z_filt_cond1 = x1, y1, z1
    x_filt_cond2, y_filt_cond2, z_filt_cond2 = x2, y2, z2
    x_filt_cond3, y_filt_cond3, z_filt_cond3 = x3, y3, z3
    
    pca_end = time.time()
    
    logger.debug(f"PCA time {pca_end-pca_start}")
    
    #############################
    
    end_time4 = time.time()
    
    logger.debug(f"Fifth time {end_time4-end_time3}")
                        
    return x_filt_cond1, y_filt_cond1, z_filt_cond1, x_filt_cond2, y_filt_cond2, z_filt_cond2, x_filt_cond3, y_filt_cond3, z_filt_cond3

def fit_reconstruct_evaluate_conductors(y_filt_cond1, z_filt_cond1, 
                                        y_filt_cond2, z_filt_cond2, 
                                        y_filt_cond3, z_filt_cond3,
                                        rmses, maxes, correlations, evaluaciones, idv,
                                        rotated_vertices, vano_length, clusters):
    
    end_time4 = time.time()
    
    def catenaria(x, a, h, k):
        return a*np.cosh((x-h)/a)+k
    
    # def catenaria(x, a, b, c, d):
    #     return a + b*x + c*x**2 + d*x**3
    
    p0 = [1, 0, 0]  # a, h, k
    
    # p0 = [0, 1, 1, 1]
    
    logger.success(f"Fitting catenaria to data")
    
    x_pol1, y_pol1, parametros1, metrics1 = fit_3D_coordinates(y_filt_cond1, z_filt_cond1, catenaria, p0)
    x_pol2, y_pol2, parametros2, metrics2 = fit_3D_coordinates(y_filt_cond2, z_filt_cond2, catenaria, p0)
    x_pol3, y_pol3, parametros3, metrics3 = fit_3D_coordinates(y_filt_cond3, z_filt_cond3, catenaria, p0)
    
    end_time5 = time.time()
    
    logger.debug(f"Sixth time {end_time5-end_time4}")

    ########################## TONI
    
    rmses.append([metrics1[0], metrics2[0], metrics3[0]])
    maxes.append([metrics1[1], metrics2[1], metrics3[1]])
    correlations.append([[metrics1[2], metrics2[2], metrics3[2]], [metrics1[3], metrics2[3], metrics3[3]]])
    
    logger.success(f"Evaluating fit")
    
    resultados_eval = evaluar_ajuste([x_pol1, x_pol2, x_pol3], [y_pol1, y_pol2, y_pol3], rotated_vertices, vano_length, clusters)
    evaluaciones[idv] = resultados_eval
    
    end_time6 = time.time()
    
    logger.debug(f"Seventh time {end_time6-end_time5}")
    
    return evaluaciones,  x_pol1, y_pol1, parametros1, x_pol2, y_pol2, parametros2,  x_pol3, y_pol3, parametros3,
    

