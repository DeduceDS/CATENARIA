from loguru import logger
import json
import time
import sys
from statistics import mode

from electra_package.modules_utils import *
from electra_package.modules_clustering import *
from electra_package.modules_preprocess import *
from electra_package.modules_plots import *
from electra_package.modules_fits import *

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
    
    logger.remove() # remove the default logger
    # Adding new levels: Critical = Text with Purple Background, Title: Light 
    logger.level("CRITICAL", color = "<bold><bg #AF5FD7>")
    try:
        logger.level("TITLE")
    except ValueError:
        logger.level("TITLE", color="<bold><fg 86>", no=21)
    logger.add(sys.stdout, format = "<lvl>{message}</lvl>", colorize=True, backtrace=True, diagnose=True, level="DEBUG")  

    # If the defined sublist is empty then we process all the data
    if len(sublist)==0:
        sublist=[data[i]['ID_VANO'] for i in range(len(data))]
    
    # If no end value is defined, then it will be the length of all the sublist
    end=int(len(sublist)) if end==None else end
    
    # Declare list of fit parameters, complete and incomplete vanos
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    
    # Dataframe with ID, Flag and n_conductors
    dataf = {'id': [], 'flag': [],'line_number':[]}
    
    # Declare lists for metrics: rmse, correlations...
    rmses = []
    maxes = []
    correlations = []
    
    # Define a dictionary to store the results of all fit evaluation to extract Puntuaciones
    evaluaciones = dict()
    
    # Loop over data
    for i in range(len(data)):

        # Check if it's between the bounds
        if all([i>=init,i<=end]):
            
            # Start the timer
            start_time = time.time()
            
            # Extract vano ID, vano length and save id
            idv=data[i]['ID_VANO']
            vano_length=data[i]["LONGITUD_2D"]
            dataf['id'].append(idv)
            
            # Check if vano ID is in sublist
            if idv in sublist:
                
                logger.critical(f"\nProcessing Vano {i}")
                logger.critical(f"\nReference {idv}")
                
                # Extract vano values: LIDAR points, extremos, polilinia....
                idv, vano_length, cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
                cond_values=np.vstack(cond_values)
                apoyo_values=np.vstack(apoyo_values)

                ##########################################
                # tree = KDTree(cond_values.T)
                # distances, indices = tree.query(apoyo_values.T)
                # min_index = np.argmin(distances)
                # print(indices[min_index])
                # cl_pt=cond_values[:,indices[min_index]]
                # print(cl_pt)
                ##########################################
                
                # Define empty arrays for scaled coordinates and labels
                X_scaled=np.array([])
                labels=np.array([])

                # Create new attributes for data element (dictionary)
                data[i]['CONDUCTORES_CORREGIDOS']={}
                data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)']={}
                data[i]['PUNTUACIONES']={}
                
                # Declare fit evaluation results as 0 tuple and save them as default
                resultados_eval = (0, 0, 0, 0, 0, 0)
                evaluaciones[idv] = resultados_eval

                end_time = time.time()
                
                logger.debug(f"First time {end_time-start_time}")
                
                # Check for lack of extreme values in original data
                # Exception to handle = redefine extreme values
                
                # Conductor extremes/apoyos analysis
                # Let's study and categorize this vano in terms of it's apoyos
                
                if np.array(extremos_values).shape[1]!=4:
        
                    # Start the timer
                    start_time1 = time.time()

                    logger.warning(f"Redefining backings")
                    
                    # Redefine and compute new extreme values
                    extremos_values = define_backings(vano_length,apoyo_values)
                    
                    end_time1 = time.time()
                    
                    logger.debug(f"Second time {end_time1-start_time1}")
                    
                    # Check for missing LIDAR apoyo points
                    # Exception to handle = bad data , correction not possible
                    if extremos_values == -1: # any(extremos_values == -1)
                        
                        # Include flag of bad extreme values
                        # Set the line value of this element as 0 ****
                        
                        # idv, vano_length, cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
                        # plot_data(f"{idv}",cond_values, apoyo_values, vert_values, extremos_values)
                        
                        logger.warning("UN APOYO LIDAR")
                        dataf['flag'].append('bad_backing')
                        dataf['line_number'].append(0)
                        continue
                    
                    # Plot filter to plot bad cases?
                    # if any([plot_filter=='all',plot_filter=='bad_backing']):
                        # plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        
                try:
                            
                    start_time2 = time.time()
                    
                    logger.success(f"Rotating vano")
                    # Preform rotation of all data 3D points over z axis to align the conductor diagonal with the Y axis.
                    mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
            
                    ##########################################
                    
                    logger.success(f"Cropping conductor")
                    # Crop conductor LIDAR points and clean outliers before any transformation
                    # cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
                    cropped_conds = clean_outliers_2(rotated_conds) # More crop?
                    # cropped_conds = clean_outliers_3(cropped_conds)

                    # outliers = pcd_o3d.select_by_index(filtered_points[1], invert=True)
                    
                    ##########################################
                    
                    # Scale conductor 3D points and get the scaler models
                    X_scaled,scaler_x,scaler_y,scaler_z = scale_conductor(cropped_conds) # Scale now?
                    
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
                        
                    end_time2 = time.time()
                    
                    logger.debug(f"Third time {end_time2-start_time2}")

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
                
                except Exception as e:
                    logger.warning(e)
                    continue
                    # print(e)

                # print (finc)
                
                # Declare lists for fits, crosses in apoyo a and crosses in apoyo b
                fits=[]
                crossesa= []
                crossesb= []
                
                # Set bad_fit, bad_cluster and good booleans to 0
                bad_fit=0
                bad_cluster=0
                good=0

                # Manage conditions: 
                # 1. If we have less than 50% of conductor (incomplete) save it as empty
                # 2. If we have n_conductors different to 3 or 6 save it as bad line number
                # 3. In other case compute the fit (50% or greater + correct number of lines)
                
                if finc[0]=='incomplete':
                    dataf['flag'].append('empty')
                    
                    # Plot the bad case
                    if any([plot_filter=='all',plot_filter=='empty']):
                        # plot_vano('{} Empty{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        pass

                elif md!=3 and md!= 6:
                    dataf['flag'].append('bad_line_number')
                    
                    # Plot the bad case
                    if any([plot_filter=='all',plot_filter=='bad_line_number']):
                        # plot_vano('{} Bad_Line_Number{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        pass
                    
                ##########################################
                
                else:

                    try:
                    
                        end_time3 = time.time()
                        
                        logger.debug(f"Fourth time {end_time3-end_time2}")
                        
                        logger.success(f"Starting fit")
                        
                        # logger.success(f"Rotating vano")
                        
                        # # Get rotated cond points and extreme values again?
                        # mat, rotated_conds = rotate_points(cond_values, extremos_values)
                        # extremos_values = mat.dot(extremos_values)
                        
                        # filter_start = time.time()
                        
                        # logger.success(f"Filtering conductor")
                        
                        # # Filter conductor values between extreme values again?
                        # x, y, z = filtering_prefit_2(rotated_conds, extremos_values)
                        
                        # filter_end = time.time()
                        
                        # logger.debug(f"Filtering time {filter_end-filter_start}")
                        
                        # # x, y, z = np.array(rotated_conds)[0], np.array(rotated_conds)[1], np.array(rotated_conds)[2]
                        
                        # ########################
                        
                        x,y,z = X_scaled[0,:], X_scaled[1,:], X_scaled[2,:]
                        
                        logger.debug(f"Prefit data shape {X_scaled.shape}")
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
                        
                        # logger.success(f"PCA filtering")
                        
                        # pca_start = time.time()
                        
                        # # PCA filtering over the conductor values
                        # x_filt_cond1, y_filt_cond1, z_filt_cond1 = PCA_filtering_prefit_2(x1, y1, z1)
                        # x_filt_cond2, y_filt_cond2, z_filt_cond2 = PCA_filtering_prefit_2(x2, y2, z2)
                        # x_filt_cond3, y_filt_cond3, z_filt_cond3 = PCA_filtering_prefit_2(x3, y3, z3)
                        
                        x_filt_cond1, y_filt_cond1, z_filt_cond1 = x1, y1, z1
                        x_filt_cond2, y_filt_cond2, z_filt_cond2 = x2, y2, z2
                        x_filt_cond3, y_filt_cond3, z_filt_cond3 = x3, y3, z3
                        
                        # pca_end = time.time()
                        
                        # logger.debug(f"PCA time {pca_end-pca_start}")
                        
                        #############################
                        
                        end_time4 = time.time()
                        
                        logger.debug(f"Fifth time {end_time4-end_time3}")
                        
                        # Función de la catenaria
                        # Fit functions : catenary, cubic pol .... 
                        
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
                
                        #########################
                        
                        # plt.figure(figsize=(10, 6))
                        # # Pintamos los puntos de cada cable
                        # plt.scatter(y1, z1, color='coral', s=30)
                        # plt.scatter(y2, z2, color='lightblue', s=30)
                        # plt.scatter(y3, z3, color='lightgreen', s=30)

                        # # Pintamos las polilíneas que hemos generado
                        # plt.plot(x_pol1, y_pol1, color='red', label='P1')
                        # plt.plot(x_pol2, y_pol2, color='blue', label='P2')
                        # plt.plot(x_pol3, y_pol3, color='green', label='P3')

                        # plt.legend()
                        # plt.title(idv)
                        # plt.show()

                        logger.success(f"Reconstructing X axis")
                        
                        x_fit1 = np.repeat(pd.Series(x1.flatten()).quantile(0.5),200)
                        x_fit2 = np.repeat(pd.Series(x2.flatten()).quantile(0.5),200)
                        x_fit3 = np.repeat(pd.Series(x3.flatten()).quantile(0.5),200)
                        
                        # ajuste de x vs y - sacar slope + interc
                        # para cada conductor
                        # generar linespace 200 con x(y)
                        
                        
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
                    except Exception as e:
                        
                        logger.warning(e)
                        bad_fit=1
                        continue
                        # raise ValueError(e)
                        
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

                logger.success(f"Setting vano score")
                
                puntuacion=puntuación_por_vanos_sin_ajuste(data, idv, evaluaciones).to_json()
                puntuacion_dict = json.loads(puntuacion)
                
                logger.success(f"Vano score {puntuacion}")
                
                for n in puntuacion_dict:
                    puntuacion_dict[n]=puntuacion_dict[n]["0"]
                puntuacion_dict['Continuidad']=finc[0]
                puntuacion_dict['Conductores identificados']=dataf['line_number'][-1]
                puntuacion_dict['Output']=dataf['flag'][-1]
                del puntuacion_dict['Vano']
                data[i]['PUNTUACIONES']=puntuacion_dict
                
                del data[i]["LIDAR"]
                
                # end_time7 = time.time()
                # logger.debug(f"Eigth time {end_time7-end_time6}")
                
    return data, rmses, maxes, correlations

