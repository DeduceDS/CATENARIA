import json
from modules_utils import *
from modules_clustering import *
from modules_preprocess import *
from modules_plots import *
from modules_fits import *

def fit_plot_vano_group(data,sublist=[],plot_filter=None,init=0,end=None,save=False,label=''):
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
    """
    
    #filter= "bad_backing", bad_cluster, bad_line_number, bad_line_orientation, bad_fit, good_fit, empty

    if len(sublist)==0:
        sublist=[data[i]['ID_VANO'] for i in range(len(data))]
    end=int(len(data)) if end==None else end
    
    # print(end)
    
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    
    dataf = {'id': [], 'flag': [],'line_number':[]}
    
    rmses = []
    maxes = []
    correlations = []
    
    for i in range(len(data)):

        if all([i>=init,i<=end]):

            idv=data[i]['ID_VANO']
            vano_length=data[i]["LONGITUD_2D"]
            dataf['id'].append(idv)
            
            if idv in sublist:

                print(f"\nProcessing Vano {i}")
                print(f"\nReference {idv}")
                cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
                cond_values=np.vstack(cond_values)
                apoyo_values=np.vstack(apoyo_values)

                # tree = KDTree(cond_values.T)
                # distances, indices = tree.query(apoyo_values.T)
                # min_index = np.argmin(distances)
                # print(indices[min_index])
                # cl_pt=cond_values[:,indices[min_index]]
                # print(cl_pt)

                X_scaled=np.array([])
                labels=np.array([])

                data[i]['CONDUCTORES_CORREGIDOS']={}
                data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)']={}

                if np.array(extremos_values).shape[1]!=4:
                    
                    extremos_values = define_backings(vano_length,apoyo_values)
                    
                    if extremos_values == -1:
                        continue
                
                    dataf['flag'].append('bad_backing')
                    dataf['line_number'].append(0)
                    
                    # if any([plot_filter=='all',plot_filter=='bad_backing']):
                        # plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    # print(i+1)
                    # print(len(dataf['flag']))
                    # print(dataf['flag'][-1])
                    # continue

                # print('bad_backing2')

                    # print('bad_backing')
                    # extremos_values=define_backings(vano_length,apoyo_values)
                    # print(extremos_values)

                    # if extremos_values==-1:
                    #     if any([plot_filter=='all',plot_filter=='bad_backing']):
                    #         plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    #     continue

                # y=((data[i]['APOYOS'][0]['COORDEANDA_Y'] + data[i]['APOYOS'][1]['COORDEANDA_Y']) / 2)
                # x=((data[i]['APOYOS'][0]['COORDENADA_X'] + data[i]['APOYOS'][1]['COORDENADA_X']) / 2)

                mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
                rotated_ymin=min(rotated_conds[1])
                rotated_ymax=max(rotated_conds[1])
                cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
                cropped_conds= clean_outliers_2(rotated_conds)
                cropped_conds =clean_outliers_3(cropped_conds)

                # outliers = pcd_o3d.select_by_index(filtered_points[1], invert=True)

                X_scaled,scaler_x,scaler_y,scaler_z = scale_conductor(cropped_conds)
                
                rotated_ymax=scaler_y.transform(np.array([rotated_ymax]).reshape(-1, 1))[0]
                rotated_ymin=scaler_y.transform(np.array([rotated_ymin]).reshape(-1, 1))[0]
    
                # Find clusters in 10 cloud of points corresponding to 3 positions in y axis

                maxy=max(rotated_extremos[1,1],rotated_extremos[1,2])
                miny=min(rotated_extremos[1,1],rotated_extremos[1,2])
                leny=maxy-miny

                l=[]
                filt=[]
                k=10

                for g in range(0,k):

                    filt0=(cropped_conds[1,:]>(miny+g*(1/k)*leny))&(cropped_conds[1,:]<(miny+(g+1)*(1/k)*leny))
                    l0=cropped_conds[:,filt0].shape[1]
                    l.append(l0)
                    filt.append(filt0)

                c=[]
                greater_var_z_than_x=[]
                ncl=[]

                for g in range(0,k):

                    l0=X_scaled[:,filt[g]].shape[1]
                    fl=pd.Series(X_scaled[2,filt[g]]).var()-pd.Series(X_scaled[0,filt[g]]).var()
                    centroids0,labels0=dbscan_find_clusters_3(X_scaled[:,filt[g]]) if l0>20 else ([],[])
                    greater_var_z_than_x.append(True if fl>=0 else False)
                    c.append(centroids0)
                    ncl.append(len(centroids0))

                md=mode(ncl)
                var_z_x=mode(greater_var_z_than_x)
                num_empty=np.array([l0<20 for l0 in l]).sum()
                completeness=np.array(['incomplete','partially incomplete','full'])
                completeness_conds=np.array([num_empty>5,all([num_empty<=5,num_empty>=2]),num_empty<2])
                finc= completeness[completeness_conds]

                fits=[]
                crossesa= []
                crossesb= []

                dataf['line_number'].append(md)
                print(f'Number of lines: {md}')

                if finc[0]=='incomplete':
                    dataf['flag'].append('empty')
                    if any([plot_filter=='all',plot_filter=='empty']):
                        plot_vano('{} Empty{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)

                elif md!=3:
                    dataf['flag'].append('bad_line_number')
                    if any([plot_filter=='all',plot_filter=='bad_line_number']):
                        plot_vano('{} Bad_Line_Number{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)

                else:
                    # X_scaled=X_scaled[:,np.logical_or.reduce(filt[1:-1])]
                    # centroids,labels=dbscan_find_clusters_4(X_scaled)
                    # labels,centroids=spectral_clustering(X_scaled,n_clusters=3,n_init=100)
                    labels, centroids = kmeans_clustering(X_scaled, n_clusters=3, max_iterations=1000)

                    total_points = X_scaled.shape[1]

                    parameters_vano=[]
                    count_good=0
                    bad_cluster=0
                    bad_fit=0
                    crl=[]
                    
                    if len(np.unique(labels))<3:
                        bad_cluster=1

                    for lab in np.unique(labels):

                        idl=idv+'_'+str(lab)
                        clust = X_scaled[:,labels == lab]
                        # corr=np.corrcoef(clust[:1,:])
                        # crl.append(corr)
                        proportion = clust.shape[1]/total_points
                        
                        if proportion< 0.15:
                            bad_cluster=1

                        else:
                            y_vals = clust[1,:].reshape(-1, 1)
                            z_vals = clust[2,:].reshape(-1, 1)
                            y_mean = np.mean(y_vals)
                            y_range = np.max(y_vals) - np.min(y_vals)
                            meanp=min(rotated_apoyos[1])+(max(rotated_apoyos[1])-min(rotated_apoyos[1]))/2
                            
                            initial_params = [1,0,0] # a, h, k
                            
                            try:
                                    
                                optim_params, _ = curve_fit(catenaria, y_vals.flatten(), z_vals.flatten())#,p0=initial_params,method = 'trf'

                                y_fit=np.linspace(rotated_ymin,rotated_ymax,1000).flatten()

                                z_fit = catenaria(y_fit, *optim_params)

                                # coefficients = np.polyfit(x_vals.flatten(), y_vals.flatten(), 1)
                                # linear_fit = np.poly1d(coefficients)
                                # slope, intercept=coefficients
                                # x_fit = invert_linear_model(y_fit, slope, intercept)
                                # ****model = LinearRegression()
                                # ****model.fit(x_vals, y_vals)
                                # ****slope = model.coef_[0][0]
                                # ****intercept = model.intercept_[0]
                                # ****x_fit = invert_linear_model(y_fit, slope, intercept)

                                x_fit = np.repeat(pd.Series(clust[0,:]).quantile(0.5),1000)
                                fit=np.vstack((x_fit, y_fit,z_fit))
                                
                                # rmse, corr = evaluate_fit(fit, X_scaled)

                                fit=un_scale_conductor(fit,scaler_x,scaler_y,scaler_z)
                                # optim_params, _ = curve_fit(catenaria, fit[1,:].flatten(), fit[2,:].flatten(), p0=initial_params, method = 'trf')

                                apoyo_values_a=rotated_apoyos[:,rotated_apoyos[1,:]<(meanp)]
                                apoyo_values_b=rotated_apoyos[:,rotated_apoyos[1,:]>(meanp)]
                                
                                tree = KDTree(fit.T)
                                distancesa, indicesa = tree.query(apoyo_values_a.T)
                                min_index = np.argmin(distancesa)
                                cl_pta=fit[:,indicesa[min_index]]
                                
                                distancesb, indicesb = tree.query(apoyo_values_b.T)
                                min_index = np.argmin(distancesb)
                                cl_ptb=fit[:,indicesb[min_index]]
            
                                fit=fit[:,(fit[1,:]>cl_pta[1])&(fit[1,:]<cl_ptb[1])]
                                
                                # cl_pta=clpt_to_array(cl_pta)
                                # cl_ptb=clpt_to_array(cl_ptb)

                                y_fit=np.linspace(cl_pta[1],cl_ptb[1],1000).flatten()
                                y_fit=scaler_y.transform(y_fit.reshape(-1,1))
                                

                                # coefficients = np.polyfit(x_vals.flatten(), y_vals.flatten(), 1)
                                # linear_fit = np.poly1d(coefficients)
                                # slope, intercept=coefficients
                                # x_fit = invert_linear_model(y_fit.flatten(), slope, intercept)
                                # ****x_fit = invert_linear_model(y_fit, slope, intercept)

                                z_fit = catenaria(y_fit, *optim_params)
                                x_fit = np.repeat(pd.Series(clust[0,:]).quantile(0.5),1000)
                                fit=np.vstack((x_fit.flatten(), y_fit.flatten(),z_fit.flatten()))
                                
                                fit=un_scale_conductor(fit,scaler_x,scaler_y,scaler_z)
                                
                                mat_neg,cl_pta=un_rotate_points(cl_pta,mat)
                                mat_neg,cl_ptb=un_rotate_points(cl_ptb,mat)
                                mat_neg,fit=un_rotate_points(fit,mat)
                                
                                fits.append(fit)
                                crossesa.append(cl_pta)
                                crossesb.append(cl_ptb)
                                data[i]['CONDUCTORES_CORREGIDOS'][str(lab)]=fit.T.tolist()

                                # data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)'][str(lab)]=optim_params.tolist()
                                
                                # print(data[i]['CONDUCTOR_CORREGIDO'][lab])
                                # print(data[i]['CONDUCTOR_CORREGIDO_PARAMETROS_(a,h,k)'][lab])
                                # print(indices[min_index])
                                # if corr>0.8:
                                count_good=count_good+1
                                # else:
                                #     bad_cluster=1
                                
                            except:
                                bad_fit=1

                    fits = np.hstack(fits)
                    # print(fits)
                    # print(fits.shape)
                    # print(cond_values)
                    crossesa=np.vstack(crossesa).T#(crossesa[0],crossesa[1],crossesa[2])
                    # print(crossesa)
                    crossesb=np.vstack(crossesb).T#(crossesb[0],crossesb[1],crossesb[2])
                    # print(crossesb)

                    # print(f"{bad_cluster =}")
                    # print(f"{bad_fit =}")
                    # print(f"{count_good =}")

                    if bad_cluster==1:
                        if all([md==3,var_z_x]):
                            dataf['flag'].append('bad_line_orientation')
                            if any([plot_filter=='all',plot_filter=='bad_line_orientation']):
                                plot_vano('{} Bad_Orientation{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        else:
                            dataf['flag'].append('bad_cluster')
                            if any([plot_filter=='bad_cluster',plot_filter=='all']):
                                plot_vano('{} Incomplete_cluster{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    elif bad_fit==1:
                        dataf['flag'].append('bad_fit')
                        if any([plot_filter=='bad_fit',plot_filter=='all']):
                            plot_vano('{} Bad_fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    elif count_good==3:
                        dataf['flag'].append('good_fit')
                        if any([plot_filter=='good_fit',plot_filter=='all']):
                            # plot_vano('{} Good_Fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                            plot_fit('{} Good_Fit{}'.format(idv,' '+finc[0]),cond_values, apoyo_values, vert_values,fits,crossesa,crossesb)

                            # plot_fit(f'{idv}',cond_values, apoyo_values, vert_values,fit)
    #             print(i+1)
    #             print(len(dataf['flag']))
    #             print(dataf['flag'][-1])
    # print(len(dataf['line_number']))
    # print(len(dataf['id']))
    # print(len(dataf['flag']))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if save==True:
        with open(timestamp+'_'+label+'_resultado.json', 'w') as file:
            json.dump(data, file)

    datafr=pd.DataFrame(dataf)
    return datafr


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

    if len(sublist)==0:
        sublist=[data[i]['ID_VANO'] for i in range(len(data))]
        
    end=int(len(data)) if end==None else end
    
    parameters=[]
    incomplete_vanos = []
    incomplete_lines=[]
    dataf = {'id': [], 'flag': [],'line_number':[]}
    
    rmses = []
    maxes = []
    correlations = []
    
    for i in range(len(data)):

        if all([i>=init,i<=end]):

            idv=data[i]['ID_VANO']
            vano_length=data[i]["LONGITUD_2D"]
            dataf['id'].append(idv)
            
            if idv in sublist:

                print(f"\nProcessing Vano {i}")
                print(f"\nReference {idv}")
                cond_values, apoyo_values, vert_values, extremos_values = extract_vano_values(data, i)
                cond_values=np.vstack(cond_values)
                apoyo_values=np.vstack(apoyo_values)

                # tree = KDTree(cond_values.T)
                # distances, indices = tree.query(apoyo_values.T)
                # min_index = np.argmin(distances)
                # print(indices[min_index])
                # cl_pt=cond_values[:,indices[min_index]]
                # print(cl_pt)

                X_scaled=np.array([])
                labels=np.array([])

                data[i]['CONDUCTORES_CORREGIDOS']={}
                data[i]['CONDUCTORES_CORREGIDOS_PARAMETROS_(a,h,k)']={}
                data[i]['PUNTUACIONES']={}

                if np.array(extremos_values).shape[1]!=4:
                    
                    dataf['flag'].append('bad_backing')
                    dataf['line_number'].append(0)
                    
                    extremos_values = define_backings(vano_length,apoyo_values)
                    
                    if extremos_values == -1:
                        continue
            
                    # if any([plot_filter=='all',plot_filter=='bad_backing']):
                    #     continue
                        # plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    # print(i+1)
                    # print(len(dataf['flag']))
                    # print(dataf['flag'][-1])

                # print('bad_backing2')

                    # print('bad_backing')
                    # extremos_values=define_backings(vano_length,apoyo_values)
                    # print(extremos_values)

                    # if extremos_values==-1:
                    #     if any([plot_filter=='all',plot_filter=='bad_backing']):
                    #         plot_vano(f'{idv} Bad_Backing',X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                    #     continue

                # y=((data[i]['APOYOS'][0]['COORDEANDA_Y'] + data[i]['APOYOS'][1]['COORDEANDA_Y']) / 2)
                # x=((data[i]['APOYOS'][0]['COORDENADA_X'] + data[i]['APOYOS'][1]['COORDENADA_X']) / 2)

                mat,rotated_conds, rotated_apoyos, rotated_vertices, rotated_extremos = rotate_vano(cond_values, extremos_values, apoyo_values, vert_values)
                rotated_ymin=min(rotated_conds[1])
                rotated_ymax=max(rotated_conds[1])
                cropped_conds = clean_outliers(rotated_conds, rotated_extremos)
                cropped_conds= clean_outliers_2(rotated_conds)
                cropped_conds =clean_outliers_3(cropped_conds)

                # outliers = pcd_o3d.select_by_index(filtered_points[1], invert=True)

                X_scaled,scaler_x,scaler_y,scaler_z = scale_conductor(cropped_conds)
                
                rotated_ymax=scaler_y.transform(np.array([rotated_ymax]).reshape(-1, 1))[0]
                rotated_ymin=scaler_y.transform(np.array([rotated_ymin]).reshape(-1, 1))[0]
    
                # Find clusters in 10 cloud of points corresponding to 3 positions in y axis

                maxy=max(rotated_extremos[1,1],rotated_extremos[1,2])
                miny=min(rotated_extremos[1,1],rotated_extremos[1,2])
                leny=maxy-miny

                l=[]
                filt=[]
                k=10

                for g in range(0,k):

                    filt0=(cropped_conds[1,:]>(miny+g*(1/k)*leny))&(cropped_conds[1,:]<(miny+(g+1)*(1/k)*leny))
                    l0=cropped_conds[:,filt0].shape[1]
                    l.append(l0)
                    filt.append(filt0)

                c=[]
                greater_var_z_than_x=[]
                ncl=[]

                for g in range(0,k):

                    l0=X_scaled[:,filt[g]].shape[1]
                    fl=pd.Series(X_scaled[2,filt[g]]).var()-pd.Series(X_scaled[0,filt[g]]).var()
                    centroids0,labels0=dbscan_find_clusters_3(X_scaled[:,filt[g]]) if l0>20 else ([],[])
                    greater_var_z_than_x.append(True if fl>=0 else False)
                    c.append(centroids0)
                    ncl.append(len(centroids0))

                md=mode(ncl)
                var_z_x=mode(greater_var_z_than_x)
                num_empty=np.array([l0<20 for l0 in l]).sum()
                completeness=np.array(['incomplete','partially incomplete','full'])
                completeness_conds=np.array([num_empty>5,all([num_empty<=5,num_empty>=2]),num_empty<2])
                finc= completeness[completeness_conds]

                fits=[]
                crossesa= []
                crossesb= []
                bad_fit=0
                bad_cluster=0
                good=0

                dataf['line_number'].append(md)
                print(f'Number of lines: {md}')

                if finc[0]=='incomplete':
                    dataf['flag'].append('empty')
                    if any([plot_filter=='all',plot_filter=='empty']):
                        # plot_vano('{} Empty{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        pass

                elif md!=3:
                    dataf['flag'].append('bad_line_number')
                    if any([plot_filter=='all',plot_filter=='bad_line_number']):
                        # plot_vano('{} Bad_Line_Number{}'.format(idv, ' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                        pass

                else:

                    try:
                        
                        x, y, z = filtering_prefit_1(rotated_conds, extremos_values)
                                
                        ########################
                        
                        clusters = clustering_prefit_1(x,y,z)
                        
                        x1, y1, z1 = clusters[0,0], clusters[1,0], clusters[2,0]
                        x2, y2, z2 = clusters[0,1], clusters[1,1], clusters[2,1]
                        x3, y3, z3 = clusters[0,2], clusters[1,2], clusters[2,2]
                        
                        ################
                        
                        x_filt_cond1, y_filt_cond1, z_filt_cond1 = PCA_filtering_prefit_1(x1, y1, z1)
                        x_filt_cond2, y_filt_cond2, z_filt_cond2 = PCA_filtering_prefit_1(x2, y2, z2)
                        x_filt_cond3, y_filt_cond3, z_filt_cond3 = PCA_filtering_prefit_1(x3, y3, z3)
                        
                        #############################
                        
                        # Función de la catenaria
                        from sklearn.preprocessing import StandardScaler
                        from scipy.optimize import curve_fit
                        
                        # def catenaria(x, a, h, k):
                        #     return a*np.cosh((x-h)/a)+k
                        
                        def catenaria(x, a, b, c, d):
                            return a + b*x + c*x**2 + d*x**3
                        
                        # p0 = [1, 0, 0]  # a, h, k
                        
                        p0 = [0, 1, 1, 1]
                        
                        x_pol1, y_pol1, parametros1, metrics1 = fit_3D_coordinates(y_filt_cond1, z_filt_cond1, catenaria, p0)
                        x_pol2, y_pol2, parametros2, metrics2 = fit_3D_coordinates(y_filt_cond2, z_filt_cond2, catenaria, p0)
                        x_pol3, y_pol3, parametros3, metrics3 = fit_3D_coordinates(y_filt_cond3, z_filt_cond3, catenaria, p0)
                        
                        ########################## TONI
                        from scipy.stats import pearsonr, spearmanr
                        
                        rmses.append([metrics1[0], metrics2[0], metrics3[0]])
                        maxes.append([metrics1[1], metrics2[1], metrics3[1]])
                        correlations.append([metrics1[2], metrics2[2], metrics3[2]])
                
                        #########################
                        
                        plt.figure(figsize=(10, 6))
                        # Pintamos los puntos de cada cable
                        plt.scatter(y1, z1, color='coral', s=30)
                        plt.scatter(y2, z2, color='lightblue', s=30)
                        plt.scatter(y3, z3, color='lightgreen', s=30)

                        # Pintamos las polilíneas que hemos generado
                        plt.plot(x_pol1, y_pol1, color='red', label='P1')
                        plt.plot(x_pol2, y_pol2, color='blue', label='P2')
                        plt.plot(x_pol3, y_pol3, color='green', label='P3')

                        plt.legend()
                        plt.title(idv)
                        plt.show()
                
                        x_fit1 = np.repeat(pd.Series(x1.flatten()).quantile(0.5),1000)
                        x_fit2 = np.repeat(pd.Series(x2.flatten()).quantile(0.5),1000)
                        x_fit3 = np.repeat(pd.Series(x3.flatten()).quantile(0.5),1000)
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
                    except:
                        bad_fit=1

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
                            # plot_vano('{} Good_Fit{}'.format(idv,' '+finc[0]),X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values)
                            # print(cond_values[0])
                            # print(cond_values[1])
                            # print(cond_values[2])
                            plot_fit_2('{} Good_Fit{}'.format(idv,' '+finc[0]),cond_values, apoyo_values, vert_values,fits)

                puntuacion=puntuación_por_vanos(data, idv).to_json()
                puntuacion_dict = json.loads(puntuacion)
                for n in puntuacion_dict:
                    puntuacion_dict[n]=puntuacion_dict[n]["0"]
                puntuacion_dict['Continuidad']=finc[0]
                puntuacion_dict['Conductores identificados']=dataf['line_number'][-1]
                puntuacion_dict['Output']=dataf['flag'][-1]
                del puntuacion_dict['Vano']
                data[i]['PUNTUACIONES']=puntuacion_dict
                print(puntuacion_dict)
    return data, rmses, maxes, correlations

