
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#### FUNCTIONS TO PLOT DATA AND FITS ####

def add_plot(fig, data, color, size, name, mode):
    """
    Add a 3D scatter plot to a given figure.

    Parameters:
    fig (plotly.graph_objs.Figure): The figure to add the plot to.
    data (list of arrays): A list containing the x, y, and z coordinates.
    color (str or array-like): The color of the markers.
    size (int): The size of the markers.
    name (str): The name of the trace.
    mode (str): The mode of the scatter plot (e.g., 'markers', 'lines', 'lines+markers').
    """

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

def plot_fit(title,cond_values, apoyo_values, vert_values,fit,crossesa,crossesb):
    """
    Create and display a 3D scatter plot with conductors, supports, vertices, fit and endpoints.

    Parameters:
    title (str): The title of the plot.
    cond_values (list of arrays): The x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The x, y, and z coordinates for vertices.
    fit (list of arrays): The x, y, and z coordinates for the selected fit.
    crossesa (list of arrays): The x, y, and z coordinates for endpoints in apoyo 1.
    crossesb (list of arrays): The x, y, and z coordinates for endpoints in apoyo 2.
    """
    # Crea el gráfico para los conductores
    fig = go.Figure(data=[go.Scatter3d(
        x=fit[0],
        y=fit[1],
        z=fit[2],
        mode='markers',
        marker=dict(
            size=2.5,
            color='green',  # Color de los conductores
        ),
        name='Conductores'  # Nombre de la traza de los conductores
    )])

    # Agrega el gráfico para el fit
    add_plot(fig, cond_values, "blue", 2.5, "Apoyos", "markers")
    # Agrega el gráfico para los apoyos
    add_plot(fig, apoyo_values, "orange", 2.5, "Apoyos", "markers")
    add_plot(fig, crossesa, "purple", 5, "Apoyos", "markers")
    add_plot(fig, crossesb, "purple", 5, "Apoyos", "markers")

    for vert in vert_values:

        # Agrega el gráfico para los vertices
        add_plot(fig, vert , "red", 5, "Vertices", "lines")

    # Agrega títulos a los ejes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=0, b=0, l=0, t=0)  # Reduce el margen alrededor del gráfico
    )

    # Muestra el gráfico
    fig.show()

def plot_fit_2(title,cond_values, apoyo_values, vert_values,fit):
    """
    Create and display a 3D scatter plot with conductors, supports, vertices, and fit.

    Parameters:
    title (str): The title of the plot.
    cond_values (list of arrays): The x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The x, y, and z coordinates for vertices.
    fit (list of arrays): The x, y, and z coordinates for the selected fit.
    """

    # Crea el gráfico para los conductores
    fig = go.Figure(data=[go.Scatter3d(
        x=fit[0],
        y=fit[1],
        z=fit[2],
        mode='markers',
        marker=dict(
            size=2.5,
            color='green',  # Color de los conductores
        ),
        name='Conductores'  # Nombre de la traza de los conductores
    )])

    # Agrega el gráfico para el fit
    add_plot(fig, cond_values, "blue", 2.5, "Apoyos", "markers")
    # Agrega el gráfico para los apoyos
    add_plot(fig, apoyo_values, "orange", 2.5, "Apoyos", "markers")

    for vert in vert_values:

        # Agrega el gráfico para los vertices
        add_plot(fig, vert , "red", 5, "Vertices", "lines")

    # Agrega títulos a los ejes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=0, b=0, l=0, t=0)  # Reduce el margen alrededor del gráfico
    )

    # Muestra el gráfico
    fig.show()

def plot_data(title,cond_values=[], apoyo_values=[], vert_values=[], extremos_values=[], color="red"):
    """
    Create and display a 3D scatter plot with conductors, supports, vertices, and endpoints.

    Parameters:
    title (str): The title of the plot.
    cond_values (list of arrays): The x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The x, y, and z coordinates for vertices.
    extremos_values (list of arrays): The x, y, and z coordinates for endpoints.
    """
    
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

    if len(apoyo_values) != 0:

        # Agrega el gráfico para los apoyos
        add_plot(fig, apoyo_values, "orange", 2.5, "Apoyos", "markers")
    
    if len(extremos_values) != 0:

        # Agrega el gráfico para los extremos
        add_plot(fig, extremos_values, "black", 5, "Extremos", "markers")

    if len(vert_values) != 0:
            
        for vert in vert_values:
        
            # Agrega el gráfico para los vertices
            add_plot(fig, vert , color, 4, "Vertices", "lines")
            add_plot(fig, vert , color, 3, "Vertices", "markers")

    # Agrega títulos a los ejes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=0, b=0, l=0, t=0)  # Reduce el margen alrededor del gráfico
    )

    # Muestra el gráfico
    fig.show()
    
def plot_vano(title,X_scaled,labels,cond_values, apoyo_values, vert_values, extremos_values):
    """
    This function generates scatter plots to visualize the clustering results on the x-y and y-z planes.
    It also generates a 3D plot of the conductor values, support values, vertices, and endpoints.

    Parameters:
    title (str): The title of the plots.
    X_scaled (numpy.ndarray): The scaled x, y, and z coordinates of the conductor points.
    labels (numpy.ndarray): The cluster assignment for each point.
    cond_values (list of arrays): The original x, y, and z coordinates for conductors.
    apoyo_values (list of arrays): The original x, y, and z coordinates for supports.
    vert_values (list of lists of arrays): The original x, y, and z coordinates for vertices.
    extremos_values (list of arrays): The original x, y, and z coordinates for endpoints.
    """
    if len(labels)!=0:
        plt.scatter(X_scaled.T[:, 0], X_scaled.T[:, 1], c=labels, cmap='viridis', label = labels)
        plt.title('Clustering con kmeans')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.tight_layout()
        plt.title(title)
        plt.show()

        plt.scatter( X_scaled.T[:, 1], X_scaled.T[:, 2],c=labels, cmap='viridis', label = labels)
        plt.title('Clustering con kmeans')
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.tight_layout()
        plt.title(title)
        plt.show()

    plot_data(title,cond_values, apoyo_values, vert_values, extremos_values)
    
def plot_net(data,labels,k=10):
    """
    Plot the network of spans based on clustering labels.

    This function extracts the middle points of the spans from the provided data, and then generates 
    a scatter plot of these points colored by their cluster labels. The plot includes labels and a legend 
    to indicate the different clusters.

    Parameters:
    data (list of dicts): The data containing information about different spans.
    labels (numpy.ndarray): The cluster labels assigned to each span.
    k (int): An optional parameter for any future enhancements or clustering requirements (default is 10).
    """
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
    
def plot_clusters(points, labels, centroids, coord):
    """
    Plot the clusters and centroids.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    labels (numpy.ndarray): The cluster labels for each point.
    centroids (numpy.ndarray): The centroids.
    coord (int): The coordinate axis used for clustering.
    """
    if coord == 0:
        coord1, coord2 = 1, 2
    elif coord == 1:
        coord1, coord2 = 0, 2
    else:
        coord1, coord2 = 0, 1

    plt.figure(figsize=(12,8))
    
    plt.subplot(1,2,1)
    plt.scatter(points[coord, :], points[coord1, :], c=labels, cmap='viridis', s=1)
    # plt.vlines(centroids, ymin=np.min(points[coord1, :]), ymax=np.max(points[coord1, :]), color='red', label='Centroids')
    
    plt.subplot(1,2,2)
    plt.scatter(points[coord, :], points[coord2, :], c=labels, cmap='viridis', s=1)
    # plt.vlines(centroids, ymin=np.min(points[coord2, :]), ymax=np.max(points[coord2, :]), color='red', label='Centroids')
    
    plt.title(f'Clusters along {["X", "Y", "Z"][coord]}-axis vs {["X", "Y", "Z"][coord1]}-axis')
    plt.xlabel(f'{["X", "Y", "Z"][coord]} Coordinate')
    plt.ylabel(f'{["X", "Y", "Z"][coord1]} Coordinate')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_2d(rotated_conds, rotated_extremos, rotated_apoyos):

    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.scatter(rotated_conds[1], rotated_conds[2], s = 1, label = "Original Conductors")
    plt.scatter(rotated_apoyos[1], rotated_apoyos[2], s = 1, label = "Cropped Conductors")
    plt.scatter(rotated_extremos[1], rotated_extremos[2], color='black')
    plt.title("YZ")

    plt.subplot(122)
    plt.scatter(rotated_conds[0], rotated_conds[1], s = 1, label = "Original Conductors")
    plt.scatter(rotated_apoyos[0], rotated_apoyos[1], s = 1, label = "Cropped Conductors")
    plt.scatter(rotated_extremos[0], rotated_extremos[1], color='black')
    plt.title("XY")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# def plot_full_net(data,labels):
#     """
#     Plot the full network of spans based on clustering labels.

#     This function processes each cluster in the provided data, extracts relevant values for each span,
#     fits catenary parameters, and generates plots for each cluster. It also provides summary statistics
#     about the spans, including counts of spans with single supports, incomplete spans, and spans analyzed.
#     Finally, it plots the distribution of parameters for all lines in the network.

#     Parameters:
#     data (list of dicts): The data containing information about different spans.
#     labels (numpy.ndarray): The cluster labels assigned to each span.
#     """

#     ids_single_backing,X=data_middlepoints(data)

#     fulldata_plot=[]
#     for lbl in np.unique(labels):

#         idval_subg=X.loc[labels==lbl,'ids'].to_list()

#         parameters,incomplete_vanos=fit_vano_group(data,sublist=idval_subg)

#         dfd=pretreatment_linegroup(parameters)

#         print(f'\nVanos con un sólo apoyo: {len(ids_single_backing)}')
#         print(f'Vanos incompletos: {len(incomplete_vanos)}')
#         print(f'Incompletos con apoyos: {len([el for el in incomplete_vanos if el not in ids_single_backing])}')
#         print(f'Sin apoyos y completos: {len([el for el in ids_single_backing if el not in incomplete_vanos])}')
#         print(f'Vanos analizados:{dfd.shape[0]}')
#         print(f'Vanos perdidos:{len(parameters)-dfd.shape[0]}\n')

#         plot_linegroup_parameters(dfd,str(lbl))
#         total=pd.concat([dfd['A1'],dfd['B1'],dfd['C1']],axis=0)
#         fulldata_plot.append(total)

#     mins=[]
#     maxs=[]
#     for ils,lbl in enumerate(np.unique(labels)):
#         plt.hist(fulldata_plot[ils],label=lbl,alpha=0.5,density=True)
#         mins.append(fulldata_plot[ils].min())
#         maxs.append(fulldata_plot[ils].max())

#     plt.xlim(min(mins)-0.2,max(maxs)+0.2)
#     plt.legend()
#     plt.title('All Lines Distribution')
#     plt.show()

def plot_linegroup_parameters(dfd,lbl):
    """
    Plot the distribution of parameters for different line groups within a cluster.

    This function takes a DataFrame containing the parameters for different line groups (A1, B1, C1)
    within a specific cluster, and generates histograms to visualize their distributions. It also plots
    the overall distribution of all parameters combined. Mean values are highlighted with a red dashed line.

    Parameters:
    dfd (pd.DataFrame): A DataFrame containing the parameters for different line groups.
    lbl (str): The label of the cluster being plotted.
    """
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