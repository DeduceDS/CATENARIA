from loguru import logger
from statistics import mode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from collections import Counter

from scipy.stats import linregress

from electra_package.modules_clustering import (
    kmeans_clustering,
    dbscan_find_clusters_3,
    extract_n_clusters,
)
from electra_package.modules_preprocess import (
    clean_outliers,
    clean_outliers_2,
    scale_conductor,
)
from electra_package.modules_fits import fit_3D_coordinates_2
from electra_package.puntuacionparavano import puntuacion_aposteriori
from electra_package.modules_plots import plot_clusters, plot_data


def analyze_polilinia_values(vert_values, cond_values, vano_length):

    initial_conductor_number = len(vert_values)
    logger.critical(f"len initial vert values: {initial_conductor_number}")
    empty_poli = 0
    not_empty = []

    for poli in vert_values:

        max_pos = poli[:, np.where(poli[1] == np.max(poli[1]))].flatten()
        min_pos = poli[:, np.where(poli[1] == np.min(poli[1]))].flatten()

        if np.array(poli).shape[1] <= 3:
            logger.warning(f"Poli with less than 3 points")
            empty_poli += 1

        elif (
            100 * abs(np.linalg.norm(max_pos - min_pos) - vano_length) / vano_length
            > 85.0
        ):

            logger.warning(
                f"Empy polilinia %: {abs(np.linalg.norm(max_pos-min_pos)-vano_length)/vano_length}"
            )
            empty_poli += 1

        else:
            not_empty.append(poli)

    if not_empty == []:
        return initial_conductor_number, 0, empty_poli

    X_scaled, scaler_x, scaler_y, scaler_z = scale_conductor(cond_values)
    max_vars = []

    for poli in not_empty:

        x_vals_scaled = scaler_x.transform(
            np.array(poli)[0, :].reshape(-1, 1)
        ).flatten()  # Flatten per curve_fit
        y_vals_scaled = scaler_y.transform(
            np.array(poli)[1, :].reshape(-1, 1)
        ).flatten()
        z_vals_scaled = scaler_z.transform(
            np.array(poli)[2, :].reshape(-1, 1)
        ).flatten()

        variances = [
            np.std(x_vals_scaled),
            np.std(y_vals_scaled),
            np.std(z_vals_scaled),
        ]
        # logger.critical(f"variances: {variances}")
        max_var = np.argmax(variances)

        max_vars.append(max_var)
        # print([x_vals_scaled, y_vals_scaled, z_vals_scaled])

    logger.critical(f"max variances: {max_vars}")

    main_coord = mode(max_vars)

    vert_values = [
        vert for i, vert in enumerate(not_empty) if max_vars[i] == main_coord
    ]

    logger.critical(
        f"number of consistent vert values: {len(vert_values)}, max variance {main_coord}"
    )
    expected_conductor_number = len(vert_values)

    return initial_conductor_number, expected_conductor_number, empty_poli


def analyze_backings(vano_length, apoyo_values, extremos_values, plot_signal):

    logger.info(f"Analyzing backings")

    logger.trace(
        f"Variance distribution in backings {np.std(apoyo_values[0,:]), np.std(apoyo_values[1,:])}"
    )

    coord = np.argmax([np.std(apoyo_values[0, :]), np.std(apoyo_values[1, :])], axis=0)

    logger.debug(f"Max variance coordinate for backings {coord}")

    # Redefine and compute new extreme values
    extremos_values = define_backings(vano_length, apoyo_values, coord, plot_signal)

    # Check for missing LIDAR apoyo points
    # Exception to handle = bad data , correction not possible
    if extremos_values == -1:  # any(extremos_values == -1)

        # Include flag of bad extreme values
        # Set the line value of this element as 0 ****
        logger.warning("NO HAY 2 APOYOS LIDAR")
        
        if plot_signal:
            
            plot_data("",[], apoyo_values, [], extremos_values)

        return -1

    return list(extremos_values)


def define_backings(vano_length, apoyo_values, coord, plot_signal):
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

    logger.warning(f"Redefining backings")

    points = np.array(apoyo_values)

    labels, centroids = kmeans_clustering(points, 2, 100, coord)

    apoyos = []
    extremos = []

    for lab in np.unique(labels):

        apoyo = np.array(apoyo_values)[:, labels == lab]

        mean_x = np.mean(apoyo[0, :])
        mean_y = np.mean(apoyo[1, :])
        max_z = np.max(apoyo[2, :])
        min_z = np.min(apoyo[2, :])

        c_mass1 = np.array([mean_x, mean_y, min_z])
        c_mass2 = np.array([mean_x, mean_y, max_z])

        extremos.append(c_mass1)
        extremos.append(c_mass2)

        apoyos.append(apoyo)

    dist = np.linalg.norm(np.array(extremos)[0, :] - np.array(extremos)[2, :])
    extremos = np.array(extremos)

    if 100 * abs(dist - vano_length) / vano_length > 15.0:

        logger.debug(
            f"Proportional absolut error of distance = {100*abs(dist - vano_length)/vano_length}"
        )
        logger.debug(f"Vano length, distance {vano_length, dist}")
        logger.debug(f"Invertir coordenadas")

        if coord == 0:
            coord = 1
        else:
            coord = 0

        labels, centroids = kmeans_clustering(points, 2, 100, coord)

        apoyos = []
        extremos = []

        for lab in np.unique(labels):

            apoyo = np.array(apoyo_values)[:, labels == lab]

            mean_x = np.mean(apoyo[0, :])
            mean_y = np.mean(apoyo[1, :])
            max_z = np.max(apoyo[2, :])
            min_z = np.min(apoyo[2, :])

            c_mass1 = np.array([mean_x, mean_y, min_z])
            c_mass2 = np.array([mean_x, mean_y, max_z])

            extremos.append(c_mass1)
            extremos.append(c_mass2)

            apoyos.append(apoyo)

        dist = np.linalg.norm(np.array(extremos)[0, :] - np.array(extremos)[2, :])
        extremos = np.array(extremos)

        if 100 * abs(dist - vano_length) / vano_length > 15.0:

            logger.debug(
                f"Proportional absolut error of distance = {100*abs(dist - vano_length)/vano_length}"
            )
            logger.debug(f"Vano length, distance {vano_length, dist}")
            logger.warning("NO HAY DOS APOYOS")

            if coord == 0:
                coord1, coord2 = 1, 2
            else:
                coord1, coord2 = 0, 2
                
            if plot_signal:

                plt.figure(figsize=(12,8))
                plt.subplot(121)
                plt.scatter(points[coord], points[coord1], c=labels, cmap='viridis', s=1)
                plt.vlines(centroids, ymin=np.min(points[coord1]), ymax=np.max(points[coord1]), color='red')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.subplot(122)
                plt.scatter(points[coord], points[coord2], c=labels, cmap='viridis', s=1)
                plt.vlines(centroids, ymin=np.min(points[coord2]), ymax=np.max(points[coord2]), color='red')
                plt.xlabel('X Coordinate')
                plt.ylabel('Z Coordinate')
                plt.title(f'Custom 1D K-means Clustering for coord {coord}')

                plt.show()

            return -1

    z_vals = np.stack(
        [
            np.array(extremos)[2, 2],
            np.array(extremos)[3, 2],
            np.array(extremos)[0, 2],
            np.array(extremos)[1, 2],
        ]
    )
    y_vals = np.stack(
        [
            np.array(extremos)[2, 1],
            np.array(extremos)[3, 1],
            np.array(extremos)[0, 1],
            np.array(extremos)[1, 1],
        ]
    )
    x_vals = np.stack(
        [
            np.array(extremos)[2, 0],
            np.array(extremos)[3, 0],
            np.array(extremos)[0, 0],
            np.array(extremos)[1, 0],
        ]
    )

    extremos_values = [x_vals, y_vals, z_vals]

    return list(extremos_values)


def analyze_conductor_configuration(X_scaled):

    logger.info(f"Analyzing conductor configuration (1)")

    a1, b1 = np.histogram(X_scaled[0, :] / np.max(X_scaled[0, :]), bins=5)
    a2, b2 = np.histogram(X_scaled[1, :] / np.max(X_scaled[1, :]), bins=5)
    a3, b3 = np.histogram(X_scaled[2, :] / np.max(X_scaled[2, :]), bins=5)

    normalized_a1 = a1 / np.max(a1)
    normalized_a2 = a2 / np.max(a2)
    normalized_a3 = a3 / np.max(a3)

    variances = [np.std(normalized_a1), np.std(normalized_a2), np.std(normalized_a3)]
    # variances = [np.std(X_scaled[0,:]), np.std(X_scaled[1,:]), np.std(X_scaled[2,:])]

    logger.critical(f"variances {variances}")

    max_var = np.argmax(variances)
    min_var = np.argmin(variances)

    cond1 = np.abs(np.max(normalized_a1) - np.min(normalized_a1)) > 0.8 * np.abs(
        np.max(normalized_a2) - np.min(normalized_a2)
    )
    cond2 = np.abs(np.max(normalized_a3) - np.min(normalized_a3)) > 0.8 * np.abs(
        np.max(normalized_a2) - np.min(normalized_a2)
    )

    cond3 = max_var == 0
    cond4 = max_var == 2

    # cond5 = (min_var/max_var < 0.5)

    logger.debug(f"Conductor variances {variances}")
    logger.debug(
        f"Histogram and variance conditions (x,z) {cond1,cond3}, {cond2,cond4}"
    )
    logger.debug(
        f"Horizontal condition {np.abs(np.max(normalized_a1)-np.min(normalized_a1))} , {np.abs(np.max(normalized_a2)-np.min(normalized_a2))}, {np.abs(np.max(normalized_a3)-np.min(normalized_a3))}"
    )

    logger.success(f"Max var coordinate for conductors {max_var}")

    if cond1 and cond3:

        logger.success("Distribución horizontal")
        return 0, max_var

    else:
        logger.warning("Other geometry")
        logger.warning(f"variances {variances}")

        # plt.figure(figsize=(12,8))
        # plt.hist(X_scaled[0,:], bins = 5, label = "x", alpha = 0.5)
        # plt.hist(X_scaled[1,:], bins = 5, label = "y", alpha = 0.5)
        # plt.hist(X_scaled[2,:], bins = 5, label = "z", alpha = 0.5)
        # plt.title("Coordinate distribution")
        # plt.legend()
        # plt.show()

        if cond2 and cond4:

            logger.success("Distribución vertical")
            return 1, max_var

        else:
            logger.warning(
                f"Unrecognized geometry: variances {np.std(normalized_a1), np.std(normalized_a2), np.std(normalized_a3)}"
            )
            return -1, -1
        
def filter_and_relabel_three_largest_clusters(labels, centroids, X_scaled):
    # Step 1: Identify the three largest clusters by size, excluding noise
    label_counts = Counter(labels[labels != -1])  # Exclude noise points (-1)
    three_largest_clusters = [cluster for cluster, _ in label_counts.most_common(3)]
    
    # Step 2: Create a mapping for the three largest clusters to new labels [0, 1, 2]
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(three_largest_clusters)}

    # Step 3: Relabel the data points to have only labels [0, 1, 2]
    filtered_labels = np.array([cluster_mapping[label] for label in labels if label in three_largest_clusters])

    # Step 4: Filter the centroids for the three largest clusters
    filtered_centroids = np.array([centroids[cluster] for cluster in three_largest_clusters])
    
    # Step 5: Filter X_scaled to keep only points belonging to the three largest clusters
    filtered_X_scaled = X_scaled[:, np.isin(labels, three_largest_clusters)]

    return filtered_X_scaled, filtered_labels, filtered_centroids


def cluster_and_evaluate(X_scaled, n_conds, coord):

    good_clust = False
    clusters = []

    for i in range(2):
        
        if coord == 0:

            if i == 0:
                labels, centroids = kmeans_clustering(
                    X_scaled, n_conds, 500, coord, mode="Normal"
                )
            else:
                labels, centroids = kmeans_clustering(
                    X_scaled, n_conds, 500, coord, mode="Random"
                )
                
        else:
            
            # Assuming X_scaled is your input data that is already defined and preprocessed
            centroids, labels = dbscan_find_clusters_3(X_scaled)  # Adjust parameters if necessary

            # Example usage with your existing centroids, labels, and X_scaled
            X_scaled, labels, centroids = filter_and_relabel_three_largest_clusters(labels, centroids, X_scaled)

        max_size = 0

        for lab in labels:
            clust = X_scaled[:, labels == lab]

            if 100 * (clust.shape[1] / len(labels)) > max_size:
                max_size = 100 * (clust.shape[1] / len(labels))

        if abs(max_size - (100 / n_conds)) > 20:  #### CHECK ###

            # logger.trace(f"bad cluster proportion, {100*(clust.shape[1]/len(labels)), (100/n_conds)}")

            logger.debug(f"{clust.shape[1],len(labels)}")
            logger.debug(f"break, bad clusters, {max_size, (100/n_conds)}")
            good_clust = False
            # plot_clusters(X_scaled, labels, centroids, coord)

            continue

        min_cent = np.min(centroids)
        max_cent = np.max(centroids)
        mid_cent = np.mean(centroids)

        # mid_cent = centroids[(centroids != min_cent)*(centroids != max_cent)]

        cent_dist1 = abs(mid_cent - max_cent)
        cent_dist2 = abs(mid_cent - min_cent)

        diff = abs(cent_dist1 - cent_dist2)

        logger.trace(f"Good centroids condition, {diff/abs(max_cent-min_cent)}")

        if diff / abs(max_cent - min_cent) < 1.0:

            logger.success("GOOD CENTROIDS")

            min_max = []

            for lab in np.unique(labels):

                clusters.append(X_scaled[:, labels == lab])
                min_max.append(
                    (
                        np.min(X_scaled[coord, labels == lab]),
                        np.max(X_scaled[coord, labels == lab]),
                    )
                )

            logger.debug(f"{len(clusters)}")

            if len(clusters) != n_conds:
                # raise ValueError(f"Bad clustering")
                logger.error(f"Bad clustering len(clusters) != n_conds")
                # plot_clusters(X_scaled, labels, centroids, coord)
                good_clust = False
                return good_clust, clusters

            # Check for overlapping clusters
            overlapping_clusters = []
            for k, (min_x_k, max_x_k) in enumerate(min_max):
                for j, (min_x_j, max_x_j) in enumerate(min_max):

                    if k != j:

                        dist1 = abs(min_x_k - max_x_j)
                        dist2 = abs(min_x_j - max_x_k)

                        # logger.trace(f"Overlapping centroids condition, {dist1/abs(max_cent-min_cent), dist2/abs(max_cent-min_cent)}")
                        if (
                            dist1 / abs(max_cent - min_cent) < 0
                            or dist2 / abs(max_cent - min_cent) < 0
                        ):
                            overlapping_clusters.append(k)

            if len(overlapping_clusters) == 0:
                logger.success(f"GOOD CLUSTERS: found {n_conds}")
                plot_clusters(X_scaled, labels, centroids, coord)
                good_clust = True
                return good_clust, clusters

            else:
                logger.warning("OVERLAPPING CLUSTERS")
                plot_clusters(X_scaled, labels, centroids, coord)
                # good_clust = False
                return good_clust, clusters

        else:
            logger.debug(f"TRY {i}")
            continue

    if not good_clust:
        logger.warning(f"BAD CLUSTERS AFTER {i} TRIALS")
        # plot_clusters(X_scaled, labels, centroids, coord)

    return good_clust, clusters


def extract_conductor_config(X_scaled, rotated_extremos, cropped_conds):

    # Conductor geometry/configuration analysis
    # Let's study and categorize this vano in terms of it's conductors

    logger.info(f"Analyzing conductor configuration (2)")

    # Find clusters in 10 cloud of points corresponding to 3 positions in y axis
    # Define boundaries of the conductor to extract min, max values and length
    maxy = max(rotated_extremos[1, 1], rotated_extremos[1, 2])
    miny = min(rotated_extremos[1, 1], rotated_extremos[1, 2])
    leny = maxy - miny  # Equal to 2D length?

    thresh_value = X_scaled.shape[1] / 20

    logger.critical(f"Using thresh value: {thresh_value}")
    # Filter and extract 10 10% length segments

    l = []  # Fragment values
    filt = []  # Index for fragments
    k = 10

    for g in range(0, k):

        filt0 = (cropped_conds[1, :] > (miny + g * (1 / k) * leny)) & (
            cropped_conds[1, :] < (miny + (g + 1) * (1 / k) * leny)
        )
        l0 = cropped_conds[:, filt0].shape[1]
        l.append(l0)
        filt.append(filt0)

    # Calculate: n points of each segment, the variance difference between x and z coordinates
    # Find clusters for segments that have more than 20 points
    # If the variance in z is greater than x then append this result

    c = []  # Centroid values resulting from clustering
    max_vars = []
    ncl = []  # Number of centroids from clustering

    for g in range(0, k):

        l0 = X_scaled[:, filt[g]].shape[1]
        # fl=pd.Series(X_scaled[2,filt[g]]).var()-pd.Series(X_scaled[0,filt[g]]).var()
        centroids0, labels0 = (
            dbscan_find_clusters_3(X_scaled[:, filt[g]])
            if l0 > thresh_value
            else ([], [])
        )
        # config, max_var = analyze_conductor_configuration(X_scaled[:,filt[g]])
        max_var = np.argmax(
            [
                np.std(X_scaled[0, filt[g]]),
                np.std(X_scaled[1, filt[g]]),
                np.std(X_scaled[2, filt[g]]),
            ]
        )
        c.append(centroids0)
        ncl.append(len(centroids0))
        max_vars.append(max_var)

    logger.debug(f"{ncl}, {centroids0}, {max_vars}")

    ##########################################
    # Obtain the mode of n_clusters along the list of size 10 == number of conductors
    mvar = mode(max_vars)

    logger.critical(f"Variances: {max_vars}")
    logger.critical(f"Max variance from mode: {mvar}")

    ##########################################
    # Obtain the mode of n_clusters along the list of size 10 == number of conductors
    md = mode([v for v in ncl if v != 0])

    logger.success(f"Number of lines from mode: {md}")

    # Do the same for the x vs z variance relation
    # var_z_x=mode(greater_var_z_than_x)

    # Compute the number of empty fragments
    num_empty = np.array([l0 < thresh_value for l0 in l]).sum()

    logger.debug(f"{[l0<thresh_value for l0 in l]}")

    # Define 3 completeness categories
    completeness = np.array(["incomplete", "partially incomplete", "full"])

    # Define the index with the completeness conditions
    completeness_conds = np.array(
        [num_empty > 8, all([num_empty <= 8, num_empty >= 2]), num_empty < 2]
    )
    # Extract final value with index
    finc = completeness[completeness_conds]

    logger.success(f"Completeness value: {finc}")

    return num_empty, finc[0], md, mvar


def fit_and_evaluate_conds(clusters, scaled_extremos):

    logger.info(f"Fitting with catenaria function")

    def catenaria(x, a, h, k):
        return a * np.cosh((x - h) / a) + k

    p0 = [1, 0, 0]  # a, h, k

    # def catenaria(x, a, b, c, d):
    #         return a + b*x + c*x**2 + d*x**3
    # p0 = [0, 1, 1, 1]

    pols = []
    params = []
    metrics_vano = []

    logger.info(f"Interquartile filtering prefit")
    # plt.figure(figsize=(12,8))

    for l, clus in enumerate(clusters):

        x_pol, y_pol, z_pol = [], [], []
        slope, intercept = 0, 0

        clus = clean_outliers_2(np.array(clus))

        y_pol, z_pol, parametros, metrics_cond = fit_3D_coordinates_2(
            clus[1, :], clus[2, :], scaled_extremos, catenaria, p0
        )
        slope, intercept, r_value1, p_value, std_err = linregress(
            clus[1, :], clus[0, :]
        )

        x_pol = slope * y_pol + intercept

        logger.debug(f"Intercept, slope: {intercept}, {slope}")

        pols.append([x_pol, y_pol, z_pol])
        params.append(parametros)
        metrics_vano.append(metrics_cond)

        # plt.subplot(2,3,l+1)
        # plt.scatter(clus[1,:], clus[2,:])
        # plt.scatter(y_pol, z_pol)
        # plt.scatter(scaled_extremos[1,:], scaled_extremos[2,:])

        # plt.subplot(2,3,l+4)
        # plt.scatter(clus[1,:], clus[0,:])
        # plt.scatter(y_pol, x_pol)
        # plt.scatter(scaled_extremos[1,:], scaled_extremos[0,:])

        # plt.title(f"Fit for cluster: {l}")
        # # logger.debug(f"Min z value, max z value {np.min(z_pol)}, {np.max(z_pol)}")
        # logger.debug(f"Min z value, max z value {np.min(clus[2,:])}, {np.max(clus[2,:])}")
        # # logger.debug(f"Min y value, max y value {np.min(y_pol)}, {np.max(y_pol)}")
        # logger.debug(f"Min y value, max y value {np.min(clus[1,:])}, {np.max(clus[1,:])}")

    plt.show()

    logger.info(f"Evaluating fits")

    # resultados_eval = evaluar_ajuste(y_pols, z_pols, rotated_vertices, vano_length, clusters)

    return np.array(pols), params, metrics_vano


def puntuate_and_save(response_vano, fit1, fit2, fit3, params, metrics, n_conds):

    logger.success(f"Saving results")
    logger.success(f"Setting vano score with new puntuation function")

    response_vano["RECONSTRUCCION"] = "Posible"

    if response_vano["FLAG"] == "None":
        response_vano["FLAG"] = "good_fit"

    response_vano["CONDUCTORES_CORREGIDOS"][str(0)] = fit1.T.tolist()
    response_vano["CONDUCTORES_CORREGIDOS"][str(1)] = fit2.T.tolist()
    response_vano["CONDUCTORES_CORREGIDOS"][str(2)] = fit3.T.tolist()
    response_vano["PARAMETROS_a_h_k"][str(0)] = params[0]
    response_vano["PARAMETROS_a_h_k"][str(1)] = params[1]
    response_vano["PARAMETROS_a_h_k"][str(2)] = params[2]

    # response_vano["PUNTUACION_APOSTERIORI"] = puntuacion_aposteriori(metrics, n_conds)

    return response_vano
