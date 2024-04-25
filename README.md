# electra-ia
- Python 3D Point Cloud spatial analysis
- 3D LIDAR Data curation, preprocess and analysis
- Application of ML models for Point Classification and Catenaria regression fit

  Preguntas: https://docs.google.com/document/d/13Um_0wgUGBLbGws6rkAb4e86pBjA_p_IMFoog77eHFU/edit?usp=sharing

-----------------------------------------------------------------------------------
- ## Objetivos principales:

1. #### Dar una nota a cada VANO --> definir criterios juntos
    - Hay que definir una métrica para evaluar el estado del vano:
    
        1. Nota por clasificación = Proporcion de puntos bien clasificados (2 categorias) - Accuracy?
        2. Nota por consistencia de puntos = Referencia Catenaria, Bézier, C-spline... - RMSE? + Correlation? - Likelyhood?

2. #### Calcular una formula que interpole los los puntos conductores---> puntos azul clarito.

    - Interpolación por ajuste de Catenaria, Bezier, C-spline
    - Interpolacion con modelo de Machine Learning/DL - Regression... - Autoencoder?

3. #### Reclasificar, clasificar mejor los puntos del conductor y de los palos

    - Explorar opciones: Reetiquetado, utilizar resultados del ajuste de conductor, clusterizar puntos bien clasificados?, sistema de reglas?

4. #### Distancia cadenaria a suelo. Distancia Minima. Nos pasan la nube de puntos del suelo.

    - Es necesario definir el suelo y (ajustar los puntos con una interpolacion?)
    - Distancia del conductor más cercano al suelo o para cada conductor?
    - Distancia de referencia = Disancia 2D entre apoyos/Distancia 3D conductor (recalculado)


- ## Resources:

1. #### Datasets: 
    - Semantic3D Segmentation Dataset: https://www.kaggle.com/datasets/kmader/point-cloud-segmentation/data
    - Other dataset: https://www.kaggle.com/datasets/kmader/point-cloud-segmentation/data
    - Really big dataset for semantic segmentation: https://www.kaggle.com/datasets/priteshraj10/point-cloud-lidar-toronto-3d

    - Point Cloud classification Datasets: https://paperswithcode.com/datasets?q=&v=lst&o=cited&task=few-shot-point-cloud-classification&mod=3d

2. #### Literature

    - Point Completion Network: https://paperswithcode.com/paper/pcn-point-completion-network
    - GCNN: https://paperswithcode.com/paper/dynamic-graph-cnn-for-learning-on-point

    - (Semantic segmentation):

        - 3D classification and segmentation: https://paperswithcode.com/paper/pointnet-deep-learning-on-point-sets-for-3d
        - Hierarchical Feature Learning: https://paperswithcode.com/paper/pointnet-deep-hierarchical-feature-learning
        - Point classification: https://paperswithcode.com/task/3d-point-cloud-classification

    - Specific literature:

        - https://idus.us.es/handle/11441/128964
        - https://crea.ujaen.es/bitstream/10953.1/14012/1/Memoria%20-%20Segmentacion%20inteligente%20de%20objetos%20en%20nubes%20de%20puntos.pdf
        - https://repositorio.unal.edu.co/handle/unal/78223
        - https://www.mdpi.com/2072-4292/15/9/2371
        - https://www.researchgate.net/publication/354251923_Classification_of_high-voltage_power_line_structures_in_low_density_ALS_data_acquired_over_broad_non-urban_areas
        - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8444095/
        - https://www.sciencedirect.com/science/article/pii/S2405896323009291
        - https://www.mdpi.com/2072-4292/11/22/2600

3. #### Code examples and libraries

    - PCL: https://pointclouds.org/documentation/
    - Open3D:

    - Examples:

        - Clustering: https://www.kaggle.com/code/mohamedhbaieb/p2m-3d-point-cloud#Reading-Functions-data-domfountain_station1

        - Point Reconstruction (Autoencoder): https://www.kaggle.com/code/horsek/transformer-based-autoencoder-for-3d-point-cloud
