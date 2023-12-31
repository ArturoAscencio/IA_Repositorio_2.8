from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(0)
X = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)]).reshape(-1, 1)

# Aplicar K-Means
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Obtener las etiquetas de cl�ster para cada punto de datos
etiquetas = kmeans.labels_

# Graficar los datos agrupados
plt.scatter(X, np.zeros_like(X), c=etiquetas, cmap='viridis')
plt.show()
