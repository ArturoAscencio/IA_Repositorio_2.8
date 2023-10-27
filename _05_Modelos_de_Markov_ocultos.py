import numpy as np
from hmmlearn import hmm

# Datos de ejemplo
observaciones = np.array([[1], [2], [3], [4], [5]]).reshape(-1, 1)

# Crear y ajustar un modelo HMM con 2 estados
modelo_hmm = hmm.GaussianHMM(n_components=2, covariance_type="full")
modelo_hmm.fit(observaciones)

# Generar secuencias de estados ocultos y probabilidades de observaciones
estados_ocultos, _ = modelo_hmm.sample(5)

print("Secuencia de estados ocultos:", estados_ocultos)
