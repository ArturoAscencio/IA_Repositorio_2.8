import numpy as np
from sklearn.mixture import GaussianMixture

# Generar datos de ejemplo
np.random.seed(0)
X = np.concatenate([np.random.normal(0, 1, 300), np.random.normal(5, 1, 700)]).reshape(-1, 1)

# Ajustar un modelo de mezcla de Gaussianas con EM
modelo_em = GaussianMixture(n_components=2, random_state=0)
modelo_em.fit(X)

# Obtener las estimaciones de los parámetros
medias = modelo_em.means_
covarianzas = modelo_em.covariances_

print("Medias:", medias)
print("Covarianzas:", covarianzas)
