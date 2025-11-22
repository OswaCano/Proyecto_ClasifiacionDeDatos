
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Cargar el dataset
df = pd.read_csv("winequality-red.csv")

# Separar variables (X) y etiqueta (y)
X = df.drop("quality", axis=1)
y = df["quality"]

# Escalar los datos (muy importante para KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configurar K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluar KNN  con distintos valores de k
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring="accuracy")
    accuracy_scores.append(scores.mean())

# Mostrar resultados
best_k = k_values[np.argmax(accuracy_scores)]
best_score = max(accuracy_scores)

print(f" Mejor número de vecinos (k): {best_k}")
print(f" Mejor precicion promedio: {best_score:.4f}")

# Gráfica del rendimiento
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='blue')
plt.title("Rendimiento del modelo KNN(clasificacion) según número de vecinos (k)")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Precicion promedio (5-Fold CV)")
plt.grid(True)
plt.xticks(k_values)
plt.show()


