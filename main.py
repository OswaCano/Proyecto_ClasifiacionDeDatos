
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

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
plt.savefig("knn_clasificacion.png", dpi=300, bbox_inches='tight')
plt.show()

knn = KNeighborsClassifier(n_neighbors=best_k)

# Split solo para importancia
X_imp_train, X_imp_test, y_imp_train, y_imp_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenamos KNN para importancia
knn.fit(X_imp_train, y_imp_train)

# Permutation Importance
result = permutation_importance(
    knn, X_imp_test, y_imp_test,
    n_repeats=10,
    random_state=42
)

importances = result.importances_mean
std = result.importances_std
features = X.columns

# Ordenar de mayor a menor
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), importances[indices])
plt.xticks(range(len(features)), features[indices], rotation=45, ha='right')
plt.title("Permutation Feature Importance (KNN)")
plt.tight_layout()

# Guardar imagen en PNG
plt.savefig("knn_feature_importance.png", dpi=300)
plt.show()