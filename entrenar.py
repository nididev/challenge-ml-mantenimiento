
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Cargar el dataset
print("Cargando dataset de sensores...")

datos = pd.read_csv("sensores_planta.csv")

# 2. Limpiar datos
datos = datos.dropna()

print(f"Registros disponibles: {len(datos)}")

# 3. Separar variables predictoras y objetivo
X = datos[
    [
        "temperatura_c",
        "vibracion_mms",
        "horas_operacion",
        "presion_psi"
    ]
]

y = datos["estado"]

# 4. Dividir los datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Crear modelo Random Forest
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6. Entrenar modelo
print("Entrenando modelo...")
modelo.fit(X_train, y_train)

# 7. Realizar predicciones
y_pred = modelo.predict(X_test)

# 8. Evaluar precisión
precision = accuracy_score(y_test, y_pred)

print("\n===================================")
print(" RESULTADOS DEL ENTRENAMIENTO")
print("===================================")

print(f"Precisión del modelo: {precision * 100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 9. Guardar modelo
joblib.dump(modelo, "modelo_mantenimiento.pkl")

print("\nModelo guardado correctamente.")
print("Archivo: modelo_mantenimiento.pkl")

if precision >= 0.80:
    print("¡OBJETIVO CUMPLIDO! Precisión igual o superior al 80%.")
else:
    print("La precisión es inferior al 80%. Se requiere revisar el modelo.")
