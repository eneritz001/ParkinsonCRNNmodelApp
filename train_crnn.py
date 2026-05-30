
from data_processor import load_datasets
from model_architecture import build_crnn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# ============================================================
#  RUTAS
# ============================================================
PATH_KCL     = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\26_29_09_2017_KCL\26-29_09_2017_KCL"
PATH_HEALTHY = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\train-clean-100"
PATH_DB_IT   = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\DB_IT"

# ============================================================
#  CARGA CON AUGMENTATION
#  aug_pd=4   → cada audio PD genera 1 original + 4 variantes
#  aug_healthy=4 → igualado a PD: ~2000 sanos vs ~2000 PD
# ============================================================
X, y = load_datasets(
    PATH_KCL, PATH_HEALTHY, PATH_DB_IT,
    max_samples=457,
    aug_pd=4,
    aug_healthy=4
)

# ==========================s==================================
#  SPLIT: IMPORTANTE — hacerlo DESPUÉS del augmentation
#  para que todas las variantes de un mismo audio original
#  caigan en el mismo lado (evita data leakage).
#
#  El bloque siguiente reconstruye qué índices son originales
#  vs aumentados y hace el split a nivel de audio original.
# ============================================================

# Número de archivos originales por clase (antes del augmentation)
# aug_pd+1 = 5 variantes por audio PD; aug_healthy+1 = 5 por sano
AUG_PD      = 5   # 1 original + 4 aumentados
AUG_HEALTHY = 5   # 1 original + 4 aumentados (igualado a PD)

n_pd      = np.sum(y == 1)
n_healthy = np.sum(y == 0)

# Índices de los audios ORIGINALES dentro del array completo
# (cada grupo de AUG_PD muestras consecutivas viene del mismo archivo)
pd_original_indices      = np.arange(0, n_pd, AUG_PD)           # 0, 5, 10, ...
healthy_original_indices = np.arange(n_pd, n_pd + n_healthy, AUG_HEALTHY)

# Split a nivel de archivo original (80/20)
from sklearn.model_selection import train_test_split as tts

pd_train_orig, pd_test_orig = tts(pd_original_indices, test_size=0.2, random_state=42)
hc_train_orig, hc_test_orig = tts(healthy_original_indices, test_size=0.2, random_state=42)

def expand_indices(orig_indices, group_size, total_samples):
    """Dado un array de índices de origen, expande a todos los aumentados."""
    idx = []
    for i in orig_indices:
        for j in range(group_size):
            candidate = i + j
            if candidate < total_samples:
                idx.append(candidate)
    return np.array(idx)

pd_train_idx = expand_indices(pd_train_orig, AUG_PD, n_pd)
pd_test_idx  = expand_indices(pd_test_orig,  AUG_PD, n_pd)
hc_train_idx = expand_indices(hc_train_orig, AUG_HEALTHY, n_pd + n_healthy)
hc_test_idx  = expand_indices(hc_test_orig,  AUG_HEALTHY, n_pd + n_healthy)

train_idx = np.concatenate([pd_train_idx, hc_train_idx])
test_idx  = np.concatenate([pd_test_idx,  hc_test_idx])

# Mezclar para que el modelo no aprenda el orden
rng = np.random.default_rng(42)
rng.shuffle(train_idx)
rng.shuffle(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"\nTrain: {len(X_train)} muestras  |  Test: {len(X_test)} muestras")
print(f"Train PD: {np.sum(y_train==1)}  |  Train HC: {np.sum(y_train==0)}")
print(f"Test  PD: {np.sum(y_test==1)}   |  Test  HC: {np.sum(y_test==0)}")

# ============================================================
#  PESOS DE CLASE (compensan cualquier desbalance residual)
# ============================================================
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print(f"\nPesos de clase: {class_weights}")

# ============================================================
#  MODELO
# ============================================================
input_shape = (X.shape[1], X.shape[2], X.shape[3])  # (frames, 40, 3)
model = build_crnn(input_shape)

# ============================================================
#  CALLBACKS
#  - EarlyStopping: para cuando val_loss no mejore en 8 épocas
#  - ReduceLROnPlateau: reduce el LR a la mitad si se estanca
#  - ModelCheckpoint: guarda solo el mejor modelo (no el último)
# ============================================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_parkinson_crnn.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
#  ENTRENAMIENTO
# ============================================================
print("\nIniciando entrenamiento con augmentation y callbacks de regularización...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,           # EarlyStopping lo cortará antes si procede
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

# Guardar también la versión final (restore_best_weights ya la tiene)
model.save("parkinson_crnn_model.h5")
print("\n✅ Modelo guardado: 'parkinson_crnn_model.h5'")
print("✅ Mejor checkpoint: 'best_parkinson_crnn.h5'")


# ============================================================
#  EVALUACIÓN Y MÉTRICAS
# ============================================================
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("\nGenerando predicciones sobre el conjunto de test...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "="*40)
print("RESULTADOS DE LA EVALUACIÓN")
print("="*40)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")
print("\nReporte completo:")
print(classification_report(y_test, y_pred, target_names=['Sano (0)', 'Parkinson (1)']))

# --- Curva de entrenamiento ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train loss')
axes[0].plot(history.history['val_loss'], label='Val loss')
axes[0].set_title('Pérdida por época')
axes[0].set_xlabel('Época')
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='Train acc')
axes[1].plot(history.history['val_accuracy'], label='Val acc')
axes[1].set_title('Accuracy por época')
axes[1].set_xlabel('Época')
axes[1].legend()

plt.tight_layout()
plt.savefig("curvas_entrenamiento.png", dpi=150)
print("Curvas guardadas en 'curvas_entrenamiento.png'")

# --- Matriz de confusión ---
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 14},
            xticklabels=['Predicción: Sano', 'Predicción: Parkinson'],
            yticklabels=['Real: Sano', 'Real: Parkinson'])
plt.title('Matriz de Confusión', fontsize=16)
plt.ylabel('Verdad real', fontsize=12)
plt.xlabel('Predicción del modelo', fontsize=12)
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=300)
print("Matriz guardada en 'matriz_confusion.png'")
plt.show()
