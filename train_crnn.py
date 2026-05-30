
from data_processor import load_datasets
from model_architecture import build_crnn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

#  RUTES
PATH_KCL     = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\26_29_09_2017_KCL\26-29_09_2017_KCL"
PATH_HEALTHY = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\train-clean-100"
PATH_DB_IT   = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\DB_IT"

#  LOAD WITH AUGMENTATION
#  aug_pd=4   → Each PD audio file generates 1 original + 4 variants
#  aug_healthy=4 → equal to PD

X, y = load_datasets(
    PATH_KCL, PATH_HEALTHY, PATH_DB_IT,
    max_samples=457,
    aug_pd=4,
    aug_healthy=4
)

# ==========================s==================================
#  SPLIT: IMPORTANTE — do this AFTER the augmentation
#  so that all variants of the same original audio
#  end up on the same side (prevents data leakage).

#  The following block identifies which tracks are original
#  and which are enhanced, and splits them at the original audio level.
# ============================================================

# Number of original files per class (before augmentation)
# aug_pd+1 = 5 variants per PD audio file; aug_healthy+1 = 5 per healthy audio file
AUG_PD      = 5   # 1 original + 4 augmented
AUG_HEALTHY = 5   

n_pd      = np.sum(y == 1)
n_healthy = np.sum(y == 0)

# Indexes of the ORIGINAL audio files within the complete array
# (each group of AUG_PD consecutive samples comes from the same file)
pd_original_indices      = np.arange(0, n_pd, AUG_PD)           
healthy_original_indices = np.arange(n_pd, n_pd + n_healthy, AUG_HEALTHY)

# Split at the original file level (80/20)
from sklearn.model_selection import train_test_split as tts

pd_train_orig, pd_test_orig = tts(pd_original_indices, test_size=0.2, random_state=42)
hc_train_orig, hc_test_orig = tts(healthy_original_indices, test_size=0.2, random_state=42)

def expand_indices(orig_indices, group_size, total_samples):
    """Given an array of source indices, expand to all the incremented indices."""
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

# Shuffle the data so that the model does not learn the order
rng = np.random.default_rng(42)
rng.shuffle(train_idx)
rng.shuffle(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"\nTrain: {len(X_train)} muestras  |  Test: {len(X_test)} muestras")
print(f"Train PD: {np.sum(y_train==1)}  |  Train HC: {np.sum(y_train==0)}")
print(f"Test  PD: {np.sum(y_test==1)}   |  Test  HC: {np.sum(y_test==0)}")


#  CLASS WEIGHTS (to compensate for any residual imbalance)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print(f"\nPesos de clase: {class_weights}")

#  MODEL

input_shape = (X.shape[1], X.shape[2], X.shape[3])  # (frames, 40, 3)
model = build_crnn(input_shape)

# ============================================================
#  CALLBACKS
#  - EarlyStopping: if val_loss does not improve after 8 epochs
#  - ReduceLROnPlateau: halves the learning rate if it plateaus
#  - ModelCheckpoint: saves only the best model (not the latest one)
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

#  TRAINING

print("\nStarting training with augmentation and regularisation callbacks...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,          
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save the final version as well
model.save("parkinson_crnn_model.h5")
print("\n✅ Modelo guardado: 'parkinson_crnn_model.h5'")
print("✅ Mejor checkpoint: 'best_parkinson_crnn.h5'")


#  ASSESSMENT AND METRICS

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("\nGenerating predictions on the test set...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "="*40)
print("EVALUATION RESULTS")
print("="*40)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")
print("\nFull report:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Parkinson (1)']))

# Training curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train loss')
axes[0].plot(history.history['val_loss'], label='Val loss')
axes[0].set_title('Loss by epoch')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='Train acc')
axes[1].plot(history.history['val_accuracy'], label='Val acc')
axes[1].set_title('Accuracy by epoch')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Curves saved in 'training_curves.png'")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 14},
            xticklabels=['Prediction: Healthy', 'Prediction: Parkinson'],
            yticklabels=['Real: Healthy', 'Real: Parkinson'])
plt.title('Confusion matrix', fontsize=16)
plt.ylabel('Real truth', fontsize=12)
plt.xlabel('Model prediction', fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("Matrix saved in 'confusion_matrix.png'")
plt.show()
