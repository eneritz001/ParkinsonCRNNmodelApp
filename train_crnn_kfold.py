
from data_processor import load_datasets
from model_architecture import build_crnn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  ROUTES
PATH_KCL     = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\26_29_09_2017_KCL\26-29_09_2017_KCL"
PATH_HEALTHY = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\train-clean-100"
PATH_DB_IT   = r"C:\Users\aldec\Downloads\ParkinsonVoiceApp\DB_IT"

#  LOAD WITH AUGMENTATION
AUG_PD      = 5   # 1 original + 4 augmented
AUG_HEALTHY = 5   

X, y = load_datasets(
    PATH_KCL, PATH_HEALTHY, PATH_DB_IT,
    max_samples=457,
    aug_pd=AUG_PD - 1,
    aug_healthy=AUG_HEALTHY - 1
)

#  RECONSTRUCTING GROUPS (original audio) TO AVOID LEAKAGE

n_pd      = int(np.sum(y == 1))
n_healthy = int(np.sum(y == 0))

# List of (start_index, label) for each original audio file
original_groups = []
i = 0
while i < n_pd:
    original_groups.append((i, AUG_PD, 1))
    i += AUG_PD
i = n_pd
while i < n_pd + n_healthy:
    original_groups.append((i, AUG_HEALTHY, 0))
    i += AUG_HEALTHY

original_groups = np.array(original_groups)  
group_labels = original_groups[:, 2]          

print(f"\nAudios originales: {len(original_groups)} "
      f"(PD: {np.sum(group_labels==1)}, HC: {np.sum(group_labels==0)})")

#  STRATIFIED CROSS-VALIDATION (5 folds, at audio level)

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

acc_scores, f1_scores = [], []
sens_scores, spec_scores, prec_scores = [], [], []
all_conf = np.zeros((2, 2), dtype=int)

input_shape = (X.shape[1], X.shape[2], X.shape[3])


def metrics_from_confusion(cm):
    """
    Calculate the 5 metrics from a 2x2 confusion matrix.
    Convention: class 1 = Parkinson’s (positive), class 0 = Healthy (negative).

    Matriz sklearn (labels [0,1]):
        cm[0,0]=TN  cm[0,1]=FP
        cm[1,0]=FN  cm[1,1]=TP
    """
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    accuracy    = (tp + tn) / total          if total else 0.0
    sensitivity = tp / (tp + fn)             if (tp + fn) else 0.0   # recall PD
    specificity = tn / (tn + fp)             if (tn + fp) else 0.0
    precision   = tp / (tp + fp)             if (tp + fp) else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) else 0.0)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def print_metrics(m, titulo):
    print(f"\n{titulo}")
    print(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
    print(f"  Accuracy     (TP+TN)/Total      : {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
    print(f"  Sensitivity  TP/(TP+FN)         : {m['sensitivity']:.4f} ({m['sensitivity']*100:.2f}%)")
    print(f"  Specificity  TN/(TN+FP)         : {m['specificity']:.4f} ({m['specificity']*100:.2f}%)")
    print(f"  Precision    TP/(TP+FP)         : {m['precision']:.4f} ({m['precision']*100:.2f}%)")
    print(f"  F1-Score     2*(P*S)/(P+S)      : {m['f1']:.4f}")

def expand(group_rows, total):
    """Expands rows (start, size, label) to all sample indices."""
    idx = []
    for start, size, _ in group_rows:
        for j in range(size):
            c = start + j
            if c < total:
                idx.append(c)
    return np.array(idx)

total_samples = len(X)

for fold, (train_g, test_g) in enumerate(
        skf.split(original_groups, group_labels), start=1):

    print("\n" + "="*50)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("="*50)

    # Expand group indexes → sample indexes
    train_idx = expand(original_groups[train_g], total_samples)
    test_idx  = expand(original_groups[test_g],  total_samples)

    rng = np.random.default_rng(42 + fold)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train PD/HC: {np.sum(y_train==1)}/{np.sum(y_train==0)}  "
          f"Test PD/HC: {np.sum(y_test==1)}/{np.sum(y_test==0)}")

    # Weight classes for this category
    w = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    cw = {0: w[0], 1: w[1]}

    # A new model for each fold (do not reuse weights)
    model = build_crnn(input_shape)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=0),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60, batch_size=32,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate this fold
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    cm_fold = confusion_matrix(y_test, y_pred, labels=[0, 1])
    all_conf += cm_fold

    m = metrics_from_confusion(cm_fold)
    acc_scores.append(m["accuracy"])
    f1_scores.append(m["f1"])
    sens_scores.append(m["sensitivity"])
    spec_scores.append(m["specificity"])
    prec_scores.append(m["precision"])

    print_metrics(m, f"--- Métricas Fold {fold} ---")

#  AGGREGATED RESULTS
acc_scores  = np.array(acc_scores)
f1_scores   = np.array(f1_scores)
sens_scores = np.array(sens_scores)
spec_scores = np.array(spec_scores)
prec_scores = np.array(prec_scores)

print("\n" + "="*55)
print("CROSS-VALIDATION RESULTS (5-fold) — mean ± standard deviation")
print("="*55)

def resumen(nombre, arr):
    print(f"  {nombre:<13}: {arr.mean():.4f} ± {arr.std():.4f}  "
          f"({arr.mean()*100:.2f}% ± {arr.std()*100:.2f}%)")

resumen("Accuracy",    acc_scores)
resumen("Sensitivity", sens_scores)
resumen("Specificity", spec_scores)
resumen("Precision",   prec_scores)
resumen("F1-Score",    f1_scores)

print("\nBreakdown per fold:")
for i in range(N_SPLITS):
    print(f"  Fold {i+1}:  acc={acc_scores[i]:.4f}  sens={sens_scores[i]:.4f}  "
          f"spec={spec_scores[i]:.4f}  prec={prec_scores[i]:.4f}  f1={f1_scores[i]:.4f}")

# Metrics calculated on the aggregated matrix (sum of the 5 folds)
print("\nAggregate confusion matrix (sum of the 5 folds):")
print(all_conf)
m_agg = metrics_from_confusion(all_conf)
print_metrics(m_agg, "--- Metrics on an aggregated matrix ---")


#  GRAPHICS

# Aggregate confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(all_conf, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 14},
            xticklabels=['Prediction: Healthy', 'Prediction: Parkinson'],
            yticklabels=['Real: Healthy', 'Real: Parkinson'])
plt.title('Aggregate Confusion Matrix (5-fold CV)', fontsize=15)
plt.ylabel('Real truth')
plt.xlabel('Model prediction')
plt.tight_layout()
plt.savefig("confusion_matrix_cv.png", dpi=300)
print("\nMatrix saved in 'confusion_matrix_cv.png'")

# Accuracy bars per fold, with the average
plt.figure(figsize=(8, 5))
folds = [f"Fold {i+1}" for i in range(N_SPLITS)]
plt.bar(folds, acc_scores * 100, color='#378ADD')
plt.axhline(acc_scores.mean() * 100, color='#D85A30', linestyle='--',
            label=f'Media: {acc_scores.mean()*100:.2f}%')
plt.ylim(min(80, acc_scores.min()*100 - 5), 100)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per fold')
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_per_fold.png", dpi=150)
print("Graph saved in 'accuracy_per_fold.png'")
plt.show()

#  TRAIN THE FINAL MODEL USING ALL THE DATA 

print("\n" + "="*50)
print("Training the final model using all the data for the app...")
print("="*50)

w_all = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
cw_all = {0: w_all[0], 1: w_all[1]}

final_model = build_crnn(input_shape)
final_model.fit(
    X, y,
    epochs=20, batch_size=32,
    class_weight=cw_all,
    verbose=1
)
final_model.save("parkinson_crnn_model.h5")
print("Final model saved: 'parkinson_crnn_model.h5'")
