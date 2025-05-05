import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random_forest import RandomForest
from collections import defaultdict
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_digits

### CREDIT FOR RANDOM FOREST CODE: ANUSHKA TREHAN (HW#3)
def encode_data(col):
    col = col.replace('?', np.nan)
    col = col.fillna("MISSING")
    unique_vals = sorted(set(col))
    val_to_index = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = np.array([val_to_index[val] for val in col])
    return encoded, val_to_index

def encode_df(df, target_col):
    feature_types = []
    df = df.dropna(subset=[target_col])

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == object:
            df[col], _ = encode_data(df[col])
            feature_types.append('categorical')
        else:
            feature_types.append('numerical')

    # encode EVERYTHING -- thinking - this would definitely reduce accuracy --- maybe change the voting fucntion in Random forest file
    df[target_col], _ = encode_data(df[target_col].astype(str).str.strip())
    return df, feature_types


def normalize_features(df, feature_types, target_col):
    feature_cols = [col for col in df.columns if col != target_col]
    for i, col in enumerate(feature_cols):
        if feature_types[i] == 'numerical':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0
    return df


def eval(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, recall, f1


def k_fold_impl(X, y, k=10, seed=42):
    np.random.seed(seed)
    X = np.array(X)
    y = np.array(y)
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    for label in class_indices:
        np.random.shuffle(class_indices[label])

    class_folds = {}
    for label in class_indices:
        class_folds[label] = np.array_split(class_indices[label], k)

    folds = []
    for fold_id in range(k):
        test_indices = []
        for label in class_folds:
            test_indices.extend(class_folds[label][fold_id])
        test_indices = np.array(test_indices)
        train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
        folds.append((train_indices, test_indices))

    return folds

def run_random_forest_experiment(data, dataset_name, target_col, is_path=True):
    
    if is_path:
        df = pd.read_csv(data)
    else:
        df = data

    df, feature_types = encode_df(df, target_col)
    df = df.dropna(subset=[target_col])  # drop rows with missing labels this is causing issues with the credit dataet
    df = normalize_features(df, feature_types, target_col)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    XX = np.column_stack((X, y))


    ntree_vals = [1, 5, 10, 20, 30, 40, 50]
    folds = k_fold_impl(X, y, k=10)
    avg_metrics = {}

    for ntree in ntree_vals:
        print(f"\n--- evaluating RF with {ntree} trees ---")
        metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            print(f"    Fold {fold_idx+1}/10...", end=" ")
            rf = RandomForest(num_trees=ntree, max_depth=10)
            rf.bootstrapping(XX[train_idx])
            for tree in rf.decision_trees:
                tree.criterion = 'entropy'
            rf.fitting(feature_types)
            y_pred = rf.voting(X[test_idx])
            acc, prec, rec, f1 = eval(y[test_idx], y_pred)
            metrics.append((acc, prec, rec, f1))
            print("✓")
        avg_metrics[ntree] = np.mean(metrics, axis=0)

    print("\n plot result for this dataset: ")
    plot_metrics(avg_metrics, dataset_name)
    print_metrics_table(avg_metrics, dataset_name)

def plot_metrics(avg_metrics, dataset_name):
    ntree_vals = sorted(avg_metrics.keys())
    accs = [avg_metrics[n][0] for n in ntree_vals]
    precs = [avg_metrics[n][1] for n in ntree_vals]
    recalls = [avg_metrics[n][2] for n in ntree_vals]
    f1s = [avg_metrics[n][3] for n in ntree_vals]

    metrics = [('Accuracy', accs), ('Precision', precs), ('Recall', recalls), ('F1 Score', f1s)]

    for label, values in metrics:
        plt.figure()
        plt.plot(ntree_vals, values, marker='o')
        plt.title(f'{label} vs ntree ({dataset_name})')
        plt.xlabel('# trees')
        plt.ylabel(label)
        plt.grid(True)
        plt.savefig(f'{dataset_name}_rf_{label.lower().replace(" ", "_")}.png')

def print_metrics_table(avg_metrics, dataset_name):
    print(f"\n result for {dataset_name.upper()} dataset:")
    print(f"{'ntree':>6} | {'accu':>9} | {'prec':>9} | {'recall':>7} | {'f1':>9}")
    print("-" * 50)
    for ntree in sorted(avg_metrics.keys()):
        acc, prec, rec, f1 = avg_metrics[ntree]
        print(f"{ntree:>6} | {acc:>9.4f} | {prec:>9.4f} | {rec:>7.4f} | {f1:>9.4f}")

def main():
    

    # Digits dataset (in-memory)
    digits = load_digits()
    df_digits = pd.DataFrame(digits.data)
    df_digits['label'] = digits.target
    run_random_forest_experiment(df_digits, "digits", "label", is_path=False)
    datasets = {
        "credit_approval": ("credit_approval.csv", "label"),
        "parkinsons": ("parkinsons.csv", "Diagnosis"),
        "rice": ("rice.csv", "label"),
    }

    for name, (path, target) in datasets.items():
        run_random_forest_experiment(path, name, target, is_path=True)

if __name__ == "__main__":
    main()
