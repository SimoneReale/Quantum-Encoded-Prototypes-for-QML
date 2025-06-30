import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.utils.algorithm_globals import algorithm_globals

from alive_progress import alive_bar
from scipy.stats import pearsonr
import matplotlib.patches as mpatches


def load_data(dataset_name: str, target_size=150, n_features=4):
    """
    Loads, preprocesses, reduces dimensionality with PCA, and downsamples datasets.
    """
    if dataset_name == "iris":
        data_loader = load_iris()
        species_mapping = {0: 0, 1: 1, 2: 1}
    elif dataset_name == "wine":
        data_loader = load_wine()
        species_mapping = {0: 0, 1: 1, 2: 1}
    elif dataset_name == "breast_cancer":
        data_loader = load_breast_cancer()
        species_mapping = None
    else:
        raise ValueError("Dataset not supported. Choose 'iris', 'wine', or 'breast_cancer'.")

    data = pd.DataFrame(data_loader.data, columns=data_loader.feature_names)
    data["target"] = data_loader.target

    if species_mapping:
        data["target"] = data["target"].map(species_mapping)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if X.shape[1] > n_features:
        print(f"Reducing '{dataset_name}' from {X.shape[1]} to {n_features} features using PCA...")
        pca = PCA(n_components=n_features)
        X_pca = pca.fit_transform(X)
        data = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_features)])
        data['target'] = y.values
    
    if len(data) > target_size:
        print(f"Downsampling '{dataset_name}' from {len(data)} to {target_size} samples...")
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=target_size, stratify=y, random_state=42
        )
        data = pd.DataFrame(X_sample, columns=data.columns[:-1])
        data['target'] = y_sample.values

    feature_columns = data.columns[:-1]
    for col in feature_columns:
        min_val, max_val = data[col].min(), data[col].max()
        data[col] = np.pi/2 * (2 * (data[col] - min_val) / (max_val - min_val) - 1) if max_val > min_val else 0

    return data


def split_data(data):
    """Splits data into training/testing sets and one-hot encodes the target."""
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot = train_test_split(
        X, y, y_one_hot, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot


def find_prototypes_optimized(X, y, kernel_matrix, n_prototypes_per_class):
    """Optimized version that finds prototypes using incremental MMD calculations."""
    P_indices = []
    mmd_state = {0: {'sum_A': 0.0, 'sum_B': 0.0, 'term_C': 0.0, 'prototypes': []},
                 1: {'sum_A': 0.0, 'sum_B': 0.0, 'term_C': 0.0, 'prototypes': []}}

    for class_label in [0, 1]:
        class_indices = np.where(y == class_label)[0]
        if len(class_indices) > 0:
            sub_kernel = kernel_matrix[np.ix_(class_indices, class_indices)]
            mmd_state[class_label]['term_C'] = np.sum(sub_kernel) / (len(class_indices)**2)

    with alive_bar(n_prototypes_per_class * 2, title=f"Finding {n_prototypes_per_class*2} Prototypes (Optimized)", bar="smooth") as bar:
        for _ in range(n_prototypes_per_class):
            for current_class in [0, 1]:
                min_cost, best_candidate_idx = float("inf"), None
                state = mmd_state[current_class]
                class_indices = np.where(y == current_class)[0]
                if len(class_indices) == 0: continue

                sum_kernel_candidates_X = np.sum(kernel_matrix[:, class_indices], axis=1)
                for candidate_idx in class_indices:
                    if candidate_idx in P_indices: continue
                    
                    current_prototypes = state['prototypes']
                    new_sum_A = state['sum_A'] + kernel_matrix[candidate_idx, candidate_idx]
                    if current_prototypes:
                        new_sum_A += 2 * np.sum(kernel_matrix[candidate_idx, current_prototypes])
                    
                    new_sum_B = state['sum_B'] + sum_kernel_candidates_X[candidate_idx]
                    m = len(current_prototypes) + 1
                    cost = (new_sum_A / (m**2)) - (2 * new_sum_B / (m * len(class_indices))) + state['term_C']

                    if cost < min_cost:
                        min_cost, best_candidate_idx = cost, candidate_idx

                if best_candidate_idx is not None:
                    state_to_update = mmd_state[current_class]
                    if state_to_update['prototypes']:
                        state_to_update['sum_A'] += 2 * np.sum(kernel_matrix[best_candidate_idx, state_to_update['prototypes']])
                    state_to_update['sum_A'] += kernel_matrix[best_candidate_idx, best_candidate_idx]
                    state_to_update['sum_B'] += sum_kernel_candidates_X[best_candidate_idx]
                    state_to_update['prototypes'].append(best_candidate_idx)
                    P_indices.append(best_candidate_idx)
                bar()

    mmd_final = {}
    for c in [0, 1]:
        state, m, n = mmd_state[c], len(mmd_state[c]['prototypes']), len(np.where(y == c)[0])
        mmd_final[c] = (state['sum_A'] / m**2) - (2 * state['sum_B'] / (m * n)) + state['term_C'] if m > 0 and n > 0 else float('inf')

    return P_indices, mmd_final[0], mmd_final[1]


def run_experiment(dataset_name, X_train, y_train, X_test, y_test_one_hot):
    """Runs the full experiment for a given dataset, returning results for plotting."""
    print(f"\n{'='*20} RUNNING EXPERIMENT ON: {dataset_name.upper()} DATASET {'='*20}")
    num_features = X_train.shape[1]

    angle_encoding_circuit = QuantumCircuit(num_features, name="AngleEncoding")
    x_params = ParameterVector("x", num_features)
    for i in range(num_features):
        angle_encoding_circuit.ry(x_params[i], i)

    feature_maps = {
        "ZZFeatureMap": ZZFeatureMap(num_features, reps=2),
        "ZFeatureMap": ZFeatureMap(num_features, reps=2),
        "AngleEncoding": angle_encoding_circuit
    }

    results = {"accuracies": {}, "mmd_class_0": {}, "mmd_class_1": {}, "prototypes": {}}
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)

    for name, fm in feature_maps.items():
        print(f"\n--- Testing Feature Map: {name} ---")
        quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=fm)
        kernel_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
        
        n_prototypes = 3
        prototype_indices, mmd_c0, mmd_c1 = find_prototypes_optimized(
            X_train, y_train, kernel_matrix_train, n_prototypes
        )
        results["mmd_class_0"][name] = mmd_c0
        results["mmd_class_1"][name] = mmd_c1
        results["prototypes"][name] = prototype_indices
        print(f"Final MMD Score (Class 0): {mmd_c0:.4f}")
        print(f"Final MMD Score (Class 1): {mmd_c1:.4f}")

        ansatz = RealAmplitudes(num_features, reps=1)
        vqc = VQC(feature_map=fm, ansatz=ansatz, loss="cross_entropy", optimizer=None)
        vqc.fit(X_train, y_train_one_hot)
        accuracy = vqc.score(X_test, y_test_one_hot)
        results["accuracies"][name] = accuracy
        print(f"VQC Classification Accuracy: {accuracy:.4f}")

    return results


def plot_results(results, dataset_name):
    """Saves validation results for a dataset as a bar chart."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6), sharey=False)
    fig.suptitle(f"Validation Results for {dataset_name.upper()} Dataset", fontsize=16)
    names = list(results["accuracies"].keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    accuracies = list(results["accuracies"].values())
    ax1.bar(names, accuracies, color=colors)
    ax1.set_title("VQC Classification Accuracy")
    ax1.set_ylabel("Accuracy Score")
    ax1.set_ylim(0, 1.1)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    mmd0_scores = list(results["mmd_class_0"].values())
    ax2.bar(names, mmd0_scores, color=colors)
    ax2.set_title("Prototype Quality for Class 0 (Lower is Better)")
    ax2.set_ylabel("MMD Score")
    
    mmd1_scores = list(results["mmd_class_1"].values())
    ax3.bar(names, mmd1_scores, color=colors)
    ax3.set_title("Prototype Quality for Class 1 (Lower is Better)")
    ax3.set_ylabel("MMD Score")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{dataset_name}_results.png")  
    plt.close() 


def plot_prototypes_pca(X, y, prototype_indices, dataset_name, best_fm_name):
    """Saves PCA-reduced view of prototypes as an image."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    prototypes = X[prototype_indices]
    prototypes_pca = pca.transform(prototypes)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', alpha=0.3, label="Class 0")
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', alpha=0.3, label="Class 1")
    plt.scatter(prototypes_pca[:, 0], prototypes_pca[:, 1],
                s=200, c='yellow', marker='X', edgecolor='black', linewidth=1.5,
                label='Selected Prototypes')
    
    plt.title(f"Prototypes for {dataset_name.upper()} using {best_fm_name} (PCA-reduced view)", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset_name}_pca.png") 
    plt.close() 


def create_correlation_plot(all_results):
    """Saves correlation scatter plot as an image."""
    accuracies_flat = []
    avg_mmd_flat = []
    labels = []
    dataset_names = []
    fm_names = []

    for result_item in all_results:
        dataset_name = result_item['dataset_name']
        for fm_name in result_item['accuracies']:
            accuracies_flat.append(result_item['accuracies'][fm_name])
           
            avg_mmd = (result_item['mmd_class_0'][fm_name] + result_item['mmd_class_1'][fm_name]) / 2
            avg_mmd_flat.append(avg_mmd)
            
            labels.append(f"{dataset_name[:4]}-{fm_name[:4]}") 
            dataset_names.append(dataset_name)
            fm_names.append(fm_name)

  
    r, p_value = pearsonr(avg_mmd_flat, accuracies_flat)

   
    plt.figure(figsize=(12, 8))
    
    unique_datasets = list(set(dataset_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_datasets)))
    dataset_color_map = {name: color for name, color in zip(unique_datasets, colors)}
    
    point_colors = [dataset_color_map[d] for d in dataset_names]

    plt.scatter(avg_mmd_flat, accuracies_flat, c=point_colors, s=100, alpha=0.8, edgecolors='k')
    
    z = np.polyfit(avg_mmd_flat, accuracies_flat, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(avg_mmd_flat), p(np.sort(avg_mmd_flat)), "r--", label="Linear Trendline")
    
    plt.title('Correlation between Model Accuracy and Prototype Quality (MMD)', fontsize=16)
    plt.xlabel('Average MMD Score (Lower is Better)', fontsize=12)
    plt.ylabel('VQC Classification Accuracy', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.text(0.95, 0.95, f'Pearson r = {r:.3f}', 
             transform=plt.gca().transAxes, fontsize=15,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    legend_patches = [mpatches.Patch(color=dataset_color_map[name], label=name.replace("_", " ").title()) for name in unique_datasets]
    plt.legend(handles=legend_patches + [plt.Line2D([0], [0], color='r', linestyle='--', label='Linear Trendline')])

    plt.savefig("correlation.png") 
    plt.close()  


if __name__ == "__main__":
    algorithm_globals.random_seed = 42
    datasets_to_run = ["iris", "wine", "breast_cancer"]
    
    all_experiment_results = [] 

    for name in datasets_to_run:
        dataset = load_data(name, target_size=150, n_features=4)
        X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot = split_data(dataset)
        
        results = run_experiment(name, X_train, y_train, X_test, y_test_one_hot)
       
        results['dataset_name'] = name
        all_experiment_results.append(results)
       
        plot_results(results, name)
        
        best_fm_name = max(results["accuracies"], key=results["accuracies"].get)
        best_prototypes_indices = results["prototypes"][best_fm_name]
        print(f"\nVisualizing prototypes for the best performing feature map: {best_fm_name}")
        plot_prototypes_pca(X_train, y_train, best_prototypes_indices, name, best_fm_name)

    print("\n" + "="*20 + " FINAL CORRELATION ANALYSIS " + "="*20)
    create_correlation_plot(all_experiment_results)