"""
================================================================================
Machine Learning Assignment: Gaussian Naive Bayes & PCA from Scratch
================================================================================
Dataset: Breast Cancer Wisconsin (Diagnostic)
Author: Data Science Implementation
Date: December 2025

This implementation covers:
1. Data Preparation with Z-score Normalization
2. Gaussian Naive Bayes Classifier (from scratch)
3. Covariance & Eigen Analysis
4. PCA Dimensionality Reduction (from scratch)
5. Classification on Reduced Data
6. Performance Comparison
7. Advanced 2D Visualization with Gaussian Contours
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# ================================================================================
# STEP 1: DATA PREPARATION
# ================================================================================

def load_and_prepare_data():
    """
    Load the Breast Cancer dataset and prepare it for analysis.
    Returns features (X) and labels (y).
    """
    print("=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80)
    
    # Load the dataset
    data = load_breast_cancer()
    X = data.data  # 30 continuous features
    y = data.target  # Binary: 0 = Malignant, 1 = Benign
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Classes: {target_names} (0 = {target_names[0]}, 1 = {target_names[1]})")
    print(f"\nClass distribution:")
    print(f"  - Malignant (0): {np.sum(y == 0)} samples")
    print(f"  - Benign (1): {np.sum(y == 1)} samples")
    
    return X, y, feature_names, target_names


def standardize_features(X_train, X_test):
    """
    Standardize features using Z-score normalization.
    
    Z-score formula: x' = (x - μ) / σ
    
    Where:
        x  = original feature value
        μ  = mean of the feature (calculated from training data only)
        σ  = standard deviation of the feature (calculated from training data only)
        x' = standardized feature value
    
    IMPORTANT: We fit (calculate μ and σ) on training data only,
    then apply the same transformation to both train and test data
    to prevent data leakage.
    """
    print("\n--- Standardization (Z-score Normalization) ---")
    
    # Calculate mean and std from training data only
    mu = np.mean(X_train, axis=0)  # Mean for each feature
    sigma = np.std(X_train, axis=0)  # Std for each feature
    
    # Avoid division by zero (add small epsilon if std is 0)
    sigma = np.where(sigma == 0, 1e-10, sigma)
    
    # Apply standardization
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    
    print(f"Training data standardized: mean ≈ 0, std ≈ 1")
    print(f"  - Train mean range: [{X_train_std.mean(axis=0).min():.4f}, {X_train_std.mean(axis=0).max():.4f}]")
    print(f"  - Train std range: [{X_train_std.std(axis=0).min():.4f}, {X_train_std.std(axis=0).max():.4f}]")
    
    return X_train_std, X_test_std, mu, sigma


# ================================================================================
# STEP 2: GAUSSIAN NAIVE BAYES FROM SCRATCH
# ================================================================================

class GaussianNB_Scratch:
    """
    Gaussian Naive Bayes Classifier implemented from scratch.
    
    Theory:
    -------
    Naive Bayes is based on Bayes' Theorem:
        P(y|X) = P(X|y) * P(y) / P(X)
    
    For classification, we compute:
        y_pred = argmax_y [ P(y) * ∏ P(x_i|y) ]
    
    For Gaussian (continuous) features, we assume each feature follows
    a Gaussian distribution within each class:
        P(x_i|y) = (1 / √(2πσ²_y)) * exp(-(x_i - μ_y)² / (2σ²_y))
    
    To avoid numerical underflow with many features, we use log probabilities:
        log P(x_i|y) = -0.5 * log(2πσ²_y) - (x_i - μ_y)² / (2σ²_y)
    """
    
    def __init__(self):
        self.classes = None  # Unique class labels
        self.class_priors = {}  # P(y) for each class
        self.means = {}  # Mean of each feature per class
        self.variances = {}  # Variance of each feature per class
    
    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model.
        
        For each class, calculate:
        1. Prior probability: P(y) = count(y) / total_samples
        2. Mean (μ) of each feature
        3. Variance (σ²) of each feature
        """
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes:
            # Get samples belonging to class c
            X_c = X[y == c]
            
            # Prior probability: P(y = c)
            self.class_priors[c] = len(X_c) / n_samples
            
            # Mean of each feature for class c
            self.means[c] = np.mean(X_c, axis=0)
            
            # Variance of each feature for class c
            # Adding small epsilon to avoid division by zero in PDF
            self.variances[c] = np.var(X_c, axis=0) + 1e-9
        
        return self
    
    def _gaussian_pdf_log(self, x, mean, var):
        """
        Calculate the log of Gaussian Probability Density Function.
        
        Log PDF formula:
            log P(x|μ,σ²) = -0.5 * log(2πσ²) - (x - μ)² / (2σ²)
        
        Using log avoids numerical underflow when multiplying many small probabilities.
        """
        # log(2π) ≈ 1.8378770664093453
        log_2pi = np.log(2 * np.pi)
        
        # Log of Gaussian PDF for each feature
        log_pdf = -0.5 * (log_2pi + np.log(var) + ((x - mean) ** 2) / var)
        
        return log_pdf
    
    def _predict_single(self, x):
        """
        Predict the class for a single sample.
        
        We compute log P(y|x) ∝ log P(y) + Σ log P(x_i|y)
        and return the class with highest probability.
        """
        log_posteriors = {}
        
        for c in self.classes:
            # Start with log prior: log P(y = c)
            log_prior = np.log(self.class_priors[c])
            
            # Sum of log likelihoods: Σ log P(x_i|y = c)
            log_likelihood = np.sum(self._gaussian_pdf_log(x, self.means[c], self.variances[c]))
            
            # Log posterior (unnormalized): log P(y|x) ∝ log P(y) + log P(x|y)
            log_posteriors[c] = log_prior + log_likelihood
        
        # Return class with maximum log posterior
        return max(log_posteriors, key=log_posteriors.get)
    
    def predict(self, X):
        """
        Predict classes for multiple samples.
        """
        return np.array([self._predict_single(x) for x in X])
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        Returns probability for each class using softmax on log posteriors.
        """
        probas = []
        for x in X:
            log_posteriors = []
            for c in self.classes:
                log_prior = np.log(self.class_priors[c])
                log_likelihood = np.sum(self._gaussian_pdf_log(x, self.means[c], self.variances[c]))
                log_posteriors.append(log_prior + log_likelihood)
            
            # Convert log posteriors to probabilities using softmax
            log_posteriors = np.array(log_posteriors)
            # Subtract max for numerical stability
            log_posteriors -= np.max(log_posteriors)
            posteriors = np.exp(log_posteriors)
            posteriors /= np.sum(posteriors)
            probas.append(posteriors)
        
        return np.array(probas)


# ================================================================================
# METRICS IMPLEMENTATION FROM SCRATCH
# ================================================================================

def calculate_accuracy(y_true, y_pred):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
             = Number of correct predictions / Total predictions
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


def calculate_precision(y_true, y_pred, positive_class=1):
    """
    Precision = TP / (TP + FP)
    
    Precision measures: Of all predicted positives, how many are actually positive?
    """
    # True Positives: Predicted positive AND actually positive
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    # False Positives: Predicted positive BUT actually negative
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall(y_true, y_pred, positive_class=1):
    """
    Recall (Sensitivity) = TP / (TP + FN)
    
    Recall measures: Of all actual positives, how many did we correctly predict?
    """
    # True Positives
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    # False Negatives: Predicted negative BUT actually positive
    fn = np.sum((y_pred != positive_class) & (y_true == positive_class))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_f1_score(y_true, y_pred, positive_class=1):
    """
    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    
    F1 is the harmonic mean of Precision and Recall.
    It balances both metrics, especially useful for imbalanced datasets.
    """
    precision = calculate_precision(y_true, y_pred, positive_class)
    recall = calculate_recall(y_true, y_pred, positive_class)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def print_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate and print all metrics for a model.
    """
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 40)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ================================================================================
# STEP 3: COVARIANCE & EIGEN ANALYSIS
# ================================================================================

def compute_covariance_and_eigen(X_train_std):
    """
    Compute the covariance matrix and perform eigen decomposition.
    
    Covariance Matrix:
    ------------------
    The covariance matrix captures the relationships between features.
    For standardized data: Cov(X) = (X^T * X) / (n - 1)
    
    Eigenvalues & Eigenvectors:
    ---------------------------
    - Eigenvectors: Directions of maximum variance (principal components)
    - Eigenvalues: Amount of variance explained in each direction
    
    Sorting by eigenvalues (descending) gives us components ordered by importance.
    """
    print("\n" + "=" * 80)
    print("STEP 3: COVARIANCE & EIGEN ANALYSIS")
    print("=" * 80)
    
    # Calculate covariance matrix
    # np.cov expects features as rows, so we transpose
    cov_matrix = np.cov(X_train_std.T)
    
    print(f"\nCovariance Matrix Shape: {cov_matrix.shape}")
    print(f"(30x30 matrix showing relationships between all feature pairs)")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Eigenvalues might be complex due to numerical precision, take real part
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues_sorted)
    explained_variance_ratio = eigenvalues_sorted / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"\n--- Eigenvalue Analysis ---")
    print(f"{'PC':<5} {'Eigenvalue':<15} {'Var. Ratio':<15} {'Cumulative Var.':<15}")
    print("-" * 50)
    for i in range(min(10, len(eigenvalues_sorted))):  # Show top 10
        print(f"PC{i+1:<3} {eigenvalues_sorted[i]:<15.4f} {explained_variance_ratio[i]:<15.4f} {cumulative_variance_ratio[i]:<15.4f}")
    print("...")
    
    return cov_matrix, eigenvalues_sorted, eigenvectors_sorted, explained_variance_ratio


# ================================================================================
# STEP 4: DIMENSIONALITY REDUCTION (PCA FROM SCRATCH)
# ================================================================================

def pca_reduce_dimensions(X_train_std, X_test_std, eigenvectors_sorted, explained_variance_ratio, n_components=2):
    """
    Reduce dimensionality using PCA (Principal Component Analysis).
    
    PCA Projection Formula:
    -----------------------
    X_reduced = X_original · W_top_k
    
    Where:
        X_original: Original data matrix (n_samples × n_features)
        W_top_k: Matrix of top k eigenvectors (n_features × k)
        X_reduced: Reduced data matrix (n_samples × k)
    
    This projection transforms data from the original feature space
    to a new space defined by the principal components.
    """
    print("\n" + "=" * 80)
    print("STEP 4: DIMENSIONALITY REDUCTION (PCA)")
    print("=" * 80)
    
    # Select top n_components eigenvectors
    W_top = eigenvectors_sorted[:, :n_components]
    
    print(f"\nProjection Matrix W Shape: {W_top.shape}")
    print(f"(Using top {n_components} principal components)")
    
    # Project data onto principal components
    X_train_pca = X_train_std @ W_top
    X_test_pca = X_test_std @ W_top
    
    print(f"\nOriginal Training Data Shape: {X_train_std.shape}")
    print(f"Reduced Training Data Shape:  {X_train_pca.shape}")
    print(f"Dimensionality reduced from {X_train_std.shape[1]}D to {X_train_pca.shape[1]}D")
    
    # Variance preserved by top components
    variance_preserved = np.sum(explained_variance_ratio[:n_components])
    
    print(f"\n--- Variance Analysis ---")
    print(f"Variance preserved by PC1: {explained_variance_ratio[0]*100:.2f}%")
    print(f"Variance preserved by PC2: {explained_variance_ratio[1]*100:.2f}%")
    print(f"Total variance preserved by top {n_components} components: {variance_preserved*100:.2f}%")
    print(f"\nInterpretation: The top 2 principal components capture {variance_preserved*100:.2f}% ")
    print(f"of the total variance in the original 30-dimensional data. This means we")
    print(f"retain most of the information while reducing dimensions from 30 to 2.")
    
    return X_train_pca, X_test_pca, W_top, variance_preserved


# ================================================================================
# STEP 7: ADVANCED VISUALIZATION
# ================================================================================

def multivariate_gaussian_pdf(X, mean, cov):
    """
    Calculate the Multivariate Gaussian PDF.
    
    Formula:
    --------
    P(x) = (1 / ((2π)^(d/2) * |Σ|^(1/2))) * exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ))
    
    Where:
        d = number of dimensions
        μ = mean vector
        Σ = covariance matrix
        |Σ| = determinant of covariance matrix
        Σ^(-1) = inverse of covariance matrix
    """
    d = len(mean)
    
    # Regularize covariance matrix to ensure it's invertible
    cov_reg = cov + np.eye(d) * 1e-6
    
    # Calculate determinant and inverse
    cov_det = np.linalg.det(cov_reg)
    cov_inv = np.linalg.inv(cov_reg)
    
    # Normalization constant
    norm_const = 1.0 / (np.power(2 * np.pi, d / 2) * np.sqrt(cov_det))
    
    # Calculate PDF for each point
    diff = X - mean
    if len(X.shape) == 1:
        exponent = -0.5 * (diff @ cov_inv @ diff.T)
    else:
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    
    return norm_const * np.exp(exponent)


def create_visualization(X_test_pca, y_test, model_2d, target_names):
    """
    Create a publication-quality 2D visualization with Gaussian contours.
    
    This visualization shows:
    1. Scatter plot of test data points colored by true class
    2. Gaussian probability contours for each class
    3. Decision boundary implied by the contours
    """
    print("\n" + "=" * 80)
    print("STEP 7: ADVANCED VISUALIZATION")
    print("=" * 80)
    
    # Set up the figure with a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate class statistics in 2D space
    class_stats = {}
    for c in [0, 1]:
        X_c = X_test_pca[y_test == c]
        class_stats[c] = {
            'mean': np.mean(X_c, axis=0),
            'cov': np.cov(X_c.T)
        }
    
    # Create meshgrid for contour plot
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Calculate Gaussian PDF for each class on the grid
    colors = {0: 'Reds', 1: 'Blues'}
    contour_colors = {0: 'darkred', 1: 'darkblue'}
    
    for c in [0, 1]:
        pdf_values = multivariate_gaussian_pdf(grid_points, 
                                                class_stats[c]['mean'], 
                                                class_stats[c]['cov'])
        pdf_grid = pdf_values.reshape(xx.shape)
        
        # Draw filled contours (light background)
        ax.contourf(xx, yy, pdf_grid, levels=10, cmap=colors[c], alpha=0.2)
        
        # Draw contour lines (probability density boundaries)
        contours = ax.contour(xx, yy, pdf_grid, levels=6, 
                              colors=contour_colors[c], alpha=0.7, linewidths=1.5)
    
    # Scatter plot of test data
    scatter_colors = {0: 'red', 1: 'blue'}
    markers = {0: 'o', 1: 's'}
    
    for c in [0, 1]:
        mask = y_test == c
        ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   c=scatter_colors[c], marker=markers[c], s=80,
                   edgecolors='white', linewidth=1.5,
                   label=f'{target_names[c]} (n={np.sum(mask)})',
                   alpha=0.8, zorder=5)
    
    # Mark class means
    for c in [0, 1]:
        ax.scatter(class_stats[c]['mean'][0], class_stats[c]['mean'][1],
                   c=scatter_colors[c], marker='*', s=400,
                   edgecolors='black', linewidth=2,
                   label=f'{target_names[c]} Mean', zorder=6)
    
    # Formatting
    ax.set_xlabel('Principal Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Principal Component 2', fontsize=14, fontweight='bold')
    ax.set_title('Breast Cancer Classification: 2D PCA Projection\nwith Gaussian Probability Contours',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text annotation explaining the visualization
    textstr = ('Contour lines represent probability density\n'
               'of the Gaussian distribution for each class.\n'
               'Stars (★) mark class centroids.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('breast_cancer_pca_visualization.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'breast_cancer_pca_visualization.png'")
    plt.show()
    
    return fig


# ================================================================================
# STEP 6: COMPARISON REPORT
# ================================================================================

def print_comparison_report(metrics_30d, metrics_2d, variance_preserved):
    """
    Print a formatted comparison table of model performance.
    """
    print("\n" + "=" * 80)
    print("STEP 6: COMPARISON REPORT")
    print("=" * 80)
    
    print("\n" + "+" + "-" * 78 + "+")
    print("|" + " " * 20 + "PERFORMANCE COMPARISON TABLE" + " " * 30 + "|")
    print("+" + "-" * 78 + "+")
    print(f"| {'Metric':<20} | {'Model A (30 Features)':<25} | {'Model B (2 Features PCA)':<25} |")
    print("+" + "-" * 78 + "+")
    print(f"| {'Accuracy':<20} | {metrics_30d['accuracy']*100:>24.2f}% | {metrics_2d['accuracy']*100:>24.2f}% |")
    print(f"| {'Precision':<20} | {metrics_30d['precision']:>25.4f} | {metrics_2d['precision']:>25.4f} |")
    print(f"| {'Recall':<20} | {metrics_30d['recall']:>25.4f} | {metrics_2d['recall']:>25.4f} |")
    print(f"| {'F1-Score':<20} | {metrics_30d['f1']:>25.4f} | {metrics_2d['f1']:>25.4f} |")
    print("+" + "-" * 78 + "+")
    print(f"| {'Dimensions':<20} | {'30':>25} | {'2':>25} |")
    print(f"| {'Variance Preserved':<20} | {'100.00%':>25} | {variance_preserved*100:>24.2f}% |")
    print("+" + "-" * 78 + "+")
    
    # Analysis
    accuracy_diff = (metrics_30d['accuracy'] - metrics_2d['accuracy']) * 100
    
    print("\n--- Analysis & Trade-off Discussion ---")
    print("\n1. DIMENSIONALITY REDUCTION IMPACT:")
    print(f"   • Reduced features from 30 to 2 ({(2/30)*100:.1f}% of original)")
    print(f"   • Preserved {variance_preserved*100:.2f}% of total variance")
    
    print("\n2. PERFORMANCE COMPARISON:")
    if accuracy_diff > 0:
        print(f"   • Accuracy decreased by {accuracy_diff:.2f}%")
    elif accuracy_diff < 0:
        print(f"   • Accuracy increased by {-accuracy_diff:.2f}%")
    else:
        print(f"   • Accuracy remained unchanged")
    
    print("\n3. TRADE-OFF ANALYSIS:")
    print("   • PROS of PCA reduction:")
    print("     - Significant reduction in computational complexity")
    print("     - Enables 2D visualization of high-dimensional data")
    print("     - Removes multicollinearity between features")
    print("     - Reduces risk of overfitting")
    print("   • CONS of PCA reduction:")
    print("     - Some information loss (variance not captured)")
    print("     - Principal components are less interpretable than original features")
    print("     - May lose discriminative features for specific tasks")
    
    print("\n4. CONCLUSION:")
    if accuracy_diff < 5:
        print("   The PCA-reduced model maintains competitive performance while")
        print("   dramatically reducing dimensionality. This is an excellent trade-off")
        print("   for applications requiring visualization or faster computation.")
    else:
        print("   While there is some performance loss, the 2D representation")
        print("   provides valuable insights through visualization and may be")
        print("   preferred when interpretability is prioritized over accuracy.")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """
    Main function to execute the complete ML pipeline.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "MACHINE LEARNING ASSIGNMENT" + " " * 36 + "║")
    print("║" + " " * 10 + "Gaussian Naive Bayes & PCA from Scratch" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # =========================================================================
    # STEP 1: Data Preparation
    # =========================================================================
    X, y, feature_names, target_names = load_and_prepare_data()
    
    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData Split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    # Standardize features
    X_train_std, X_test_std, mu, sigma = standardize_features(X_train, X_test)
    
    # =========================================================================
    # STEP 2: Gaussian Naive Bayes on 30 Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: GAUSSIAN NAIVE BAYES (30 FEATURES)")
    print("=" * 80)
    
    # Create and train the model
    model_30d = GaussianNB_Scratch()
    model_30d.fit(X_train_std, y_train)
    
    # Make predictions
    y_pred_30d = model_30d.predict(X_test_std)
    
    # Calculate and print metrics
    metrics_30d = print_metrics(y_test, y_pred_30d, "Gaussian NB (30 Features)")
    
    # =========================================================================
    # STEP 3: Covariance & Eigen Analysis
    # =========================================================================
    cov_matrix, eigenvalues, eigenvectors, explained_var_ratio = compute_covariance_and_eigen(X_train_std)
    
    # =========================================================================
    # STEP 4: PCA Dimensionality Reduction
    # =========================================================================
    X_train_pca, X_test_pca, W_top, variance_preserved = pca_reduce_dimensions(
        X_train_std, X_test_std, eigenvectors, explained_var_ratio, n_components=2
    )
    
    # =========================================================================
    # STEP 5: Re-Classification on 2D Reduced Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: GAUSSIAN NAIVE BAYES (2 PCA FEATURES)")
    print("=" * 80)
    
    # Create and train new model on 2D data
    model_2d = GaussianNB_Scratch()
    model_2d.fit(X_train_pca, y_train)
    
    # Make predictions
    y_pred_2d = model_2d.predict(X_test_pca)
    
    # Calculate and print metrics
    metrics_2d = print_metrics(y_test, y_pred_2d, "Gaussian NB (2 PCA Features)")
    
    # =========================================================================
    # STEP 6: Comparison Report
    # =========================================================================
    print_comparison_report(metrics_30d, metrics_2d, variance_preserved)
    
    # =========================================================================
    # STEP 7: Advanced Visualization
    # =========================================================================
    create_visualization(X_test_pca, y_test, model_2d, target_names)
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print("\nAll steps have been executed successfully!")
    print("Check 'breast_cancer_pca_visualization.png' for the saved visualization.")


# Run the main function
if __name__ == "__main__":
    main()
