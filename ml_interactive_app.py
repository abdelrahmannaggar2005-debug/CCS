"""
================================================================================
Interactive Machine Learning Classifier with GUI
================================================================================
Breast Cancer Classification with Gaussian Naive Bayes & PCA
Interactive Streamlit Application with Real-time Visualization
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# ================================================================================
# GAUSSIAN NAIVE BAYES FROM SCRATCH
# ================================================================================

class GaussianNB_Scratch:
    """
    Gaussian Naive Bayes Classifier implemented from scratch.
    """
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.variances = {}
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9
        
        return self
    
    def _gaussian_pdf_log(self, x, mean, var):
        log_2pi = np.log(2 * np.pi)
        log_pdf = -0.5 * (log_2pi + np.log(var) + ((x - mean) ** 2) / var)
        return log_pdf
    
    def predict_single(self, x):
        log_posteriors = {}
        for c in self.classes:
            log_prior = np.log(self.class_priors[c])
            log_likelihood = np.sum(self._gaussian_pdf_log(x, self.means[c], self.variances[c]))
            log_posteriors[c] = log_prior + log_likelihood
        return max(log_posteriors, key=log_posteriors.get)
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
    
    def predict_proba_single(self, x):
        log_posteriors = []
        for c in self.classes:
            log_prior = np.log(self.class_priors[c])
            log_likelihood = np.sum(self._gaussian_pdf_log(x, self.means[c], self.variances[c]))
            log_posteriors.append(log_prior + log_likelihood)
        
        log_posteriors = np.array(log_posteriors)
        log_posteriors -= np.max(log_posteriors)
        posteriors = np.exp(log_posteriors)
        posteriors /= np.sum(posteriors)
        return posteriors


# ================================================================================
# METRICS IMPLEMENTATION
# ================================================================================

def calculate_metrics(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ================================================================================
# DATA PREPARATION & MODEL TRAINING (CACHED)
# ================================================================================

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset with PCA transformation."""
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma = np.where(sigma == 0, 1e-10, sigma)
    
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    
    # Covariance and Eigen analysis
    cov_matrix = np.cov(X_train_std.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # PCA - reduce to 2D
    W_top = eigenvectors_sorted[:, :2]
    X_train_pca = X_train_std @ W_top
    X_test_pca = X_test_std @ W_top
    
    # Calculate explained variance
    total_variance = np.sum(eigenvalues_sorted)
    explained_variance_ratio = eigenvalues_sorted / total_variance
    
    # Covariance matrices per class (for original features)
    cov_class_0 = np.cov(X_train_std[y_train == 0].T)
    cov_class_1 = np.cov(X_train_std[y_train == 1].T)
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'X_train_std': X_train_std, 'X_test_std': X_test_std,
        'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca,
        'y_train': y_train, 'y_test': y_test,
        'mu': mu, 'sigma': sigma, 'W_top': W_top,
        'eigenvalues': eigenvalues_sorted,
        'eigenvectors': eigenvectors_sorted,
        'explained_variance': explained_variance_ratio,
        'feature_names': feature_names,
        'target_names': target_names,
        'cov_matrix': cov_matrix,
        'cov_class_0': cov_class_0,
        'cov_class_1': cov_class_1
    }


@st.cache_resource
def train_models(_data):
    """Train both 30D and 2D models."""
    # Model on 30 features
    model_30d = GaussianNB_Scratch()
    model_30d.fit(_data['X_train_std'], _data['y_train'])
    
    # Model on 2 PCA features
    model_2d = GaussianNB_Scratch()
    model_2d.fit(_data['X_train_pca'], _data['y_train'])
    
    return model_30d, model_2d


def confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Draw confidence ellipse for a 2D Gaussian distribution.
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                       facecolor=facecolor, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def create_interactive_plot(data, model_2d, user_point=None, prediction=None):
    """Create the interactive visualization plot with Decision Boundary."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X_test_pca = data['X_test_pca']
    y_test = data['y_test']
    target_names = data['target_names']
    
    # Calculate class statistics in 2D
    class_stats = {}
    for c in [0, 1]:
        X_c = data['X_train_pca'][data['y_train'] == c]
        class_stats[c] = {
            'mean': np.mean(X_c, axis=0),
            'cov': np.cov(X_c.T)
        }
    
    # Create meshgrid for contours and decision boundary
    x_min, x_max = X_test_pca[:, 0].min() - 2, X_test_pca[:, 0].max() + 2
    y_min, y_max = X_test_pca[:, 1].min() - 2, X_test_pca[:, 1].max() + 2
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Calculate decision boundary using the model
    Z = model_2d.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Draw decision boundary as a contour line
    ax.contour(xx, yy, Z, levels=[0.5], colors=['darkgreen'], linewidths=3, linestyles='-')
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#ffcccc', '#ccccff'], alpha=0.3)
    
    # Colors
    colors = {0: ('red', 'Reds'), 1: ('blue', 'Blues')}
    
    # Draw confidence ellipses for each class
    for c in [0, 1]:
        mean = class_stats[c]['mean']
        cov = class_stats[c]['cov']
        
        # Draw multiple ellipses (1, 2, 3 standard deviations)
        for n_std, alpha in [(1, 0.4), (2, 0.25), (3, 0.1)]:
            confidence_ellipse(mean, cov, ax, n_std=n_std,
                              facecolor=colors[c][0], alpha=alpha,
                              edgecolor=colors[c][0], linewidth=2)
    
    # Scatter plot of test data
    for c in [0, 1]:
        mask = y_test == c
        ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   c=colors[c][0], s=60, alpha=0.8,
                   edgecolors='white', linewidth=1,
                   label=f'{target_names[c].capitalize()}')
    
    # Draw class means with arrows (principal directions)
    for c in [0, 1]:
        mean = class_stats[c]['mean']
        eigvals, eigvecs = np.linalg.eig(class_stats[c]['cov'])
        idx = np.argmax(eigvals)
        direction = eigvecs[:, idx] * np.sqrt(eigvals[idx]) * 1.5
        
        ax.annotate('', xy=(mean[0] + direction[0], mean[1] + direction[1]),
                    xytext=(mean[0], mean[1]),
                    arrowprops=dict(arrowstyle='-|>', color=colors[c][0], lw=3, mutation_scale=15))
        ax.annotate('', xy=(mean[0] - direction[0], mean[1] - direction[1]),
                    xytext=(mean[0], mean[1]),
                    arrowprops=dict(arrowstyle='-|>', color=colors[c][0], lw=3, mutation_scale=15))
    
    # Plot user point if provided
    if user_point is not None:
        color = 'red' if prediction == 0 else 'blue'
        ax.scatter(user_point[0], user_point[1], c='black', s=400,
                   marker='*', edgecolors='yellow', linewidth=2,
                   label='User Point', zorder=10)
    
    ax.set_xlabel('Principal Component 1 (PC1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Principal Component 2 (PC2)', fontsize=12, fontweight='bold')
    ax.set_title('Mahalanobis Decision Boundary + Ellipses', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig


# ================================================================================
# MAIN STREAMLIT APP
# ================================================================================

def main():
    st.title("🔬 Breast Cancer Classification")
    st.markdown("### Gaussian Naive Bayes with PCA Visualization")
    
    # Load data and train models
    data = load_and_prepare_data()
    model_30d, model_2d = train_models(data)
    
    # Calculate metrics for both models
    y_pred_30d = model_30d.predict(data['X_test_std'])
    y_pred_2d = model_2d.predict(data['X_test_pca'])
    
    metrics_30d = calculate_metrics(data['y_test'], y_pred_30d)
    metrics_2d = calculate_metrics(data['y_test'], y_pred_2d)
    
    # Sidebar for user input
    st.sidebar.markdown("## 🎯 Test Point")
    st.sidebar.markdown("Enter values for the 2 principal components:")
    
    # Get range for sliders
    pc1_min, pc1_max = float(data['X_test_pca'][:, 0].min() - 1), float(data['X_test_pca'][:, 0].max() + 1)
    pc2_min, pc2_max = float(data['X_test_pca'][:, 1].min() - 1), float(data['X_test_pca'][:, 1].max() + 1)
    
    pc1_value = st.sidebar.slider("PC1 Value", pc1_min, pc1_max, 0.0, 0.1)
    pc2_value = st.sidebar.slider("PC2 Value", pc2_min, pc2_max, 0.0, 0.1)
    
    # Make prediction
    user_point = np.array([pc1_value, pc2_value])
    prediction = model_2d.predict_single(user_point)
    probabilities = model_2d.predict_proba_single(user_point)
    
    # Display prediction
    pred_class = data['target_names'][prediction].capitalize()
    pred_color = "🔴" if prediction == 0 else "🔵"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Prediction")
    
    if prediction == 0:
        st.sidebar.error(f"{pred_color} **{pred_class}**")
    else:
        st.sidebar.success(f"{pred_color} **{pred_class}**")
    
    st.sidebar.markdown(f"**Confidence:**")
    st.sidebar.progress(float(probabilities[prediction]))
    st.sidebar.markdown(f"{probabilities[prediction]*100:.1f}%")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create and display plot
        fig = create_interactive_plot(data, model_2d, user_point, prediction)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### 📈 Model Comparison")
        
        # Metrics comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            '30 Features': [f"{metrics_30d['accuracy']*100:.2f}%",
                           f"{metrics_30d['precision']:.4f}",
                           f"{metrics_30d['recall']:.4f}",
                           f"{metrics_30d['f1']:.4f}"],
            '2 PCA': [f"{metrics_2d['accuracy']*100:.2f}%",
                      f"{metrics_2d['precision']:.4f}",
                      f"{metrics_2d['recall']:.4f}",
                      f"{metrics_2d['f1']:.4f}"]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 📊 Variance Explained")
        st.markdown(f"**PC1:** {data['explained_variance'][0]*100:.2f}%")
        st.progress(float(data['explained_variance'][0]))
        st.markdown(f"**PC2:** {data['explained_variance'][1]*100:.2f}%")
        st.progress(float(data['explained_variance'][1]))
        st.markdown(f"**Total:** {(data['explained_variance'][0]+data['explained_variance'][1])*100:.2f}%")
        
        st.markdown("### 🔑 Top 5 Eigenvalues")
        eigen_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(5)],
            'Eigenvalue': [f"{data['eigenvalues'][i]:.4f}" for i in range(5)],
            'Variance %': [f"{data['explained_variance'][i]*100:.2f}%" for i in range(5)]
        })
        st.dataframe(eigen_df, use_container_width=True, hide_index=True)
    
    # Expandable sections for detailed analysis
    with st.expander("📋 Covariance Matrix Analysis"):
        st.markdown("#### Covariance Matrix of Training Data (30x30)")
        st.markdown("*Showing first 5x5 elements:*")
        cov_df = pd.DataFrame(
            data['cov_matrix'][:5, :5],
            columns=[data['feature_names'][i][:15] for i in range(5)],
            index=[data['feature_names'][i][:15] for i in range(5)]
        )
        st.dataframe(cov_df)
        
        st.markdown("#### Covariance Matrix per Class (First 5x5)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Malignant (Class 0)**")
            cov0_df = pd.DataFrame(data['cov_class_0'][:5, :5].round(4))
            st.dataframe(cov0_df)
        with col_b:
            st.markdown("**Benign (Class 1)**")
            cov1_df = pd.DataFrame(data['cov_class_1'][:5, :5].round(4))
            st.dataframe(cov1_df)
    
    with st.expander("📊 Eigenvalue Analysis & Feature Selection"):
        st.markdown("#### Why Select Top 2 Principal Components?")
        st.markdown("""
        **Selection Criteria:**
        1. **Variance Explained:** PC1 and PC2 together explain ~63% of total variance
        2. **Elbow Method:** Eigenvalues drop significantly after the first few components
        3. **Visualization:** 2D allows for intuitive visual representation
        4. **Performance Trade-off:** Only ~2% accuracy loss for 93% dimension reduction
        """)
        
        # Scree plot
        fig_scree, ax_scree = plt.subplots(figsize=(8, 4))
        x_vals = range(1, 11)
        ax_scree.bar(x_vals, data['explained_variance'][:10] * 100, alpha=0.7, color='steelblue')
        ax_scree.plot(x_vals, np.cumsum(data['explained_variance'][:10]) * 100, 'ro-', label='Cumulative')
        ax_scree.axhline(y=80, color='g', linestyle='--', label='80% threshold')
        ax_scree.set_xlabel('Principal Component')
        ax_scree.set_ylabel('Variance Explained (%)')
        ax_scree.set_title('Scree Plot')
        ax_scree.legend()
        st.pyplot(fig_scree)
        plt.close()
    
    with st.expander("ℹ️ About the Algorithm"):
        st.markdown("""
        ### Gaussian Naive Bayes (Implemented from Scratch)
        
        **Bayes' Theorem:**
        $$P(y|X) = \\frac{P(X|y) \\cdot P(y)}{P(X)}$$
        
        **Gaussian PDF for each feature:**
        $$P(x_i|y) = \\frac{1}{\\sqrt{2\\pi\\sigma_y^2}} \\exp\\left(-\\frac{(x_i - \\mu_y)^2}{2\\sigma_y^2}\\right)$$
        
        **PCA Projection:**
        $$X_{reduced} = X_{original} \\cdot W_{top-k}$$
        
        **Z-score Standardization:**
        $$x' = \\frac{x - \\mu}{\\sigma}$$
        """)


if __name__ == "__main__":
    main()
