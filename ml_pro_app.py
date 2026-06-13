
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="ML Classifier - Academic",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .prediction-malignant {
        background: linear-gradient(135deg, #ff6b6b 0%, #c0392b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(192, 57, 43, 0.4);
        margin: 1rem 0;
    }
    .prediction-benign {
        background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(72, 198, 239, 0.4);
        margin: 1rem 0;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

class GaussianNaiveBayes:
    """
    Professional Gaussian Naive Bayes Classifier from Scratch.
    
    Mathematical Foundation:
    ========================
    Bayes' Theorem: P(y|X) = P(X|y) × P(y) / P(X)
    
    For classification: ŷ = argmax_y [P(y) × ∏ᵢ P(xᵢ|y)]
    
    Gaussian PDF: P(xᵢ|y) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))
    
    Using log-probabilities to prevent numerical underflow:
    log P(xᵢ|y) = -½log(2πσ²) - (xᵢ-μ)²/(2σ²)
    """
    
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = {}
        self.means_ = {}
        self.variances_ = {}
        self.n_features_ = None
    
    def fit(self, X, y):
        """Fit Gaussian Naive Bayes according to X, y."""
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = len(y)
        
        for cls in self.classes_:
            X_cls = X[y == cls]
            
            # Prior probability P(y)
            self.priors_[cls] = len(X_cls) / n_samples
            
            # Mean μ for each feature
            self.means_[cls] = np.mean(X_cls, axis=0)
            
            # Variance σ² with smoothing to prevent division by zero
            self.variances_[cls] = np.var(X_cls, axis=0) + self.var_smoothing
        
        return self
    
    def _log_gaussian_pdf(self, x, mean, var):
        """Calculate log of Gaussian PDF."""
        return -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
    
    def _log_posterior(self, x):
        """Calculate log posterior for each class."""
        posteriors = {}
        for cls in self.classes_:
            log_prior = np.log(self.priors_[cls])
            log_likelihood = np.sum(self._log_gaussian_pdf(x, self.means_[cls], self.variances_[cls]))
            posteriors[cls] = log_prior + log_likelihood
        return posteriors
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([max(self._log_posterior(x), key=self._log_posterior(x).get) for x in X])
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        probas = []
        for x in X:
            log_post = self._log_posterior(x)
            log_vals = np.array([log_post[cls] for cls in self.classes_])
            # Softmax with numerical stability
            log_vals -= np.max(log_vals)
            probs = np.exp(log_vals)
            probs /= np.sum(probs)
            probas.append(probs)
        return np.array(probas)


class MetricsCalculator:
    """Calculate classification metrics from scratch."""
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Calculate confusion matrix elements."""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp, tn, fp, fn
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Accuracy = (TP + TN) / Total"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        """Precision = TP / (TP + FP)"""
        tp, tn, fp, fn = MetricsCalculator.confusion_matrix(y_true, y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def recall(y_true, y_pred):
        """Recall = TP / (TP + FN)"""
        tp, tn, fp, fn = MetricsCalculator.confusion_matrix(y_true, y_pred)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """F1 = 2 × (Precision × Recall) / (Precision + Recall)"""
        p = MetricsCalculator.precision(y_true, y_pred)
        r = MetricsCalculator.recall(y_true, y_pred)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @staticmethod
    def specificity(y_true, y_pred):
        """Specificity = TN / (TN + FP)"""
        tp, tn, fp, fn = MetricsCalculator.confusion_matrix(y_true, y_pred)
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    @staticmethod
    def all_metrics(y_true, y_pred):
        """Return all metrics as dictionary."""
        return {
            'Accuracy': MetricsCalculator.accuracy(y_true, y_pred),
            'Precision': MetricsCalculator.precision(y_true, y_pred),
            'Recall': MetricsCalculator.recall(y_true, y_pred),
            'F1-Score': MetricsCalculator.f1_score(y_true, y_pred),
            'Specificity': MetricsCalculator.specificity(y_true, y_pred)
        }

class PCAFromScratch:
    """
    Principal Component Analysis from Scratch.
    
    Mathematical Foundation:
    ========================
    1. Standardize: X' = (X - μ) / σ
    2. Covariance Matrix: Σ = (1/(n-1)) × X'ᵀ × X'
    3. Eigen Decomposition: Σv = λv
    4. Sort by eigenvalues (descending)
    5. Project: X_pca = X' × W (top-k eigenvectors)
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.covariance_matrix_ = None
    
    def fit(self, X):
        """Fit PCA model."""
        # Step 1: Standardization
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-10
        X_std = (X - self.mean_) / self.std_
        
        # Step 2: Covariance Matrix
        self.covariance_matrix_ = np.cov(X_std.T)
        
        # Step 3: Eigen Decomposition
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix_)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Step 4: Sort by eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Select top components
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Explained variance ratio
        total_var = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        
        return self
    
    def transform(self, X):
        """Apply dimensionality reduction."""
        X_std = (X - self.mean_) / self.std_
        return X_std @ self.components_
    
    def fit_transform(self, X):
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


@st.cache_data
def load_data():
    """Load and prepare the Breast Cancer dataset."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': data.feature_names,
        'target_names': data.target_names
    }


@st.cache_resource
def train_pipeline(_data):
    """Complete training pipeline."""
    X_train, y_train = _data['X_train'], _data['y_train']
    X_test = _data['X_test']
    
    # PCA transformation
    pca = PCAFromScratch(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Standardize original data for 30D model
    mean_30d = np.mean(X_train, axis=0)
    std_30d = np.std(X_train, axis=0)
    std_30d[std_30d == 0] = 1e-10
    X_train_std = (X_train - mean_30d) / std_30d
    X_test_std = (X_test - mean_30d) / std_30d
    
    # Train models
    model_30d = GaussianNaiveBayes()
    model_30d.fit(X_train_std, y_train)
    
    model_2d = GaussianNaiveBayes()
    model_2d.fit(X_train_pca, y_train)
    
    # Class statistics for visualization
    class_stats = {}
    for cls in [0, 1]:
        X_cls = X_train_pca[y_train == cls]
        class_stats[cls] = {
            'mean': np.mean(X_cls, axis=0),
            'cov': np.cov(X_cls.T)
        }
    
    # Per-class covariance matrices (original features)
    cov_per_class = {}
    for cls in [0, 1]:
        X_cls = X_train_std[y_train == cls]
        cov_per_class[cls] = np.cov(X_cls.T)
    
    return {
        'pca': pca,
        'model_30d': model_30d,
        'model_2d': model_2d,
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'X_train_std': X_train_std,
        'X_test_std': X_test_std,
        'class_stats': class_stats,
        'cov_per_class': cov_per_class
    }

def create_professional_plot(pipeline, data, user_point=None, prediction=None, probabilities=None):
    """Create publication-quality visualization."""
    
    # Set up figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    
    X_train_pca = pipeline['X_train_pca']
    X_test_pca = pipeline['X_test_pca']
    y_train = data['y_train']
    y_test = data['y_test']
    class_stats = pipeline['class_stats']
    target_names = data['target_names']
    
    # Create meshgrid for decision boundary
    margin = 1.5
    x_min = min(X_train_pca[:, 0].min(), X_test_pca[:, 0].min()) - margin
    x_max = max(X_train_pca[:, 0].max(), X_test_pca[:, 0].max()) + margin
    y_min = min(X_train_pca[:, 1].min(), X_test_pca[:, 1].min()) - margin
    y_max = max(X_train_pca[:, 1].max(), X_test_pca[:, 1].max()) + margin
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                          np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Decision boundary using model predictions
    Z_pred = pipeline['model_2d'].predict(grid).reshape(xx.shape)
    
    # Probability contours for smooth visualization
    Z_proba = pipeline['model_2d'].predict_proba(grid)[:, 1].reshape(xx.shape)
    
    # Create custom colormaps
    colors_red = ['#0f0f23', '#3d1a1a', '#6b2020', '#9a2626', '#c93030', '#ff4444']
    colors_blue = ['#0f0f23', '#1a2a3d', '#203050', '#264080', '#3060b0', '#4488ff']
    
    cmap_decision = LinearSegmentedColormap.from_list('decision', ['#ff4444', '#4488ff'], N=256)
    
    # Draw probability gradient background
    ax.contourf(xx, yy, Z_proba, levels=50, cmap=cmap_decision, alpha=0.3)
    
    # Draw decision boundary (sharp line where P = 0.5)
    ax.contour(xx, yy, Z_proba, levels=[0.5], colors=['#00ff88'], linewidths=3, linestyles='-')
    
    # Draw Mahalanobis confidence ellipses
    colors = {0: '#ff4444', 1: '#4488ff'}
    
    for cls in [0, 1]:
        mean = class_stats[cls]['mean']
        cov = class_stats[cls]['cov']
        
        # Eigendecomposition for ellipse orientation
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Angle of rotation
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Draw 1σ, 2σ, 3σ ellipses
        for n_std, alpha in [(3, 0.15), (2, 0.25), (1, 0.4)]:
            width = 2 * n_std * np.sqrt(eigenvalues[0])
            height = 2 * n_std * np.sqrt(eigenvalues[1])
            
            ellipse = Ellipse(
                xy=mean, width=width, height=height, angle=angle,
                facecolor=colors[cls], edgecolor=colors[cls],
                alpha=alpha, linewidth=2
            )
            ax.add_patch(ellipse)
        
        # Draw principal axes (arrows)
        for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            arrow_length = np.sqrt(eigval) * 2
            ax.annotate('', 
                xy=mean + eigvec * arrow_length,
                xytext=mean,
                arrowprops=dict(arrowstyle='-|>', color=colors[cls], lw=3, 
                               mutation_scale=20, shrinkA=0, shrinkB=0))
            ax.annotate('', 
                xy=mean - eigvec * arrow_length,
                xytext=mean,
                arrowprops=dict(arrowstyle='-|>', color=colors[cls], lw=3,
                               mutation_scale=20, shrinkA=0, shrinkB=0))
    
    # Plot data points
    markers = {0: 'o', 1: 's'}
    for cls in [0, 1]:
        mask = y_test == cls
        ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   c=colors[cls], marker=markers[cls], s=100, alpha=0.85,
                   edgecolors='white', linewidths=1.5,
                   label=f'{target_names[cls].capitalize()}',
                   zorder=5)
    
    # Plot user point if provided
    if user_point is not None:
        # Outer glow effect
        ax.scatter(user_point[0], user_point[1], c='#ffff00', s=800,
                   marker='*', alpha=0.3, zorder=9)
        ax.scatter(user_point[0], user_point[1], c='#ffff00', s=500,
                   marker='*', alpha=0.5, zorder=9)
        # Main star
        ax.scatter(user_point[0], user_point[1], c='black', s=350,
                   marker='*', edgecolors='#ffff00', linewidths=3,
                   zorder=10, label='User Point')
    
    # Styling
    ax.set_xlabel('Principal Component 1 (PC1)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Principal Component 2 (PC2)', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Mahalanobis Decision Boundary + Gaussian Ellipses', 
                 fontsize=18, fontweight='bold', color='white', pad=20)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=12, fancybox=True, 
                       framealpha=0.9, edgecolor='white')
    legend.get_frame().set_facecolor('#1a1a2e')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Grid
    ax.grid(True, alpha=0.2, color='white', linestyle='--')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Tick colors
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">Breast Cancer Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Gaussian Naive Bayes with Principal Component Analysis | Implementation from Scratch</p>', unsafe_allow_html=True)
    
    # Load data and train models
    data = load_data()
    pipeline = train_pipeline(data)
    
    # Calculate metrics
    y_pred_30d = pipeline['model_30d'].predict(pipeline['X_test_std'])
    y_pred_2d = pipeline['model_2d'].predict(pipeline['X_test_pca'])
    
    metrics_30d = MetricsCalculator.all_metrics(data['y_test'], y_pred_30d)
    metrics_2d = MetricsCalculator.all_metrics(data['y_test'], y_pred_2d)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Test Point Input")
        st.markdown("---")
        
        # Get data range for sliders
        X_pca = pipeline['X_train_pca']
        pc1_range = (float(X_pca[:, 0].min() - 1), float(X_pca[:, 0].max() + 1))
        pc2_range = (float(X_pca[:, 1].min() - 1), float(X_pca[:, 1].max() + 1))
        
        st.markdown("#### Principal Component 1")
        pc1 = st.slider("PC1", pc1_range[0], pc1_range[1], 0.0, 0.05, label_visibility="collapsed")
        
        st.markdown("#### Principal Component 2")
        pc2 = st.slider("PC2", pc2_range[0], pc2_range[1], 0.0, 0.05, label_visibility="collapsed")
        
        st.markdown("---")
        
        # Make prediction
        user_point = np.array([pc1, pc2])
        prediction = pipeline['model_2d'].predict(user_point.reshape(1, -1))[0]
        probabilities = pipeline['model_2d'].predict_proba(user_point.reshape(1, -1))[0]
        
        pred_class = data['target_names'][prediction].capitalize()
        confidence = probabilities[prediction] * 100
        
        st.markdown("### Prediction Result")
        
        if prediction == 0:
            st.markdown(f"""
            <div class="prediction-malignant">
                MALIGNANT<br>
                <small>Confidence: {confidence:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-benign">
                BENIGN<br>
                <small>Confidence: {confidence:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Class Probabilities")
        st.markdown(f"**Malignant:** {probabilities[0]*100:.1f}%")
        st.progress(float(probabilities[0]))
        st.markdown(f"**Benign:** {probabilities[1]*100:.1f}%")
        st.progress(float(probabilities[1]))
    
    # Main content
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        # Create and display plot
        fig = create_professional_plot(pipeline, data, user_point, prediction, probabilities)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### Model Performance Comparison")
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Metric': list(metrics_30d.keys()),
            '30 Features': [f"{v*100:.2f}%" for v in metrics_30d.values()],
            '2 PCA': [f"{v*100:.2f}%" for v in metrics_2d.values()]
        })
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Explained Variance Ratio")
        
        pca = pipeline['pca']
        var1, var2 = pca.explained_variance_ratio_[:2]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("PC1", f"{var1*100:.1f}%")
        with col_b:
            st.metric("PC2", f"{var2*100:.1f}%")
        
        st.metric("Total (2 PCs)", f"{(var1+var2)*100:.1f}%", 
                  delta=f"-{(1-var1-var2)*100:.1f}% info loss")
        
        st.markdown("---")
        st.markdown("### Eigenvalue Spectrum")
        eigen_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(5)],
            'λ': [f"{pca.eigenvalues_[i]:.3f}" for i in range(5)],
            'Var%': [f"{pca.explained_variance_ratio_[i]*100:.1f}%" for i in range(5)]
        })
        st.dataframe(eigen_df, hide_index=True, use_container_width=True)
    
    # Expandable sections
    st.markdown("---")
    
    with st.expander("Covariance Matrix Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Overall Covariance Matrix (5×5)")
            cov_display = pd.DataFrame(
                pipeline['pca'].covariance_matrix_[:5, :5].round(3),
                columns=[f"F{i+1}" for i in range(5)],
                index=[f"F{i+1}" for i in range(5)]
            )
            st.dataframe(cov_display)
        
        with col2:
            st.markdown("#### Per-Class Covariance (5×5)")
            tab1, tab2 = st.tabs(["Malignant", "Benign"])
            
            with tab1:
                cov0 = pd.DataFrame(pipeline['cov_per_class'][0][:5, :5].round(3))
                st.dataframe(cov0)
            
            with tab2:
                cov1 = pd.DataFrame(pipeline['cov_per_class'][1][:5, :5].round(3))
                st.dataframe(cov1)
    
    with st.expander("Feature Selection Rationale", expanded=False):
        st.markdown("""
        ### Why Select Top 2 Principal Components?
        
        **1. Explained Variance Analysis:**
        - PC1 captures the direction of maximum variance in the data
        - PC2 captures the second-highest variance orthogonal to PC1
        - Together, they explain ~63% of total variance
        
        **2. Elbow Method:**
        The eigenvalue spectrum shows a clear "elbow" after the first few components,
        indicating that additional components contribute diminishing returns.
        
        **3. Dimensionality Reduction Trade-off:**
        | Aspect | 30 Features | 2 PCA Features |
        |--------|-------------|----------------|
        | Accuracy | ~93% | ~91% |
        | Interpretability | Low | High (2D visualization) |
        | Computation | O(30) | O(2) |
        | Overfitting Risk | Higher | Lower |
        
        **4. Visualization Benefit:**
        2D projection enables intuitive visualization of:
        - Decision boundaries
        - Class separability
        - Probability distributions (Gaussian ellipses)
        """)
        
        # Scree plot
        fig_scree, ax_scree = plt.subplots(figsize=(10, 4))
        fig_scree.patch.set_facecolor('#0f0f23')
        ax_scree.set_facecolor('#0f0f23')
        
        n_show = 10
        x_vals = np.arange(1, n_show + 1)
        bars = ax_scree.bar(x_vals, pipeline['pca'].explained_variance_ratio_[:n_show] * 100,
                           color='#4488ff', alpha=0.7, edgecolor='white')
        line = ax_scree.plot(x_vals, np.cumsum(pipeline['pca'].explained_variance_ratio_[:n_show]) * 100,
                            'o-', color='#ff4444', linewidth=2, markersize=8, label='Cumulative')
        ax_scree.axhline(y=80, color='#00ff88', linestyle='--', linewidth=2, label='80% Threshold')
        
        ax_scree.set_xlabel('Principal Component', color='white', fontsize=12)
        ax_scree.set_ylabel('Variance Explained (%)', color='white', fontsize=12)
        ax_scree.set_title('Scree Plot', color='white', fontsize=14, fontweight='bold')
        ax_scree.tick_params(colors='white')
        ax_scree.legend(facecolor='#1a1a2e', labelcolor='white')
        ax_scree.grid(True, alpha=0.2, color='white')
        
        for spine in ax_scree.spines.values():
            spine.set_color('white')
        
        st.pyplot(fig_scree)
        plt.close()
    
    with st.expander("Mathematical Foundation", expanded=False):
        st.markdown("""
        ### Gaussian Naive Bayes
        
        **Bayes' Theorem:**
        $$P(y|X) = \\frac{P(X|y) \\cdot P(y)}{P(X)}$$
        
        **Gaussian Probability Density Function:**
        $$P(x_i|y) = \\frac{1}{\\sqrt{2\\pi\\sigma_y^2}} \\exp\\left(-\\frac{(x_i - \\mu_y)^2}{2\\sigma_y^2}\\right)$$
        
        **Log-Likelihood (for numerical stability):**
        $$\\log P(x_i|y) = -\\frac{1}{2}\\log(2\\pi\\sigma_y^2) - \\frac{(x_i - \\mu_y)^2}{2\\sigma_y^2}$$
        
        ---
        
        ### Principal Component Analysis
        
        **Z-Score Standardization:**
        $$x' = \\frac{x - \\mu}{\\sigma}$$
        
        **Covariance Matrix:**
        $$\\Sigma = \\frac{1}{n-1} X'^T X'$$
        
        **Eigenvalue Decomposition:**
        $$\\Sigma v = \\lambda v$$
        
        **PCA Projection:**
        $$X_{reduced} = X' \\cdot W_{top-k}$$
        
        ---
        
        ### Mahalanobis Distance
        $$D_M(x) = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}$$
        
        The ellipses in the visualization represent constant Mahalanobis distance contours (1σ, 2σ, 3σ).
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #5A6C7D; padding: 1rem;'>
        <p>Machine Learning Implementation | Gaussian Naive Bayes and PCA from Scratch</p>
        <p><small>All algorithms implemented using NumPy only (no sklearn models)</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
