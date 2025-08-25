from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_data_filling(df):
    # Convert DataFrame to a boolean matrix: True for non-NaN values, False for NaN values
    data_filling = ~df.isna()
    
    # Plotting
    plt.figure(figsize=(20, 20))
    plt.imshow(data_filling, cmap='Greens', interpolation='none', aspect=6./35)
    plt.title('Data Filling Plot')
    plt.xlabel('Features')
    plt.ylabel('Data sample ID')
    # plt.colorbar(label='Data Presence (1: non-NaN, 0: NaN)', ticks=[0, 1])
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, fontsize=7, rotation=90)
    # plt.grid(axis='x', color='black', linestyle='-', linewidth=0.5)
    for x in np.arange(-0.5, len(df.columns.values), 1):
        plt.axvline(x=x, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    
def analyze_outliers(data, outlier_threshold=5, nan_mask=None, verbose=True):
    """
    Analyze outliers in scaled imputed data and visualize the results.

    Parameters:
        data (pd.DataFrame): Scaled and imputed dataframe to analyze
        outlier_threshold (float): Number of standard deviations to consider as outlier threshold
        verbose (bool): Whether to print statistics and generate plots

    Returns:
        np.ndarray: Boolean mask indicating outlier values that should be imputed
    """
    # Get NaN mask from the original data
    if nan_mask is None:
        nan_mask = data.isna().values

    # Check for extreme values beyond the threshold
    extreme_values = np.abs(data) > outlier_threshold
    outlier_counts = extreme_values.sum(axis=0)
    outliers = np.array(nan_mask) & np.array(extreme_values)

    if verbose:
        outlier_percentages = pd.Series(
            (outlier_counts / len(data)) * 100,
            index=data.columns
        )

        # Display columns with high outlier percentages
        high_outlier_cols = outlier_percentages[outlier_percentages > 5].sort_values(ascending=False)
        if len(high_outlier_cols) > 0:
            print(f"Columns with >5% outliers (beyond ±{outlier_threshold} std):")
            for col, pct in high_outlier_cols.items():
                print(f"  {col}: {pct:.1f}%")

        # Check overall statistics
        scaled_data = np.array(data)
        
        print(f"\nOverall statistics of scaled imputed data:")
        print(f"  Min value: {scaled_data.min():.2f}")
        print(f"  Max value: {scaled_data.max():.2f}")
        print(f"  Mean: {scaled_data.mean():.4f}")
        print(f"  Std: {scaled_data.std():.4f}")

        print(f"Fraction of outliers in the imputed dataset: {np.sum(outliers) / outliers.size * 100:.1f}%")

        # Visualize distribution of scaled values
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

        # Histogram of all scaled values
        ax1.hist(scaled_data.flatten(), bins=100, alpha=0.7, edgecolor='black')
        ax1.axvline(x=-outlier_threshold, color='r', linestyle='--', label=f'±{outlier_threshold}σ threshold')
        ax1.axvline(x=outlier_threshold, color='r', linestyle='--')
        ax1.set_xlabel('Scaled Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Scaled Imputed Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot per feature
        sample_size = scaled_data.shape[1]
        sample_indices = np.arange(sample_size)
        ax2.boxplot(scaled_data[:, sample_indices], labels=[data.columns[i] for i in sample_indices])
        ax2.axhline(y=-outlier_threshold, color='r', linestyle='--', alpha=0.5, label=f'±{outlier_threshold}σ')
        ax2.axhline(y=outlier_threshold, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Scaled Value')
        ax2.set_title(f'Box Plot of all Features')
        ax2.tick_params(axis='x', rotation=90, labelsize=5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    return outliers


def VIF_contributors(df, target_feature):
    import statsmodels.api as sm
    # Create a copy to avoid modifying original
    df_copy = df.copy()

    # Drop the target feature and remove NaN values
    X = df_copy.drop(columns=[target_feature]).dropna()
    y = df_copy[target_feature].dropna()

    # Find common indices between X and y
    common_idx = X.index.intersection(y.index)

    # Align indexes using common indices
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Reset indices to avoid KeyError
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # Fit linear regression
    model = sm.OLS(y, sm.add_constant(X)).fit()

    print(model.summary())  # shows coefficients, R^2, etc.
    return model.params.sort_values(key=abs, ascending=False)


def plot_correlation_heatmap(df, method='pearson', verbose=True):
    """
    Plot correlation heatmap for the dataframe's numeric columns.

    Args:
        df (pd.DataFrame): DataFrame with numeric columns to analyze
        method (str): Correlation method ('pearson', 'spearman', or 'kendall')
        figsize (tuple): Figure size as (width, height)
        cmap (str): Colormap for the heatmap
        annot (bool): Whether to annotate cells with correlation values
        fontsize (int): Font size for axis labels
        title (str): Title for the plot
        verbose (bool): Whether to display the plot

    Returns:
        pd.DataFrame: Correlation matrix
    """
    corr = df.corr(numeric_only=True, method=method)
    
    if verbose:
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, xticklabels=corr.columns, yticklabels=corr.columns)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    return corr


def filter_by_correlation(corr: pd.DataFrame, threshold: float = 0.9, verbose=True) -> list[str]:
    """
    Find highly correlated features to drop based on correlation threshold.

    Args:
        corr: Correlation matrix DataFrame
        threshold: Correlation threshold above which features are considered highly correlated

    Returns:
        List of column names that are highly correlated and should be dropped
    """
    corr_matrix = corr.abs()
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if len(to_drop) > 0 and verbose:
        print("Highly correlated features to drop:", to_drop)

    return to_drop


def calculate_VIF(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for features in a dataframe.

    Args:
        df: Pandas DataFrame with numeric features

    Returns:
        DataFrame with feature names and their VIF values
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Select only numeric columns and drop rows with NaN values
    X = df.select_dtypes(include=['float64', 'int64']).dropna()

    # Create a DataFrame to store the results
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Sort by VIF:
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    return vif_data


def analyze_NaN_distribution(dataframe):
    """
    Analyze the distribution of NaN values across columns in a dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe to analyze

    Returns:
        pd.Series: Series containing percentage of NaN values for each column (sorted)
    """
    # Calculate the relative NaN count for all columns in the dataset
    nan_percentage = (dataframe.isna().sum() / len(dataframe) * 100).sort_values(ascending=False)

    # Display the results
    print("Percentage of NaN values in each column:")
    for col, pct in nan_percentage.items():
        print(f"{col}: {pct:.2f}%")

    print(f"\nColumn with highest NaN percentage: {nan_percentage.index[0]} ({nan_percentage.iloc[0]:.2f}%)")
    print(f"Column with lowest NaN percentage: {nan_percentage.index[-1]} ({nan_percentage.iloc[-1]:.2f}%)")

    return nan_percentage


# ---- X: (n_samples, n_features) after preprocessing; columns = feature names ----
# Optionally, provide a supervised target y (1D) to pick cluster reps by MI; else None.
def feature_corr_clustering(X: pd.DataFrame, y: np.ndarray | None = None,
                            corr_method: str = 'spearman',
                            prune_threshold: float = 0.95,
                            max_features_per_cluster: int = 1,
                            random_state: int = 0):
    """
    Returns:
      corr_df: DataFrame [p x p] with correlation
      order:   list of feature names in clustered order (for plotting)
      keep:    list of selected features after pruning (≥1 per cluster)
      clusters: dict cluster_id -> list of features
    """
    rng = np.random.default_rng(random_state)
    feats = X.columns.tolist()
    # 1) Correlation matrix (abs, to cluster by strength regardless of sign)
    corr = X.corr(method=corr_method).fillna(0.0)
    corr_abs = corr.abs()

    # 2) Distance for clustering
    # Ensure diagonal is 1 so distance 0 there; clip small negatives from num noise
    np.fill_diagonal(corr_abs.values, 1.0)
    dist = 1.0 - corr_abs.values
    # Force symmetry and zeros on diagonal
    dist = 0.5*(dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    # 3) Hierarchical clustering on condensed distance
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')  # ‘average’ is stable; ‘complete’ is stricter

    # 4) Determine clusters by threshold on correlation (equivalently distance)
    # If we want |rho| >= prune_threshold to be "same cluster", set t = 1 - prune_threshold
    t = 1.0 - prune_threshold
    cluster_ids = fcluster(Z, t=t, criterion='distance')

    # Group features
    clusters = {}
    for f, cid in zip(feats, cluster_ids):
        clusters.setdefault(cid, []).append(f)

    # 5) Choose representatives
    keep = []
    if y is not None:
        # compute MI of each feature with y (supervised tie-breaker)
        mi = mutual_info_regression(X.values, y, random_state=random_state)
        mi_map = dict(zip(feats, mi))
    else:
        mi_map = None

    for cid, members in clusters.items():
        if len(members) <= max_features_per_cluster:
            keep.extend(members)
            continue
        # Rank members: by MI if available, else by variance as proxy
        if mi_map is not None:
            ranked = sorted(members, key=lambda f: mi_map.get(f, 0.0), reverse=True)
        else:
            variances = X[members].var().to_dict()
            ranked = sorted(members, key=lambda f: variances[f], reverse=True)
        keep.extend(ranked[:max_features_per_cluster])

    # 6) Order features for plotting according to dendrogram leaves
    dend = dendrogram(Z, no_plot=True, labels=feats)
    order = dend['ivl']  # feature names in clustered order

    corr_df = corr.loc[order, order]
    return corr_df, order, keep, clusters


def plot_corr_heatmap(corr_df: pd.DataFrame, title: str = "Feature correlation (clustered)"):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(corr_df.values, interpolation='nearest', vmin=-1, vmax=1)
    plt.title(title)
    plt.xticks(ticks=np.arange(len(corr_df)), labels=corr_df.columns, rotation=90, fontsize=7)
    plt.yticks(ticks=np.arange(len(corr_df)), labels=corr_df.index, fontsize=7)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
