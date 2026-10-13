#%% ------------------- Phase bump predictor --------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc


# Extract phase bump (Z1, first element of LO_coefs) from fitted dataset
phase_bump = muse_fitted_df['LO_coefs'].apply(lambda x: x[0])
phase_bump.name = 'Phase bump'

# Combine telemetry features with phase bump target
bump_df = reduced_telemetry.join(phase_bump, how='inner')

# Binarize phase bump: above median => 1 (bump present)
bump_threshold = bump_df['Phase bump'].median()
bump_df['Phase bump'] = (bump_df['Phase bump'] > bump_threshold).astype(int)

bump_df = bump_df.select_dtypes(exclude=['object'])
bump_df = bump_df.select_dtypes(exclude=['bool'])

bump_df.replace([np.inf, -np.inf], np.nan, inplace=True)
bump_df = bump_df.map(lambda x: np.nan if pd.isna(x) or abs(x) > np.finfo(np.float64).max else x)
bump_df.dropna(inplace=True)


X = bump_df.drop('Phase bump', axis=1)
y = bump_df['Phase bump']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose a model
model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print("\nPrecision-Recall AUC Score:")
print(pr_auc)

#%%
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Apply SMOTE
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Choose a model and perform hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

model = RandomForestClassifier(class_weight='balanced', random_state=0)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_resampled, y_resampled)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print("\nPrecision-Recall AUC Score:")
print(pr_auc)

#%%
from sklearn.inspection import partial_dependence
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#%%
from sklearn.inspection import permutation_importance

result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=0)

# Plot the results
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Feature Importance")
plt.show()

#%%
from sklearn.inspection import partial_dependence
import seaborn as sns

most_important_feature = 'frequency'

X_train_df = pd.DataFrame(X_train, columns=X.columns)

# Calculate partial dependence
pdp_results = partial_dependence(best_model, X_train_df, [most_important_feature], grid_resolution=50)

# Extract the partial dependence values and axes
pdp_values = pdp_results['average']
pdp_values = pdp_values[0]  # For single feature
pdp_axis   = pdp_results['values'][0]  # For single feature

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(pdp_axis, pdp_values)
plt.xlabel(most_important_feature)

plt.ylabel('Partial Dependence')
plt.title(f'Partial Dependence of {most_important_feature}')
plt.show()


#%%
import shap

X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_df)

# SHAP dependence plot
shap.dependence_plot(most_important_feature, shap_values[1], X_test_df)

