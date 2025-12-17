import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score


def load_and_prepare_data(file_path):
    """Load dataset and prepare features/target."""
    data = pd.read_csv(file_path)
    X = data.drop('Crime Rate', axis=1)
    y = data['Crime Rate']
    return X, y


def build_preprocessor():
    """Create preprocessing pipeline for categorical and numerical features."""
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, ['State', 'PinCode']),
        ('numerical', numerical_transformer, ['Year', 'Population'])
    ])
    
    return preprocessor


def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate Linear Regression and Random Forest models."""
    
    # Linear Regression
    lr_model = LinearRegression(fit_intercept=True)
    lr_model.fit(X_train, y_train)
    
    cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=5, n_jobs=-1)
    lr_pred = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    print("="*60)
    print("LINEAR REGRESSION MODEL")
    print("="*60)
    print(f"Cross-validation scores: {cv_scores_lr}")
    print(f"Mean CV Score: {np.mean(cv_scores_lr):.4f} (+/- {np.std(cv_scores_lr):.4f})")
    print(f"Test MSE: {lr_mse:.4f}")
    print(f"Test R² Score: {lr_r2:.4f}")
    print(f"Test RMSE: {np.sqrt(lr_mse):.4f}\n")
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, n_jobs=-1)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print("="*60)
    print("RANDOM FOREST REGRESSOR MODEL")
    print("="*60)
    print(f"Cross-validation scores: {cv_scores_rf}")
    print(f"Mean CV Score: {np.mean(cv_scores_rf):.4f} (+/- {np.std(cv_scores_rf):.4f})")
    print(f"Test MSE: {rf_mse:.4f}")
    print(f"Test R² Score: {rf_r2:.4f}")
    print(f"Test RMSE: {np.sqrt(rf_mse):.4f}\n")
    
    # Model comparison
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'Mean CV Score': [f"{np.mean(cv_scores_lr):.4f}", f"{np.mean(cv_scores_rf):.4f}"],
        'Test MSE': [f"{lr_mse:.4f}", f"{rf_mse:.4f}"],
        'Test R² Score': [f"{lr_r2:.4f}", f"{rf_r2:.4f}"]
    })
    
    print(comparison.to_string(index=False))
    
    best_model_name = "Linear Regression" if lr_r2 > rf_r2 else "Random Forest"
    best_model = lr_model if lr_r2 > rf_r2 else rf_model
    best_r2 = max(lr_r2, rf_r2)
    
    print(f"\n✓ Best Model: {best_model_name} (R² = {best_r2:.4f})\n")
    
    return best_model, best_model_name


def main():
    print("Crime Rate Prediction - ML Pipeline\n")
    print("="*60)
    
    # Load data
    file_path = "Book1.csv"
    try:
        X, y = load_and_prepare_data(file_path)
        print(f"✓ Dataset loaded: {X.shape[0]} records, {X.shape[1]} features\n")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}\n")
    
    # Build preprocessor
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"✓ Features preprocessed: {X_train_transformed.shape[1]} features after encoding\n")
    
    # Train models
    best_model, best_name = train_models(
        X_train_transformed, X_test_transformed, y_train, y_test, preprocessor
    )
    
    print("="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
