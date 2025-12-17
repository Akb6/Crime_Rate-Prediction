# Crime Rate Prediction

A machine learning project that predicts crime rates based on location, year, and population.

## Overview

This project builds a regression model to predict crime rates for different regions across years. It uses data from US counties with features like:
- **State** (categorical)
- **PinCode/County Code** (categorical)
- **Year** (numerical)
- **Population** (numerical)
- **Crime Rate** (target to predict)

## Dataset

The dataset (`Book1.csv`) contains historical crime statistics for US counties from 1967–2017:
- **44+ records** covering multiple states and decades
- **Manually compiled** from public crime statistics and census data
- **Features**: State, PinCode, Year, Population, Crime Rate

**Dataset Attribution:**
This dataset was compiled from public US crime statistics and population census data. It represents a curated subset of county-level crime records for educational and research purposes.

## Model Performance

| Model | CV Score | Test MSE | Test R² |
|-------|----------|----------|---------|
| **Linear Regression** | -0.44 | 4.16 | 0.15 |
| Random Forest | -0.39 | 5.78 | -0.18 |

**Best Model:** Linear Regression (R² = 0.15)

## How It Works

1. **Data Preprocessing**
   - Categorical features (State, PinCode) → OneHotEncoder
   - Numerical features (Year, Population) → StandardScaler
   - Combined via ColumnTransformer pipeline

2. **Model Training**
   - Trains Linear Regression and Random Forest Regressor
   - Uses 5-fold cross-validation for robust evaluation
   - Compares models using MSE and R² metrics

3. **Predictions**
   - Model predicts crime rate given state, pincode, year, population
   - Example: "Cherokee, AL" (2010, Pop 154) → Crime Rate ~8.4

## Quick Start

git clone https://github.com/Akb6/Crime_Rate-Prediction
cd Crime_Rate-Prediction

Install dependencies
pip install pandas numpy scikit-learn

Run the pipeline
python crime_rate_prediction.py

## Output

Crime Rate Prediction - ML Pipeline

✓ Dataset loaded: 44 records, 4 features

✓ Data split: Train=35, Test=9

✓ Features preprocessed: 7 features after encoding

============================================================
LINEAR REGRESSION MODEL
Mean CV Score: -0.4357 (+/- 0.7194)
Test MSE: 4.1647
Test R² Score: 0.1524
Test RMSE: 2.0408

============================================================
RANDOM FOREST REGRESSOR MODEL
Mean CV Score: -0.3866 (+/- 0.8131)
Test MSE: 5.7757
Test R² Score: -0.1754
Test RMSE: 2.4033

============================================================
MODEL COMPARISON
text
          Model Mean CV Score Test MSE Test R² Score
Linear Regression -0.4357 4.1647 0.1524
Random Forest -0.3866 5.7757 -0.1754

✓ Best Model: Linear Regression (R² = 0.1524)

============================================================
✓ TRAINING COMPLETE


## Why R² is Low

The small dataset (44 records) and limited feature set mean the model explains ~15% of variance. Adding features like literacy rate, unemployment, or urbanization would improve predictions.

## Technical Stack

- **Python 3.8+**
- **scikit-learn** - Machine Learning
- **pandas** - Data processing
- **numpy** - Numerical computing

## Future Improvements

- Add more features (literacy rate, unemployment, education levels)
- Collect more historical data (500+ records)
- Try advanced models (Gradient Boosting, XGBoost)
- Time-series analysis for trend prediction
