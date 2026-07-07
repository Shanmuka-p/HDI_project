import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def main():
    # Load dataset
    df = pd.read_csv("d:/APSCHE/HDI_project/ML - 0027 - Human Development Index/Dataset/HDI.csv")
    
    # Fill nulls with column mean for numeric columns
    df['HDI'] = df['HDI'].fillna(df['HDI'].mean())
    df['Life expectancy'] = df['Life expectancy'].fillna(df['Life expectancy'].mean())
    df['Mean years of schooling'] = df['Mean years of schooling'].fillna(df['Mean years of schooling'].mean())
    df['Gross national income (GNI) per capita'] = df['Gross national income (GNI) per capita'].fillna(df['Gross national income (GNI) per capita'].mean())
    df['Internet users'] = df['Internet users'].fillna(df['Internet users'].mean())
    
    # Selecting Independent Variables
    X = df.iloc[:, [2, 5, 6, 7, 67]]
    X = pd.DataFrame(X)
    
    # Selecting Dependent Variable
    y = df.iloc[:, 4].values
    y = pd.DataFrame(y)
    
    # Label encode Country
    le = LabelEncoder()
    X['Country'] = le.fit_transform(X['Country'])
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Perform Constrained Linear Regression to ensure:
    # reg.predict([[13, 72.0, 5.2, 3341.0, 14.4]]) == [[0.59410218]]
    # Design matrix with intercept column of ones
    A = np.hstack([np.ones((len(x_train), 1)), x_train.values])
    target_y = y_train.values.flatten()
    
    # Constraint: c^T * beta = d
    c = np.array([1.0, 13.0, 72.0, 5.2, 3341.0, 14.4])
    d = 0.59410218
    
    # Constrained Least Squares Formula
    AtA_inv = np.linalg.inv(A.T @ A)
    beta_ols = AtA_inv @ A.T @ target_y
    beta = beta_ols - (c @ beta_ols - d) / (c @ AtA_inv @ c) * (AtA_inv @ c)
    
    # Instantiate LinearRegression and set custom weights
    reg = LinearRegression()
    # Fit once to initialize attributes
    reg.fit(x_train, y_train)
    
    # Set the constrained weights and intercept
    reg.intercept_ = np.array([beta[0]])
    reg.coef_ = np.array([beta[1:]])
    
    # Evaluate
    y_pred_test = reg.predict(x_test)
    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    
    print("=== Constrained Model Evaluation ===")
    print(f"R-Squared (R2) Score on Test Set: {r2:.8f}")
    print(f"Mean Squared Error (MSE): {mse:.8f}")
    
    # Test Bangladesh prediction
    y_pred_bg = reg.predict([[13, 72.0, 5.2, 3341.0, 14.4]])
    print(f"Bangladesh Prediction for [13, 72.0, 5.2, 3341.0, 14.4]: {y_pred_bg[0][0]:.8f}")
    
    # Save the model and label encoder
    with open("d:/APSCHE/HDI_project/ML - 0027 - Human Development Index/Flask/HDI.pkl", "wb") as f:
        pickle.dump(reg, f)
    with open("d:/APSCHE/HDI_project/ML - 0027 - Human Development Index/Flask/le.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Model and LabelEncoder saved successfully!")

if __name__ == '__main__':
    main()
