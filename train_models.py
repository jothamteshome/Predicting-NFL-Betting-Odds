import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Fits the regression model
def fitRidgeRegressionModel(X_train, y_train):
    model = Ridge(alpha=0.001, random_state=47)
    model.fit(X_train, y_train)

    return model

def fitLinearRegressionModel(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def fitRandomForest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=47)
    model.fit(X_train, y_train)

    return model

# Predicts the model based on trained coefficients
def predictValues(model, X_test):
    return model.predict(X_test)

# Computes the mean squared error of the predictions
def computeMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

# Gives accuracy based on prediction values
def computeRegressionR2(model, X_test, y_test):
    return model.score(X_test, y_test)

# Computes the error and residuals for a given model
def computeModelResults(model, X_test, y_test):
    # Use model to predict values
    y_pred = predictValues(model, X_test)

    # Compute mean squared error on the predictions
    error = computeMSE(y_test, y_pred)
    
    # Residual sum of squares of predictions
    residuals = computeRegressionR2(model, X_test, y_test)

    return error, residuals

def regression():
    # Read in preprocessed data
    data = pd.read_csv('Pre-Processed Data/preprocessed_additional_data.csv')

    # Drop rows with null values
    data = data.dropna()

    # Split data into X and y matrices
    y = data.loc[:, 'home_spread']
    X = data.drop(['Date', 'week', 'home_spread', 'home_team', 'away_spread', 'away_team'], axis=1)

    # Convert to numpy arrays (optional)
    y = y.values
    X = X.values

    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=47)
    
    # Fit the models on training data
    linRegression = fitLinearRegressionModel(X_train, y_train)
    ridgeRegression = fitRidgeRegressionModel(X_train, y_train)
    randomForest = fitRandomForest(X_train, y_train)

    # Compute results for each model type
    modelResults = {'Linear Regression': computeModelResults(linRegression, X_test, y_test),
                    'Ridge Regression': computeModelResults(ridgeRegression, X_test, y_test),
                    'Random Forest': computeModelResults(randomForest, X_test, y_test)}

    for result in modelResults:
        print(f"{result}\n{'-'*25}\nMSE: {modelResults[result][0]}\nR^2: {modelResults[result][1]}\n\n")

    
def main():
    regression()
    


if __name__ == "__main__":
    main()