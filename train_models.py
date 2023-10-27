import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Fits the regression model
def fitRegressionModel(X_train, y_train):
    model = Ridge(alpha=0.001, random_state=47)
    model.fit(X_train, y_train)

    return model

# Predicts the model based on trained coefficients
def predictValues(model, X_test):
    return model.predict(X_test)

# Computes the mean squared error of the predictions
def computeMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

# Gives accuracy based on prediction values
def computeRegressionAccuracy(model, X_test, y_test):
    return model.score(X_test, y_test)

def regression():
    # Read in preprocessed data
    data = pd.read_csv('Pre-Processed Data/preprocessed_data.csv')

    # Replace outcome labels with integer labels
    data['home_outcome'] = data['home_outcome'].replace({'L': 0, 'W': 1, 'T': 2})

    # Split data into X and y matrices
    y = data.loc[:,'home_spread']
    X = data.drop(['Date', 'home_spread', 'home_team', 'away_spread', 'away_team'], axis=1)
    
    # Convert to numpy arrays (optional)
    y = y.values
    X = X.values
    
    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=47)
    
    # Fit the model on training data
    model = fitRegressionModel(X_train, y_train)

    # Use model to predict values
    y_pred = predictValues(model, X_test)

    # Compute mean squared error on the predictions
    error = computeMSE(y_test, y_pred)
    
    # Accuracy of predictions
    accuracy = computeRegressionAccuracy(model, X_test, y_test)

    return error, accuracy
    
def main():
    print(regression())
    


if __name__ == "__main__":
    main()