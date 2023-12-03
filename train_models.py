import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# Computes the error and residuals of the sportsbook's predictions
def computeSportsbookResults(home_spread, acutal_home_spread):
    # Compute mean squared error on sportsbook's predictions
    error = computeMSE(home_spread, acutal_home_spread)

    # Residual sum of squares of sportbook's predictions
    residuals = r2_score(acutal_home_spread, home_spread)

    return error, residuals

def regression(fit_actual_spread=False):
    # Read in preprocessed data
    data = pd.read_csv('Pre-Processed Data/preprocessed_additional_data.csv')

    # Drop rows with null values
    data = data.dropna()

    # Split data into X and y matrices
    if fit_actual_spread:
        y = data.loc[:, 'actual_home_spread']
    else:
        y = data.loc[:, 'home_spread']
    X = data.drop(['Date', 'week', 'home_spread', 'home_team', 'away_spread', 'away_team', 'home_score', 'away_score', 'actual_home_spread'], axis=1)

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

    if fit_actual_spread:
        # Grab sportsbook's home spread and the actual home spread from the result of the games
        home_spread = data.loc[:, 'home_spread']
        actual_home_spread = data.loc[:, 'actual_home_spread']

        # Convert to numpy arrays (optional)
        home_spread = home_spread.values
        actual_home_spread = actual_home_spread.values

        book_error, book_residual = computeSportsbookResults(home_spread, actual_home_spread)
        print(f"Sportsbook's Prediction\n{'-' * 25}\nMSE: {book_error}\nR^2: {book_residual}\n\n")


def main():
    print('Predicting Spread')
    regression()
    print('Predicting Point Differential\n')
    regression(fit_actual_spread=True)
    


if __name__ == "__main__":
    main()