import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Fits the regression model
def fitRidgeRegressionModel(X_train, y_train):
    model = Ridge(alpha=0.001, random_state=47, solver='lsqr')
    model.fit(X_train, y_train)

    return model

def fitLassoRegressionModel(X_train, y_train):
    model = Lasso(0.01, random_state=47, max_iter=5000)
    model.fit(X_train, y_train)

    return model

# Fit a base linear regression model
def fitLinearRegressionModel(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

# Fit a random forest regression model
def fitRandomForest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=47)
    model.fit(X_train, y_train)

    return model

# Fit supoort vector regression model
def fitSVM(X_train, y_train):
    model = SVR(C=0.001, epsilon=0.001, kernel='linear')
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
    lassoRegression = fitLassoRegressionModel(X_train, y_train)
    ridgeRegression = fitRidgeRegressionModel(X_train, y_train)
    randomForest = fitRandomForest(X_train, y_train)
    svm = fitSVM(X_train, y_train)
    ensembleRegression = VotingRegressor(estimators=[('linR', linRegression), ('lasR', lassoRegression),
                                                     ('ridR', ridgeRegression), ('rfR', randomForest),
                                                     ('svr', svm)])
    
    ensembleRegression.fit(X_train, y_train)

    # Compute results for each model type
    modelResults = {'Linear Regression': computeModelResults(linRegression, X_test, y_test),
                    'Lasso Regression': computeModelResults(lassoRegression, X_test, y_test),
                    'Ridge Regression': computeModelResults(ridgeRegression, X_test, y_test),
                    'Random Forest': computeModelResults(randomForest, X_test, y_test),
                    'SVM': computeModelResults(svm, X_test, y_test),
                    'Voting': computeModelResults(ensembleRegression, X_test, y_test)}
    
    with open('Linear Model Results.txt', 'a') as f:
        print(f"{'Model Type'.center(20)}|{'MSE'.center(20)}|{'R-squared'.center(20)}")
        f.write(f"{'Model Type'.center(20)}|{'MSE'.center(20)}|{'R-squared'.center(20)}\n")

        for result in modelResults:
            error, residuals = round(modelResults[result][0], 2), round(modelResults[result][1], 3)
            print(f"{result.center(20)}|{str(error).center(20)}|{str(residuals).center(20)}")
            f.write(f"{result.center(20)}|{str(error).center(20)}|{str(residuals).center(20)}\n")
        
        f.write(f"\n{'-'*100}\n")

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