import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

# Define a custom neural network class for regression
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Fits the neural network model
def fitNeuralNetwork(X_train, y_train, epochs=100, learning_rate=0.001):
    model = NeuralNetwork(X_train.shape[1])
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model

# Predicts the model based on the trained neural network
def predictValues(model, X_test):
    inputs = torch.tensor(X_test, dtype=torch.float32)
    return model(inputs).detach().numpy()

# Computes the mean squared error and R-squared of the predictions
def evaluateRegressionModel(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    return mse, r_squared

def regression():
    # Read in preprocessed data
    data = pd.read_csv('Pre-Processed Data/preprocessed_data.csv')

    # Replace outcome labels with integer labels
    data['home_outcome'] = data['home_outcome'].replace({'L': 0, 'W': 1, 'T': 2})

    # Split data into X and y matrices
    y = data.loc[:, 'home_spread']
    X = data.drop(['Date', 'home_spread', 'home_team', 'away_spread', 'away_team'], axis=1)

    # Convert to numpy arrays (optional)
    y = y.values
    X = X.values

    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=47)

    # Fit the neural network model on training data
    model = fitNeuralNetwork(X_train, y_train)

    # Use the trained model to predict values
    y_pred = predictValues(model, X_test)

    # Reshape y_test to have the same shape as y_pred
    y_test = y_test.reshape(y_pred.shape)

    # Evaluate the model
    mse, r_squared = evaluateRegressionModel(y_test, y_pred)

    return mse, r_squared

def main():
    mse, r_squared = regression()
    print("Mean Squared Error:", mse)
    print("R-squared (R²):", r_squared)

if __name__ == "__main__":
    main()