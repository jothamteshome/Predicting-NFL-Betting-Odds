import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

EPOCHS = 300

# Define a custom neural network class for regression
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout5(x)
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x


def fitNeuralNetwork(X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.001):
    model = NeuralNetwork(X_train.shape[1])
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    prev_val_loss = float('inf')
    patience = 20  # Number of epochs to wait for improvement

    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Validation loss
        with torch.no_grad():
            model.eval()
            val_inputs = torch.tensor(X_val, dtype=torch.float32)
            val_labels = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)

        if val_loss < prev_val_loss:
            prev_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Load the best model state
    model.load_state_dict(best_model_state)

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
    data = pd.read_csv('Pre-Processed Data/preprocessed_additional_data.csv')

    # Drop rows with null values
    data = data.dropna()

    # Split data into X and y matrices
    y = data.loc[:, 'home_spread']
    X = data.drop(['Date', 'week', 'home_spread', 'home_team', 'away_spread', 'away_team'], axis=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to numpy arrays
    y = y.values

    # Split into training and testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, train_size=0.8, random_state=47)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=47)

    # Fit the neural network model on training data
    model = fitNeuralNetwork(X_train, y_train, X_val, y_val, epochs=EPOCHS)

    # Use the trained model to predict values for training data
    y_pred = predictValues(model, X_train)

    # Reshape y_train to have the same shape as y_pred
    y_train = y_train.reshape(y_pred.shape)

    # Evaluate the model on training
    mse_train, r_squared_train = evaluateRegressionModel(y_train, y_pred)

    # Use the trained model to predict values
    y_pred = predictValues(model, X_test)

    # Reshape y_test to have the same shape as y_pred
    y_test = y_test.reshape(y_pred.shape)

    # Evaluate the model on testing
    mse_test, r_squared_test = evaluateRegressionModel(y_test, y_pred)

    return mse_train, r_squared_train,mse_test, r_squared_test

def main():
    mse_train, r_squared_train, mse_test, r_squared_test = regression()
    print("Training Mean Squared Error:", mse_train)
    print("Training R-squared (R²):", r_squared_train)
    print("Testing Mean Squared Error:", mse_test)
    print("Testing R-squared (R²):", r_squared_test)

if __name__ == "__main__":
    main()
