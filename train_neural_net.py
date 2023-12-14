import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

EPOCHS = 1000

# Define a custom neural network class for regression
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropouts):
        super(NeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()
        in_size = input_dim

        for out_size, dropout_rate in zip(hidden_sizes, dropouts):
            self.layers.append(nn.Linear(in_size, out_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            in_size = out_size

        self.layers.append(nn.Linear(in_size, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def fitNeuralNetwork(X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.001):
    hidden_sizes = [512, 256, 128, 64, 32, 16, 8]
    output_dim = 1
    dropouts = [0.5 ,0.25, 0.125, 0.1, 0.0625, 0.03, 0.0]
    model = NeuralNetwork(X_train.shape[1], hidden_sizes, output_dim, dropouts)
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    prev_val_loss = float('inf')
    patience = 20  # Number of epochs to wait for improvement

    train_losses = []
    val_losses = []

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

        # Append training and validation losses
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    plot_performance(train_losses, val_losses, 'Training and Validation Loss')

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

# Computes the error and residuals of the sportsbook's predictions
def computeSportsbookResults(home_spread, acutal_home_spread):
    # Compute mean squared error on sportsbook's predictions
    error = mean_squared_error(acutal_home_spread, home_spread)

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

    book_error, book_residual = 0, 0
    if fit_actual_spread:
        # Grab sportsbook's home spread and the actual home spread from the result of the games
        home_spread = data.loc[:, 'home_spread']
        actual_home_spread = data.loc[:, 'actual_home_spread']

        # Convert to numpy arrays (optional)
        home_spread = home_spread.values[:900]
        actual_home_spread = actual_home_spread.values[:900]

        book_error, book_residual = computeSportsbookResults(home_spread, actual_home_spread)

    return mse_train, r_squared_train,mse_test, r_squared_test, book_error, book_residual

def plot_performance(train_loss, val_loss, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    print('Predicting Spread\n')
    mse_train, r_squared_train, mse_test, r_squared_test, book_error, book_residual = regression()
    print("Training Mean Squared Error:", mse_train)
    print("Training R-squared (R²):", r_squared_train)
    print("Testing Mean Squared Error:", mse_test)
    print("Testing R-squared (R²):", r_squared_test)

    print('\n\nPredicting Point Differential\n')
    mse_train, r_squared_train, mse_test, r_squared_test, book_error, book_residual = regression(fit_actual_spread=True)
    print("Training Mean Squared Error:", mse_train)
    print("Training R-squared (R²):", r_squared_train)
    print("Testing Mean Squared Error:", mse_test)
    print("Testing R-squared (R²):", r_squared_test)
    print("Sportsbook's Mean Square Error: ", book_error)
    print("Sportsbook's R-squared (R²):", book_residual)

if __name__ == "__main__":
    main()
