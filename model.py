import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    ShuffleSplit,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn
import torch
import torch.optim as optim
from utils import *
from tqdm import tqdm

plt.rc("font", family="AppleGothic")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SVR_Model:
    def __init__(self, C=0.001, epsilon=0.1):
        self.svr_rbf = SVR(kernel="rbf", C=C, epsilon=epsilon)

    def train(self, X_train, y_train):
        print("SVR Training")
        self.svr_rbf.fit(X_train, y_train)
        print("SVR Training Done")

    def predict(self, X):
        y_pred = self.svr_rbf.predict(X)

        return y_pred


class RFR_Model:
    def __init__(
        self,
        n_estimators=500,
        random_state=42,
        min_samples_leaf=3,
        min_samples_split=2,
        bootstrap=True,
    ):
        self.rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
        )

    def train(self, X_train, y_train):
        print("RFR Training")
        self.rf_regressor.fit(X_train, y_train)
        print("RFR Training Done")

    def predict(self, X):
        y_pred = self.rf_regressor.predict(X)

        return y_pred


class _LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(_LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTM_Model:
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        learning_rate,
        gradient_threshold,
        epoch,
    ):
        self.model = _LSTMModel(input_size, hidden_size, num_layers, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )
        self.epoch = epoch
        self.gradient_threshold = gradient_threshold

    def train(self, X_train, y_train):
        print("LSTM Training")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

        self.model.train()
        for epoch in tqdm(range(self.epoch)):
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_threshold
            )

            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epoch}], Loss: {loss.item():.4f}")

            # self.scheduler.step()

        print("LSTM Training Done")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).detach().numpy()
        return y_pred

    def save_model(self, path):
        torch.save(self.model.state_dict(), f"{path}.pth")
        print("모델 상태 저장 완료")


def result_of_T5toT8(model_type, params, X, y):
    X_train = X[t1_to_t4_indices].copy()
    y_train = y[t1_to_t4_indices].copy()

    normalizer = Normalize()
    X_train, y_train = normalizer.fit_transform(X_train, y_train)

    if model_type == "SVR":
        model = SVR_Model(**params)
    elif model_type == "RFR":
        model = RFR_Model(**params)
    elif model_type == "LSTM":
        model = LSTM_Model(**params)

    model.train(X_train, y_train)
    y_pred_train = model.predict(X_train)

    y_train_original = normalizer.inverse_normalize_data(y_train)
    y_pred_train_original = normalizer.inverse_normalize_data(y_pred_train)
    print(
        f"{model_type} Train RMSE: ",
        np.sqrt(mean_squared_error(y_train_original, y_pred_train_original)),
    )
    print(f"{model_type} Train R2: ", r2_score(y_train_original, y_pred_train_original))

    # T5 결과
    X_test = X[t5_indices].copy()
    y_test = y[t5_indices].copy()
    X_test, y_test = normalizer.normalize_data(X_test, y_test)
    y_pred_test = model.predict(X_test)
    y_test_original = normalizer.inverse_normalize_data(y_test)
    y_pred_test_original = normalizer.inverse_normalize_data(y_pred_test)

    print(
        f"{model_type} Test RMSE of T5: ",
        np.sqrt(mean_squared_error(y_test_original, y_pred_test_original)),
    )
    print(
        f"{model_type} Test R2 of T5: ", r2_score(y_test_original, y_pred_test_original)
    )
    t5_results = (y_test_original, y_pred_test_original)

    # T6 결과
    X_test = X[t6_indices].copy()
    y_test = y[t6_indices].copy()
    X_test, y_test = normalizer.normalize_data(X_test, y_test)
    y_pred_test = model.predict(X_test)
    y_test_original = normalizer.inverse_normalize_data(y_test)
    y_pred_test_original = normalizer.inverse_normalize_data(y_pred_test)

    print(
        f"{model_type} Test RMSE of T6: ",
        np.sqrt(mean_squared_error(y_test_original, y_pred_test_original)),
    )
    print(
        f"{model_type} Test R2 of T6: ", r2_score(y_test_original, y_pred_test_original)
    )
    t6_results = (y_test_original, y_pred_test_original)

    # T7 결과
    X_test = X[t7_indices].copy()
    y_test = y[t7_indices].copy()
    X_test, y_test = normalizer.normalize_data(X_test, y_test)
    y_pred_test = model.predict(X_test)
    y_test_original = normalizer.inverse_normalize_data(y_test)
    y_pred_test_original = normalizer.inverse_normalize_data(y_pred_test)

    print(
        f"{model_type} Test RMSE of T7: ",
        np.sqrt(mean_squared_error(y_test_original, y_pred_test_original)),
    )
    print(
        f"{model_type} Test R2 of T7: ", r2_score(y_test_original, y_pred_test_original)
    )
    t7_results = (y_test_original, y_pred_test_original)

    # T8 결과
    X_test = X[t8_indices].copy()
    y_test = y[t8_indices].copy()
    X_test, y_test = normalizer.normalize_data(X_test, y_test)
    y_pred_test = model.predict(X_test)
    y_test_original = normalizer.inverse_normalize_data(y_test)
    y_pred_test_original = normalizer.inverse_normalize_data(y_pred_test)

    print(
        f"{model_type} Test RMSE of T8: ",
        np.sqrt(mean_squared_error(y_test_original, y_pred_test_original)),
    )
    print(
        f"{model_type} Test R2 of T8: ", r2_score(y_test_original, y_pred_test_original)
    )
    t8_results = (y_test_original, y_pred_test_original)

    results = {
        "T5": t5_results,
        "T6": t6_results,
        "T7": t7_results,
        "T8": t8_results,
        "model": model,
    }

    return results
