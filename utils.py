from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class Normalize:
    scaler_x = None
    scaler_y = None

    def __init__(self):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def fit_transform(self, X_train, y_train):
        X_train_normalized = self.scaler_x.fit_transform(X_train)
        y_train_normalized = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        return X_train_normalized, y_train_normalized

    def normalize_data(self, X_test, y_test):
        X_test_normalized = self.scaler_x.transform(X_test)
        y_test_normalized = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        return X_test_normalized, y_test_normalized

    def inverse_normalize_data(self, y_pred):
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()


def performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2
