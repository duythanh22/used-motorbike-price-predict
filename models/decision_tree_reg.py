from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

class ModelDecisionTreeRegressor:

    def __init__(self, model=None):
        if model is None:
            self.model = DecisionTreeRegressor()
        else:
            self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def evaluate(self, y_true, y_pred):
        self.score_mae = mean_absolute_error(y_true, y_pred)
        self.score_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        self.score_mape = mean_absolute_percentage_error(y_true, y_pred)
        self.score_r2 = r2_score(y_true, y_pred)

        print(f"MAE: {self.score_mae:.05f}")
        print(f"RMSE: {self.score_rmse:.05f}")
        print(f"MAPE: {self.score_mape:.05f}")
        print(f"R2_SQUARE: {self.score_r2:.05f}")

        return self.score_mae, self.score_rmse, self.score_mape, self.score_r2

    def plot_line(self, width, height, X_test, y_test):
        f, ax = plt.subplots(1)
        f.set_figheight(height)
        f.set_figwidth(width)

        sns.lineplot(x=range(0, X_test[0:100].shape[0]), y=self.y_pred[0:100], ax=ax, color='red', label='Giá trị dự đoán')
        sns.lineplot(x=range(0, X_test[0:100].shape[0]), y=y_test[0:100], ax=ax, color='green', label='Giá trị thực tế')

        ax.set_title('Biểu đồ đường giá xe dự đoán và thực thế')
        ax.set_xlabel(xlabel='Sample', fontsize=14)
        ax.set_ylabel(ylabel='Price', fontsize=14)
        plt.show()

    def plot_scatter(self, width, height, y_pred, y_test):
        f, ax = plt.subplots(1)
        f.set_figheight(height)
        f.set_figwidth(width)
        plt.scatter(y_test, y_pred, color='red')
        plt.plot(y_test, y_test)
        plt.grid()
        plt.xlabel('Giá thực tế')
        plt.ylabel('Giá dự đoán')
        plt.title('Biểu đồ phân tán giá xe dự đoán và thực thế')
        plt.show()

    def plot_residuals(self, width, height, y_pred, y_test):
        f, ax = plt.subplots(1)
        f.set_figheight(height)
        f.set_figwidth(width)
        residuals = y_pred - y_test

        # Vẽ biểu đồ Residuals
        plt.scatter(y_test, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Price')
        plt.ylabel('Residuals')
        plt.title('Biểu đồ dư thừa giá xe dự đoán và thực tế')
        plt.show()
