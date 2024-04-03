import pandas as pd
import joblib
from models import decision_tree_reg as dt_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor

def read_and_process_data(file_path):
    df = pd.read_csv(file_path)
    categorical_columns = ['Brand', 'Model', 'Capacity', 'Type']

    encoder = OneHotEncoder(drop='first', sparse_output=False)

    label_encoder = LabelEncoder()

    df['Capacity'] = label_encoder.fit_transform(df['Capacity'])
    df['Type'] = label_encoder.fit_transform(df['Type'])

    encoded_data = encoder.fit_transform(df[categorical_columns[:-2]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns[:-2]))

    df = pd.concat([df.drop(categorical_columns[:-2], axis=1), encoded_df], axis=1)

    y = df['Price']
    df = df.drop('Price', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=42)

    encoded_columns = df.columns
    joblib.dump(encoded_columns, 'models/encoded_columns.pkl')

    return X_train, X_test, y_train, y_test


def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Score (Negative MAE): ", grid_search.best_score_)

    return grid_search.best_params_


def train_and_evaluate_model(model_params, X_train, X_test, y_train, y_test):
    model = dt_model.DecisionTreeRegressor(**model_params)
    dt_reg = dt_model.ModelDecisionTreeRegressor(model)
    dt_reg.train(X_train, y_train)
    dt_y_pred = dt_reg.predict(X_test)

    dt_score_mae, dt_score_rmse, dt_score_mape, dt_r2 = dt_reg.evaluate(y_test, dt_y_pred)

    joblib.dump(dt_reg, 'models/decision_tree_model.pkl')

    return dt_reg, dt_y_pred, dt_score_mae, dt_score_rmse, dt_score_mape, dt_r2


def plot_results(dt_reg, X_test, y_test, dt_y_pred):
    dt_reg.plot_line(20, 5, X_test, y_test)
    dt_reg.plot_scatter(20, 5, dt_y_pred, y_test)
    dt_reg.plot_residuals(20, 5, dt_y_pred, y_test)

def load_model():
    loaded_model = joblib.load('models/decision_tree_model.pkl')
    return loaded_model


if __name__ == "__main__":
    file_path = "data/clean_motorbikes_all.csv"
    X_train, X_test, y_train, y_test = read_and_process_data(file_path)

    # model = DecisionTreeRegressor()
    #
    # param_grids = {
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': [None, 'sqrt', 'log2']
    # }
    #
    # best_params = perform_grid_search(model, param_grids, X_train, y_train)
    #
    # dt_reg, dt_y_pred, dt_score_mae, dt_score_rmse, dt_score_mape, dt_r2 = \
    #     train_and_evaluate_model(best_params, X_train, X_test, y_train, y_test)
    #
    # plot_results(dt_reg, X_test, y_test, dt_y_pred)


    # read model, visualize

    model = load_model()
    y_pred = model.predict(X_test)
    dt_score_mae, dt_score_rmse, dt_score_mape, dt_r2 = model.evaluate(y_test, y_pred)
    plot_results(model, X_test, y_test, y_pred)