from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

param_grid = [
    {
        "imputer": [None],
        "model": [CatBoostClassifier(random_state=42, verbose=0)],
        "model__n_estimators": [750, 1000],
        "model__max_depth": [8, 10, 12],
        "model__l2_leaf_reg": [10, 20],
        "model__min_data_in_leaf": [1, 5],
        "model__auto_class_weights": ['None', 'Balanced'],
        "scaler": [None]
    },
    {
        "model": [LogisticRegression(random_state=42)],
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__penalty":['none', 'elasticnet', 'l1', 'l2'],
        "model__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "model__class_weight": ['balanced', None],
    },
]