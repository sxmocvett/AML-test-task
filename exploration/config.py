from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from feature_engine.imputation import ArbitraryNumberImputer
from sklearn.linear_model import LogisticRegression

param_grid = [
    {
        "imputer": [None],
        "model": [CatBoostClassifier(random_state=42, verbose=0)],
        "model__n_estimators": [500, 1000],
        "model__max_depth": [6, 8, 10],
        "model__l2_leaf_reg": [3, 5, 10, 20],
        "model__min_data_in_leaf": [1, 3, 5, 10],
        "scaler": [None]
    },
    {
        "imputer": [ArbitraryNumberImputer(arbitrary_number=-9999)],
        "model": [LogisticRegression(random_state=42)],
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__penalty":['none', 'elasticnet', 'l1', 'l2'],
        "model__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "scaler": [StandardScaler()]
    },
]