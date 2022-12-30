import joblib
from pathlib import Path
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from exploration.Modules.DataPreprocessingClass import DataPreprocessing
from exploration.Modules.FeatureGeneratorClass import FeatureGenerator
from exploration.Modules.FeatureSelectorClass import FeatureSelector
from exploration.config import param_grid


PATH_TO_PROJECT = Path(os.getcwd()).parent
PATH_TO_TRAIN_DATA = PATH_TO_PROJECT.joinpath('data/raw_data/train.csv')
PATH_TO_TEST_DATA = PATH_TO_PROJECT.joinpath('data/raw_data/test.csv')
PATH_TO_ARTEFACTS = PATH_TO_PROJECT.joinpath('exploration/artefacts')
PATH_TO_PREPARED_DATA = PATH_TO_PROJECT.joinpath('data/prepared_data')
MODEL_FILE = 'model.pickle'
TRAIN_FEATURES_FILE = 'features_train.pickle'
TEST_FEATURES_FILE = 'features_test.pickle'
TRAIN_TARGET_FILE = 'target_train.pickle'
SUBMIT_FILE = 'submit.csv'

# если данные не подготовлены, выполняем предобработку данных
if TRAIN_FEATURES_FILE not in os.listdir(PATH_TO_PREPARED_DATA) \
        or TEST_FEATURES_FILE not in os.listdir(PATH_TO_PREPARED_DATA):
    # чтение train и test даты
    train_data = pd.read_csv(PATH_TO_TRAIN_DATA)
    test_data = pd.read_csv(PATH_TO_TEST_DATA)
    print('файлы train_data и test_data прочитаны')

    # первичная подготока данных
    preprocessor = DataPreprocessing()
    preprocessor.fit(train_data)
    train_data = preprocessor.transform(train_data)
    test_data = preprocessor.transform(test_data)
    print('первичная подготовка данных выполнена')

    # генерация признаков
    feature_generator = FeatureGenerator()
    feature_generator.fit(train_data)
    train_data = feature_generator.transform(train_data)
    test_data = feature_generator.transform(test_data)
    print('генерация признаков завершена')

    # отбор признаков
    feature_selector = FeatureSelector()
    feature_selector.fit(train_data)
    features_train = feature_selector.transform(train_data)
    features_test = feature_selector.transform(test_data)
    print('отбор признаков завершен')

    # сохранение гововых файлов
    target_train = train_data['target']
    features_train.to_pickle(f'{PATH_TO_PREPARED_DATA}/{TRAIN_FEATURES_FILE}')
    features_test.to_pickle(f'{PATH_TO_PREPARED_DATA}/{TEST_FEATURES_FILE}')
    target_train.to_pickle(f'{PATH_TO_PREPARED_DATA}/{TRAIN_TARGET_FILE}')
    print(f'подготовленные файлы сохранены в {PATH_TO_PREPARED_DATA}')
else:
    features_train = pd.read_pickle(f'{PATH_TO_PREPARED_DATA}/{TRAIN_FEATURES_FILE}')
    features_test = pd.read_pickle(f'{PATH_TO_PREPARED_DATA}/{TEST_FEATURES_FILE}')
    target_train = pd.read_pickle(f'{PATH_TO_PREPARED_DATA}/{TRAIN_TARGET_FILE}')
    print('подготовленные файлы уже существуют, переходим к созданию модели')

if MODEL_FILE not in os.listdir(PATH_TO_ARTEFACTS):
    # пайплайн модели
    pipeline_stages = []
    pipeline_stages.append(("imputer", SimpleImputer(strategy='median')))
    pipeline_stages.append(("scaler", StandardScaler()))
    pipeline_stages.append(("model", LogisticRegression(verbose=0)))
    pipeline = Pipeline(pipeline_stages)
    splitter = StratifiedKFold(n_splits=5)
    # поиск лучших параметров
    grid = GridSearchCV(pipeline, param_grid=param_grid, scoring="roc_auc", cv=splitter, verbose=2)
    grid.fit(features_train, target_train)
    best_model = grid.best_estimator_
    print(f'результат обучения модели ROC_AUC - {grid.best_score_}')
    joblib.dump(best_model, f'{PATH_TO_ARTEFACTS}/{MODEL_FILE}')
else:
    print('модель уже сохранена, переходим к сабмиту')
    best_model = joblib.load(f'{PATH_TO_ARTEFACTS}/{MODEL_FILE}')

calibrated_clf = CalibratedClassifierCV(best_model, cv=StratifiedKFold(n_splits=5))
calibrated_clf.fit(features_train, target_train)
y_pred_proba = calibrated_clf.predict_proba(features_test)
y_pred_proba = y_pred_proba[:, 1]

# # обучаем лучшую модель
# best_model.fit(features_train, target_train)
# y_pred_proba = best_model.predict_proba(features_test)
# y_pred_proba = y_pred_proba[:, 1]
# print('лучшая модель обучена')
#
# выполняем submit
submit_data = pd.DataFrame(data=y_pred_proba, columns=['Predicted'])
submit_data.to_csv(f'{PATH_TO_PREPARED_DATA}/{SUBMIT_FILE}', index_label='Id', sep=",")
print('сабмит сохранен')


