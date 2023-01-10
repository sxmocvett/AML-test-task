import warnings
import pandas as pd
from feature_engine.imputation import AddMissingIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from typing import List
import itertools
warnings.filterwarnings('ignore')


class FeatureGenerator:
    """Генерация новых признаков"""

    def __init__(self):
        """Параметры класса"""
        self.X_copy = None
        self.outliers_searcher = None
        self.embeding_columns = None
        self.count_columns = None
        self.ratio_columns = None
        self.ami = AddMissingIndicator()
        self.outliers_searcher_fitted_dict = {}
        self.model_feature_selector = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)

    @staticmethod
    def _math_embeding_columns(data: pd.DataFrame, columns: List):
        """Генерация признаков из эмбедингов"""
        embedings_features = data[columns].copy()
        embedings_features['mul_emb'] = embedings_features.prod(axis=1)
        embedings_features['sum_emb'] = embedings_features.sum(axis=1)
        embedings_features['sum_abs_emb'] = abs(embedings_features).sum(axis=1)
        embedings_features.drop(columns=columns, inplace=True)
        return embedings_features

    @staticmethod
    def _math_count_columns(data: pd.DataFrame, columns: List):
        """Генерация признаков из количественных параметров"""
        count_features = data[columns].copy()
        count_features['mul_ct'] = count_features.fillna(1).prod(axis=1)
        count_features['sum_ct'] = count_features.fillna(0).sum(axis=1)
        count_features.drop(columns=columns, inplace=True)
        return count_features

    @staticmethod
    def _math_ratio_columns(data: pd.DataFrame, columns: List):
        """Генерация признаков из ratio параметров"""
        ratio_features = data[columns].copy()
        ratio_features['mul_rt'] = ratio_features.fillna(1).prod(axis=1)
        ratio_features['sum_rt'] = ratio_features.fillna(0).sum(axis=1)
        ratio_features.drop(columns=columns, inplace=True)
        return ratio_features

    def fit(self, data: pd.DataFrame):
        """Сохранение статистик"""
        # убираем признаки, которые не будут участвовать в генерации новых
        self.X_copy = data.drop(columns=['target', 'entity_0017_cd_a',
                                         'entity_0017_cd_b', 'entity_0017_cd_c',
                                         'entity_0017_cd_d', 'repeat_user']
                                )

        # определение списка признаков emb, ct и rt
        self.embeding_columns = [col for col in self.X_copy.columns if 'emb' in col]
        self.count_columns = [col for col in self.X_copy.columns if 'ct' in col]
        self.ratio_columns = [col for col in self.X_copy.columns if 'rt' in col]

        # заполнение пропусков
        self.X_copy.fillna(-9999, inplace=True)
        for col in tqdm(self.X_copy.columns):
            self.outliers_searcher = IsolationForest(random_state=42, verbose=0)
            self.outliers_searcher.fit(self.X_copy[[col]])
            self.outliers_searcher_fitted_dict[col] = self.outliers_searcher

    def transform(self, data: pd.DataFrame):
        """Трансформация данных"""
        # добавление признаков - индикаторов пропусков
        data = self.ami.fit_transform(data)

        # удаление одинаковых столбцов
        data.drop(columns=['entity_0008_ct_na', 'entity_0012_ct_na', 'entity_0014_rt_na'], inplace=True)

        # добавление признаков - индикаторов выбросов
        for col in self.X_copy.columns:
            if col in data.columns:
                anomalies_detected = self.outliers_searcher_fitted_dict[col].predict(data[[col]].fillna(-9999))
                data[f'{col}_outlier'] = anomalies_detected

        # добавление математических признаков
        data_ratio = self._math_ratio_columns(data, self.ratio_columns)
        data_count = self._math_count_columns(data, self.count_columns)
        data_embeding = self._math_embeding_columns(data, self.embeding_columns)
        data = pd.concat([data, data_ratio, data_count, data_embeding], axis=1)

        return data