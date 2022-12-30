import warnings
import pandas as pd
from feature_engine.imputation import AddMissingIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
warnings.filterwarnings('ignore')


class FeatureGenerator:
    """Генерация новых признаков"""

    def __init__(self):
        """Параметры класса"""
        self.X_copy = None
        self.outliers_searcher = None
        self.ami = AddMissingIndicator()
        self.outliers_searcher_fitted_dict = {}
        self.model_feature_selector = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)

    def fit(self, data: pd.DataFrame):
        """Сохранение статистик"""
        # убираем признаки, которые не будут участвовать в генерации новых
        self.X_copy = data.drop(columns=['target', 'entity_0017_cd_a',
                                      'entity_0017_cd_b', 'entity_0017_cd_c',
                                      'entity_0017_cd_d', 'repeat_user']
                             )
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

        return data