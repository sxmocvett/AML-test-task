import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


class FeatureSelector:
    """Отбор признаков"""

    def __init__(self):
        """Параметры класса"""
        self.rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
        self.important_columns = []

    def fit(self, data: pd.DataFrame):
        """Сохранение статистик"""
        # разделение фич и таргета
        X_copy = data.drop(columns=['target'])
        y = data['target']
        # заполняем пропуски (с пропусками не работает)
        X_copy.fillna(-9999, inplace=True)
        # инициализация селектора
        feature_selector = BorutaPy(self.rfc, n_estimators='auto', verbose=2, random_state=1)
        feature_selector.fit(X_copy.values, y.values)
        # определение отобранных признаков
        for col, status in zip(X_copy.columns, feature_selector.support_):
            if status:
                self.important_columns.append(col)

    def transform(self, data: pd.DataFrame):
        """Трансформация данных"""
        # формирование конечного набора признаков
        data = data[self.important_columns]
        return data