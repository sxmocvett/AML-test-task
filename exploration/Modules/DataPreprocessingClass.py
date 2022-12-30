import pandas as pd
import numpy as np
import re


class DataPreprocessing:
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""

    def _get_days(self, data: pd.DataFrame, col: str):
        """Получение количества дней из строковой даты"""
        day_numbers = []
        for amount_days in data[col]:
            days = int(re.search(r'[0-9]+', amount_days)[0])
            day_numbers.append(days)
        data[col] = day_numbers
        return data

    def fit(self, data: pd.DataFrame):
        """Сохранение статистик"""

    def transform(self, data: pd.DataFrame):
        """Трансформация данных"""
        # удаление колонки Id
        if 'Id' in data.columns:
            data.drop(columns='Id', inplace=True)

        # получение дней из fact_dt
        self._get_days(data, 'fact_dt')

        # изменение значений entity_0016_rt
        data['entity_0016_rt'] = np.where(data['entity_0016_rt'] < 0.5,
                                          data['entity_0016_rt'] + 1,
                                          data['entity_0016_rt'])

        # преобразование категориальных переменных
        data = pd.get_dummies(data)

        # поиск повторяющихся юзеров
        repeated_users = data[data.duplicated(subset='user_id')]['user_id'].unique()
        data['repeat_user'] = np.where(data['user_id'].isin(repeated_users), 1, 0)

        # удаление колонки user_id
        data.drop(columns='user_id', inplace=True)

        return data