import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from typing import List, Set, Dict, Tuple, Optional
from math import ceil
from itertools import product



class Vizualization:
    """
    Визуализация зависимостей категориальных и количественных
    признаков друг от друга
    Атрибуты:
        data - датафрейм
        cat_columns - список категориальных признаков
        num_columns - список количественных признаков
        cat_combo_2 - набор уникальных пар категориальных признаков
        num_combo_2 - набор уникальных пар количественных признаков
        num_combo_3 - набор уникальных троек количественных признаков
    """
    def __init__(self, data: pd.DataFrame, train_data: pd.DataFrame=None, test_data: pd.DataFrame=None):
        if data is not None:
            self.data = data
            self.cat_columns = data.select_dtypes(include='object').columns.tolist()
            self.num_columns = data.select_dtypes(exclude='object').columns.tolist()
            self.cat_combo_2 = itertools.combinations(self.cat_columns, 2)
            self.num_combo_2 = itertools.combinations(self.num_columns, 2)
            self.num_combo_3 = itertools.combinations(self.num_columns, 3)
        if (train_data is not None) and (test_data is not None):
            self.train_data = train_data
            self.test_data = test_data
            
    def num_num_target_plot(self, target, trend):
        """
        Построение графиков: x - количестенный признак, 
                             y - количестенный признак
        -------------------------------------------------------------------
        """
        w = 7*3
        h = len(self.num_columns)*5
        plt.figure(figsize=[w, h])
        figNumber = 1
        for col in self.num_columns:
            if col != target:
                ax = plt.subplot(len(self.num_columns), 3, figNumber)
                data_without_nan = self.data.dropna(subset=[col, target])
                X = data_without_nan[col]
                X_index = X.index
                Y = data_without_nan[target][X_index]
                ax.scatter(x=X, y=Y, s=3)
                ax.set_xlabel(col)
                ax.set_ylabel(target)
                if trend == True:
                    # calc the trendline
                    z = np.polyfit(X, Y, 1)
                    p = np.poly1d(z)
                    ax.plot(X, p(X), "r--")
                figNumber += 1
                plt.grid()
                
    def cat_num_target_box_plot(self, target):
        """
        Построение графиков: x - количестенный признак, 
                             y - количестенный признак
        -------------------------------------------------------------------
        """
        # Разделяем по признаку object и не object
        plt.figure(figsize=[30, 200])   
        plot_number = 1

        for cat_col in self.cat_columns:
            ax = plt.subplot(len(self.cat_columns), 2, plot_number)
            sns.boxplot(x=cat_col, y=target, data=self.data, ax=ax)
            ax.legend(list(self.data[cat_col].unique()))
#             plt.subplots_adjust(hspace=0.2)
#             plt.subplots_adjust(wspace=0.2)
            plt.grid()
            plot_number += 1
        
    def nums_cat_box_plot(self, cat_col):
        plt.figure(figsize=[35, 250])   
        plot_number = 1
        for num_col in self.num_columns:
            ax = plt.subplot(len(self.num_columns), 2, plot_number)
            sns.boxplot(x=cat_col, y=num_col, data=self.data, ax=ax)
            values_data = self.data[~self.data[num_col].isna()][cat_col].value_counts().sort_index()
            dict_values = {}
            for index, value in zip(values_data.index, values_data.values):
                dict_values[index]=f'*{value}*'
            ax.set_title(f'{dict_values}')
            # ax.text(0.02, 0.98, "Text", ha="left", va="top", transform=ax.transAxes)
            plt.grid()
            plot_number += 1
            
    def numTargetHist(self, save_folder: str, target: str, bins: int=50):
        """
        Построение графика: распределение значений количественного признака
        -------------------------------------------------------------------
        bins - кол-во столбцов гистограммы
        """
        labels = self.data[target].unique()
        width = 3*8
        height = np.round(len(self.num_columns) / 3)*15
        plt.figure(figsize=[width, height])
        figNumber = 1
        #self.num_columns.remove(target)
        for num in self.num_columns:
            ax = plt.subplot(len(self.num_columns), 3, figNumber)
            try:
                ax.hist(self.data[self.data[target]==labels[0]][num], bins, alpha=0.5, label=labels[0])
            except:
                pass
            try:
                ax.hist(self.data[self.data[target]==labels[1]][num], bins, alpha=0.5, label=labels[1])
            except:
                pass
            ax.set_xlabel(num)
            ax.set_ylabel('Count')
            ax.legend(loc='upper right')
            figNumber += 1
            plt.grid()
        if save_folder:
            plt.savefig(save_folder)

    def numHistograms(self, bins: int=50):
        """
        Построение графика: распределение значений количественного признака
        -------------------------------------------------------------------
        bins - кол-во столбцов гистограммы
        """
        width = 3*8
        height = np.round(len(self.num_columns) / 3)*15
        plt.figure(figsize=[width, height])
        figNumber = 1
        for num in self.num_columns:
            ax = plt.subplot(len(self.num_columns), 3, figNumber)
            ax.hist(self.data[num], bins, alpha=0.5)
            ax.set_xlabel(num)
            ax.set_ylabel('Count')
            figNumber += 1
            plt.grid()

    def doublecat_count_plot(self):
        """
        Построение графиков: x - классы категориального признака,
                             y - кол-во элементов, 
                             color - классы категориального признака
        -------------------------------------------------------------------
        """
        for cat in self.cat_combo_2:
            fig = plt.figure()
            catplot = sns.catplot(x=cat[0],
                                  hue=cat[1],
                                  data=self.data,
                                  height=5,
                                  aspect=2,
                                  kind="count")
            catplot.fig.suptitle(cat[0]+' / '+cat[1]+' / '+'count', y=1.05)
            plt.grid()
            
    def doublecat_target_plot(self, target: str):
        """
        Построение графиков: x - классы категориального признака,
                             y - кол-во элементов, 
                             color - классы категориального признака
        -------------------------------------------------------------------
        target - название целевой переменной
        """
        for cat in self.cat_columns:
            if cat != target:
                fig = plt.figure()
                catplot = sns.catplot(x=cat,
                                      hue=target,
                                      data=self.data,
                                      height=4,
                                      aspect=2,
                                      kind="count")
                catplot.fig.suptitle(cat+' / '+target+' / '+'count', y=1.05)
                plt.grid()
    
    def triplenum_plot(self):
        """
        Построение графиков: x, y, color - количественные признаки
        -------------------------------------------------------------------
        """
        for num in self.num_combo_3:
            fig = plt.figure()
            plt.figure(figsize=[15, 10], dpi=120)
            plt.scatter(x=self.data[num[0]],
                        y=self.data[num[1]],
                        c=self.data[num[2]], 
                        cmap='viridis', 
                        s=1)
            plt.title(num[0]+' / '+num[1]+' / '+num[2], y=1.05)
            plt.xlabel(num[0])
            plt.ylabel(num[1])
            plt.colorbar(label=num[2])
    
    def num_cat_dencity_plot(self):
        """
        Построение графиков: x - количестенный признак,
                             y - плотность,
                             color - классы категориального признака
        -------------------------------------------------------------------
        """
        for num, cat in list(itertools.product(self.num_columns, self.cat_columns)):
                plt.figure(dpi=100)
                sns.kdeplot(data=self.data, x=num, hue=cat, bw=.5)
                plt.title(num+' / '+cat+' / '+'dencity')
                plt.grid()
                
    def nums_cat_dencity_plot(self, cat_col):
        f = plt.figure()
        f.set_figheight(120)
        f.set_figwidth(30)
        plot_number = 1
        cols = ceil(len(self.num_columns) / 2)
        for num_col in self.num_columns:
            f.add_subplot(cols, 2, plot_number)
            sns.kdeplot(data=self.data, x=num_col, hue=cat_col)
            plot_number+=1
                
    def num_num_cat_plot(self):
        """
        Построение графиков: x - количестенный признак, 
                             y - количестенный признак, 
                             color - классы категориального признака
        -------------------------------------------------------------------
        """
        for num, cat in list(product(self.num_combo_2, self.cat_columns)):
                plt.figure(figsize=[12, 8], dpi=80)
                data_ = self.data[[num[0], num[1], cat]]
                sns.scatterplot(x=num[0],
                                y=num[1],
                                hue=cat,
                                data=data_)
                plt.title(num[0]+' / '+num[1]+' / '+cat, y=1.05)

    def cat_cat_num_plot(self):
        """
        Построение графиков: x - классы категориального признака, 
                             y - количестенный признак, 
                             color - классы категориального признака
        -------------------------------------------------------------------
        """
        for cat, num in list(itertools.product(self.cat_combo_2, self.num_columns)):
                plt.figure()
                g = sns.catplot(x=cat[0],
                                y=num,
                                hue=cat[1],
                                data=self.data,
                                height=5,
                                aspect=2)
                g.fig.suptitle(cat[0]+' / '+cat[1] + ' / '+num, y=1.05)

    def cat_histograms(self):
        """
        Построение графиков: x - классы категориального признака, 
                             y - кол-во элементов
        -------------------------------------------------------------------
        """
        plt.figure(figsize=[20, 25], dpi=70)
        i = 1
        for cat in self.cat_columns:
            ax = plt.subplot(len(self.cat_columns), 2, i)
            sns.countplot(x=cat,data=self.data)
            ax.set_xlabel(cat)
            ax.set_ylabel('Count')
            ax.set_title ('Count of categories for {}'. format(cat), fontsize = 15)
            plt.subplots_adjust(hspace = 1)
            i += 1

    def num_histograms(self, bins: int=50):
        """
        Построение графика: распределение значений количественного признака
        -------------------------------------------------------------------
        bins - кол-во столбцов гистограммы
        """
        #plt.figure(figsize=[10, 15])
        i = 1
        for num in self.num_columns:
            ax = plt.subplot(len(self.num_columns), 3, i)
            data[num].hist(bins)
            ax.set_xlabel(num)
            ax.set_ylabel('Count')
            i += 1
    
    def time_moving_average_plot(self, columns: List[str], n_roll: int):
        """
        Построение графика: временной ряд со скользщяим средним
        -------------------------------------------------------------------
        columns - название отображаемых временных рядов
        n_roll - величина окна
        """
        fig = go.Figure()

        for prod in columns:
            fig.add_trace(go.Scatter(x=self.data.index,
                                     y=self.data[prod],
                                     name=prod))
            fig.add_trace(go.Scatter(x=self.data.index,
                                     y=self.data[prod].rolling(window=n_roll).mean(),
                                     name=prod+'_smoothing'))
            
            fig.update_layout(yaxis_title="Sales", xaxis_title="Time",
                          title=f"Sales for {columns} with average_smoothing ({n_roll} {self.data.index.inferred_freq})",
                          template='plotly_white',
                          xaxis=dict(
                              rangeselector=dict(
                                      buttons=list([
                                          dict(count=6, label='6m',
                                               step='month', stepmode='backward'),
                                          dict(count=12, label='12m',
                                               step='month', stepmode='backward'),
                                          dict(count=18, label='18m',
                                               step='month', stepmode='backward'),
                                          dict(step='all')])), rangeslider=dict(visible=True), type='date'))
        return fig
    
    def time_average_plot(self, columns: List[str], average_period: str):
        """
        Построение графика: временной ряд с группировкой по периоду
        -------------------------------------------------------------------
        columns - название отображаемых временных рядов
        average_period - название периода (day, month, year)
        """
        fig = go.Figure()
        example = self.data.groupby(by=[average_period]).mean()
    
        for prod in columns:
            fig.add_trace(go.Scatter(x=example.index,
                                     y=example[prod],
                                     mode='lines',
                                     name=prod))
        
        fig.update_layout(yaxis_title="Sales",
                          xaxis_title=average_period,
                          title=f'Mean sales for {columns} per {average_period}',
                          template='plotly_white')
        return fig
    
    def time_boxplot(self, columns: List[str]):
        """
        Построение графика box-plot временного ряда
        -------------------------------------------------------------------
        columns - название отображаемых временные ряды
        """        
        fig = go.Figure()

        for prod in columns:
            fig.add_trace(go.Box(y=self.data[prod], name=prod))
        
        fig.update_layout(yaxis_title="Sales",
                          xaxis_title='Categories',
                          title=f'Sales for {columns}',
                          template='plotly_white')
        return fig
    
    def time_exp_plot(self, columns: List[str], alpha: float):
        """
        Построение графика временного ряда с экспоненциальным сглаживанием
        -------------------------------------------------------------------
        columns - названия отображаемых временных рядов
        alpha - коэффициент экспоненциального сглаживания
        """
        fig = go.Figure()

        def exponential_smoothing(series, alpha):
            result = [series[0]] # first value is same as series
            for n in range(1, len(series)):
                result.append(alpha * series[n] + (1 - alpha) * result[n-1])
            return pd.Series(result, index=series.index)
    
        for prod in columns:
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=self.data[prod], 
                                     name=prod))
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=exponential_smoothing(self.data[prod], alpha), 
                                     name=prod+f'_smoothing (a={alpha})'))

        fig.update_layout(yaxis_title="Sales", xaxis_title="Time",
                      title=f"Sales for {columns} with exp_smoothing",
                      template='plotly_white',
                      xaxis=dict(
                          rangeselector=dict(
                                  buttons=list([
                                      dict(count=6, label='6m',
                                               step='month', stepmode='backward'),
                                      dict(count=12, label='12m',
                                           step='month', stepmode='backward'),
                                      dict(count=18, label='18m',
                                           step='month', stepmode='backward'),
                                      dict(step='all')])), rangeslider=dict(visible=True), type='date'))

        return fig
    
    def time_doubleexp_plot(self, columns: List[str], alpha: float, beta: float):
        """
        Построение графика временного ряда с двойным экспоненциальным 
        сглаживанием
        -------------------------------------------------------------------
        columns - названия отображаемых временных рядов
        alpha - коэффициент экспоненциального сглаживания
        beta - коэффициент экспоненциального сглаживания
        """       
        fig = go.Figure()
    
        def double_exponential_smoothing(series, alpha, beta):
            result = [series[0]]
            for n in range(1, len(series)):
                if n == 1:
                    level, trend = series[0], series[1] - series[0]
                if n >= len(series):
                    value = result[-1]
                else:
                    value = series[n]
                last_level, level = level, alpha*value + (1-alpha)*(level+trend)
                trend = beta*(level-last_level) + (1-beta)*trend
                result.append(level+trend)
        
            return pd.Series(result, index=series.index)

        for prod in columns:
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=self.data[prod], 
                                     name=prod))
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=double_exponential_smoothing(self.data[prod], alpha, beta), 
                                     name=prod+f'_smoothing (a={alpha}, b={beta})'))

        fig.update_layout(yaxis_title="Sales", 
                          xaxis_title="Time",
                          title=f"Sales for {columns} with doubleexp_smoothing",
                          template='plotly_white',
                          xaxis=dict(
                              rangeselector=dict(
                                  buttons=list([
                                      dict(count=6, label='6m',
                                               step='month', stepmode='backward'),
                                      dict(count=12, label='12m',
                                           step='month', stepmode='backward'),
                                      dict(count=18, label='18m',
                                           step='month', stepmode='backward'),
                                      dict(step='all')])), rangeslider=dict(visible=True), type='date'))

        return fig
