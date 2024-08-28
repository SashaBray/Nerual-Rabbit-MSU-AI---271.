import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import alive_progress
from alive_progress import alive_bar
import copy

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statistics

import torch
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

import json
import requests

import random
import math

from Library_working_with_time_series import TimeRow, Operations

import pickle

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from typing import Tuple

import cProfile

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression



class CalculationTask():
    """Класс является носителем информации о расчетном задании"""
    def __init__(self, task_file=None, years=None, prediction_years_iteration = 0, target=None, success=None, directory = 'Destination_files\\'.replace('\\', os.sep),
                 column_predict=None, column_models=None, models_list = None,
                 task_file_reserw=None, login=None, password=None,
                 culture=None, years_to_back=None, current_day=None, avegage_last=None, years_for_correct=1,
                 prediction_years_secret=None, correction=False) -> None:

        self.task_file = task_file
        self.prediction_years = years
        self.prediction_years_iteration = prediction_years_iteration
        self.target = target
        self.success = success
        self.directory = directory

        self.column_predict = column_predict
        self.column_models = column_models
        self.models_list = models_list

        self.task_file_reserw = task_file_reserw

        self.login = login
        self.password = password
        self.culture = culture

        self.years_to_back = years_to_back
        self.current_day = current_day
        self.avegage_last = avegage_last
        self.years_for_correct = years_for_correct
        self.prediction_years_secret = prediction_years_secret
        self.correction = correction


class ModelWrapper():
    """Класс обертка для моделей"""
    def __init__(self, model_name,
                 parms_file = None,
                 timeline_file = None,
                 task=None,
                 masks=None):

        self.model_name = model_name
        self.parms_file = parms_file
        self.timeline_file = timeline_file
        self.task = task
        self.masks = masks

    def activate(self, directory='Service_files\Models\\'.replace('\\', os.sep),
                 parm_file='\parms.json', prossesing='\prossesing.json'.replace('\\', os.sep),
                 masks_directory="Service_files\Common_information\\".replace('\\', os.sep), masks_file='Cultures_masks.xlsx'):

        way = directory + self.model_name

        file_parms_way = way + parm_file
        file_prossesing_way = way + prossesing

        print('file_parms_way', file_parms_way)

        self.parm_file_info = read_inf(file_parms_way.replace(' ', ''))
        self.prossesing_file_info = read_inf(file_prossesing_way.replace(' ', ''))

        self.parms = self.parm_file_info['parm_list']
        self.prossesing = self.prossesing_file_info

        try:
            self.parms_scalars = self.parm_file_info["scalars"]
        except:
            self.parms_scalars = None

        self.way_to_masks = masks_directory + masks_file
        print('self.way_to_masks', self.way_to_masks)
        self.pd_masks = pd.read_excel(self.way_to_masks, index_col='Unnamed: 0')

        self.first_day, self.last_day = self.parm_file_info["time_borders"][0], self.parm_file_info["time_borders"][1]
        self.normalization_parms = self.parm_file_info["normalization"]

        print('model.name', self.model_name)
        print('bool', ('LM' in self.model_name) and not('KML' in self.model_name))

        # try:
        if 'NN_CN' in self.model_name: # self.model_name == 'NN_CN_v1_winter_wheat':
            model_file_name = '\\model_scripted.pt'.replace('\\', os.sep)
            model = torch.jit.load(way + model_file_name)

        if 'RF' in self.model_name: # self.model_name == 'RF_v1_winter_wheat':
            # way = 'Service_files\\Models\\'.replace('\\', os.sep) + self.model_name
            model_file_name = '\\my_model.pickle'.replace('\\', os.sep)
            model = pickle.load(open(way + model_file_name, "rb"))
            print(model)

        if ('LM' in self.model_name) and not('KML' in self.model_name):
            model = Simple_linear_model()
            print('Выбрана простая линейная модель')

        if ('KML' in self.model_name):
            if 'LM' in self.model_name:
                model = Complex_linear_model()
            if 'RF' in self.model_name:
                model = Pretrained_Random_Forest()


        if 'EXM' in self.model_name:
            model = Simple_exp_model()
            print('Выбрана простая экспоненциальная модель')

        self.model = model
        print('Активация модели успешно произведена!')
        pass

        if model == None:
            print('Ошибка! Модель не найдена! Проверьте корректность введенного в таблицу имени модели.')
        else:
            print('Модель успешно активирована!')



class Answer():
    """Класс ответа. Хранит вердикт и дополнительную информацию, которую нужно вывести в отчет"""
    def __init__(self, verdict: [float], additional_inf={}, inf_unconditional={}):
        self.verdict = verdict
        self.additional_inf = additional_inf
        self.inf_unconditional = inf_unconditional

class DataPrepearer():
    """Класс, принимающий на вход данные, полученные на сайте Vega и преобразует их в потребную для модели форму,
    производит отсечение нужного диапазона, нормализацию, усреднение, введение рядов с константой"""
    def __init__(self, model_name, parms_file=None, timeline_file=None, input_type=None):
        self.model_name = model_name
        self.parms_file = parms_file
        self.timeline_file = timeline_file
        self.input_type = input_type

    def activate(self):
        """Приведение экземпляра в готовность"""
        pass

    def transform_data(self):
        """Трансформация данных в нужный вид"""
        pass

class VegaAPI():
    """Класс для запроса дынных на сайте Vega-Scince"""
    def __init__(self, login, password):
        self.login = login
        self.password = password

    def connect(self):
        self.user = ['Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36',
                'Mozilla/5.0 (iPad; CPU OS 6_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/6.0 Mobile/10A5376e Safari/8536.25',
                'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
                'Opera/9.80 (Windows NT 6.2; WOW64) Presto/2.12.388 Version/12.17']  # Лист фальшивых имен
        index = random.randint(0, len(self.user) - 1)
        self.header = {'User-Agent': self.user[index], 'Content-Type': 'application/x-www-form-urlencoded', 'charset': 'UTF-8'}
        self.session = requests.Session()

            # Ссылка на сайт
        self.URL = 'http://sci-vega.ru/login/login.pl'

            # Логин и пароль
        Login = 'fki22_SB'
        Password = 'MAILoveforever2022'
        # Login = self.password
        # Password = self.password

        self.data = {'first': Login, 'second': Password, 'mode': 'in', 'from': '/'}
        self.response = self.session.post(self.URL, data=self.data, headers=self.header)


    def district(self, product_type, year, reg_uid):
        # print(product_type, year, reg_uid, ' Loading in progress...')  # Выводим на экран
        success = False
        try:  # Пробуем запросить данные
            product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)

            Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&x2=366&query=' \
                              +'[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + reg_uid + '%22,%22rows%22:{%22' \
                              + product_type + '%22:[' + year + ']}}]&mode=basic&num_points=1&highcharts=1&a_week=51&a_year=' \
                              + year + '&label_year=' + year

            print('Save_graph_link', Save_graph_link)
            Graph = self.session.get(Save_graph_link).text
            # print('Graph', Graph)
            Result = json.loads(Graph)
            success = True

        except:  # Ошибка скорее всего значит завершение сессии. Нужно зайти снова и повторить запрос
            try:
                self.session = requests.Session()
                self.response = self.session.post(URL, data=data, headers=header)

                Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&x2=366&query=' \
                                  + '[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + reg_uid + '%22,%22rows%22:{%22' \
                                  + product_type + '%22:[' + year + ']}}]&mode=basic&num_points=1&highcharts=1&a_week=51&a_year=' \
                                  + year + '&label_year=' + year

                print('Save_graph_link', Save_graph_link)
                Graph = self.session.get(Save_graph_link).text
                Result = json.loads(Graph)
                success = True

            except:
                print('No internet...')
                Result = None

        return Result, success

    def region(self, product_type, year, reg_uid):
        # print(product_type, year, reg_uid, ' Loading in progress...')  # Выводим на экран
        success = False
        a_week = '51'

        try:  # Пробуем запросить данные

            product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)

            print('product_type, year, reg_uid', product_type, year, reg_uid)

            a_week = '51'

            Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&' \
                              + 'x2=366&query=[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22' \
                              + reg_uid + '%22,%22rows%22:{%22reg_' + product_type + '%22:[' + year + ']}}]&mode' \
                              + '=basic&num_points=1&highcharts=1&no_cache=34576&a_week=' + a_week \
                              + '&a_year=' + year + '&label_year=' + year


            print('Save_graph_link', Save_graph_link)
            # print('type Save_graph_link', type(Save_graph_link))

            Graph = self.session.get(Save_graph_link).text
            # print('Graph', Graph)
            Result = json.loads(Graph)
            success = True

        except:  # Ошибка скорее всего значит завершение сессии. Нужно зайти снова и повторить запрос
            try:
                self.session = requests.Session()
                self.response = self.session.post(URL, data=data, headers=header)
                Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&'\
                                  + 'x2=366&query=[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22'\
                                  + reg_uid + '%22,%22rows%22:{%22reg_' + product_type + '%22:[' + year + ']}}]&mode'\
                                  + '=basic&num_points=1&highcharts=1&no_cache=34576&a_week=' + a_week\
                                  + '&a_year=' + year + '&label_year=' + year


                print('Save_graph_link', Save_graph_link)
                Graph = self.session.get(Save_graph_link).text
                Result = json.loads(Graph)
                success = True

            except:
                print('No internet...')
                Result = None

        return Result, success


def print_metrics(y_true, y_predicted, x_train, answer) -> Answer:

    print('')
    print('true_for_metrics', y_true)
    print('predicted_for_metrics', y_predicted)
    print('')

    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    x_shape, y_shape = max(y_true.shape), min(y_true.shape)

    print(f"Mean squared error: {mean_squared_error(y_true, y_predicted):.3f}")
    print(
        "Root mean squared error: ",
        f"{mean_squared_error(y_true, y_predicted, squared=False):.3f}",
    )
    print(f"Mean absolute error: {mean_absolute_error(y_true, y_predicted):.3f}")
    print(f"R2 score: {r2_score(y_true, y_predicted):.3f}")

    answer.inf_unconditional.setdefault('R2 score', r2_score(y_true, y_predicted))      # Расчет основных метрик
    answer.inf_unconditional['R2 score'] = r2_score(y_true, y_predicted)

    answer.inf_unconditional.setdefault('Mean squared error', mean_squared_error(y_true, y_predicted))
    answer.inf_unconditional['Mean squared error'] = mean_squared_error(y_true, y_predicted)

    answer.inf_unconditional.setdefault('Root mean squared error', mean_squared_error(y_true, y_predicted, squared=False))
    answer.inf_unconditional['Root mean squared error'] = mean_squared_error(y_true, y_predicted, squared=False)

    answer.inf_unconditional.setdefault('Max absolute error', abs(y_true.reshape(-1, x_shape) - y_predicted.reshape(-1, x_shape)).max())
    answer.inf_unconditional['Max absolute error'] = abs(y_true.reshape(-1, x_shape) - y_predicted.reshape(-1, x_shape)).max()

    answer.inf_unconditional.setdefault('Mean absolute error', mean_absolute_error(y_true, y_predicted))
    answer.inf_unconditional['Mean absolute error'] = mean_absolute_error(y_true, y_predicted)

    try:       # Расчет и запись коэффициента корреляции
        print('Расчет и запись коэффициента корреляции', y_true, y_predicted)
        correleatio_factor = statistics.correlation(y_true, y_predicted)
        answer.inf_unconditional.setdefault('Pearson correlation (real productove and predicted)', correleatio_factor)
        answer.inf_unconditional['Pearson correlation (real productove and predicted)'] = correleatio_factor
    except:
        print('Недостаточно точек для рассчета коэффициента кореляции!')
        answer.inf_unconditional.setdefault('Pearson correlation (real productove and predicted)', None)
        answer.inf_unconditional['Pearson correlation (real productove and predicted)'] = None

    print('answer -', answer.inf_unconditional)

    return answer


def print_settlement_information(answer, data, target, task):
    '''[maximums, prods, target_arg, years]'''

    answer.additional_inf = {}
    years = data[3]
    maximums = data[0]

    headline = task.prediction_years[task.prediction_years_iteration] + '_ndvi_max'
    value = target
    answer.additional_inf.setdefault(headline, value)

    for i in range(len(years)):
        year = years[i]
        headline = str(year) + '_ndvi_max'
        value = maximums[i]
        answer.additional_inf.setdefault(headline, value)

    return answer


def create_years_ndvi_line_graphic(data, task, row, way='Report_files\\Graphs\\'.replace('\\', os.sep), file_name='New_graphic', vector=True):
    '''Функция получает информацию и создает гарфик изменения ndvi и урожайности в разные годы'''

    YEARS = copy.deepcopy(data[3]) # Заводим массив годов
    # if not int(task.prediction_years[task.prediction_years_iteration]) in YEARS:
    #     YEARS.append(int(task.prediction_years[task.prediction_years_iteration]))  # Если в списке нет изучаемого года, он добавляется

    NDVI_YEARS = copy.deepcopy(data[0])    # Заводим переменную с макс знач ndvi

    # print('heheh', int(task.prediction_years[task.prediction_years_iteration]) in NDVI_YEARS)

    if not int(task.prediction_years[task.prediction_years_iteration]) in YEARS:
        NDVI_YEARS_interim = [data[2]]
        YEARS_interim = [int(task.prediction_years[task.prediction_years_iteration])]

        # NDVI_YEARS_interim.append(NDVI_YEARS)  # Если в списке нет изучаемого года, добавляем ndvi этого года
        # YEARS_interim.append(YEARS)

        NDVI_YEARS = NDVI_YEARS_interim + NDVI_YEARS
        YEARS = YEARS_interim + YEARS

    PROD = copy.deepcopy(data[1])  # График урожайности
    P_YEARS = copy.deepcopy(data[3])  # Годы под урожайность



    print('YEARS     ', YEARS, len(YEARS))
    print('NDVI_YEARS', NDVI_YEARS, len(NDVI_YEARS))

    print('P_YEARS', P_YEARS, len(P_YEARS))
    print('PROD   ', PROD, len(PROD))

    YEARS = [str(item) for item in YEARS]
    P_YEARS = [str(item) for item in P_YEARS]

    print('str YEARS     ', YEARS, len(YEARS))
    print('str_P_YEARS', P_YEARS, len(P_YEARS))

    YEARS.reverse()     # Разворачиваем все листы, чтобы соблюсти порядок
    P_YEARS.reverse()
    NDVI_YEARS.reverse()
    PROD.reverse()


    fig, ax1 = plt.subplots(figsize=(12, 6))  # График урожайности и пика NDVI
    ax2 = ax1.twinx()
    ax1.plot(YEARS, NDVI_YEARS, label=' NDVI ', color='blue')
    ax2.plot(P_YEARS, PROD, label=' Урожайность ', color='darkred')
    ax1.set_ylabel('NDVI', color='blue', size=11)
    ax2.set_ylabel('Урожайность (ц/га)', color='darkred', size=11)

    if str(row['district']) == 'nan':
        plt.title('NDVI/Урожайность по годам: ' + row['region'], size=14)
    else:
        plt.title('NDVI/Урожайность по годам: ' + row['region'] + ' - ' + row['district'], size=14)

    fig.legend(bbox_to_anchor=(0.65, 0.07), ncol=2, fontsize=11)
    plt.grid(True)

    if vector:
        pdf = PdfPages(way + file_name + '.pdf')
        pdf.savefig()
        pdf.close()

    else:
        plt.savefig(way + file_name + '.jpg')
        plt.close()
    pass


def get_regression_line_for_graph(ndvi, prod, regressor):

    prod = np.array(prod)
    ndvi = np.array(ndvi)

    ndvi_min = ndvi.min()
    ndvi_max = ndvi.max()

    n = (ndvi_max - ndvi_min) / 20
    n = 1 if n==0 else n

    points = np.arange(ndvi_min - 10 * n, ndvi_max + 10 * n, n)

    regression_points = []
    for i in range(len(points)):
        point = np.array(points[i]).reshape(1, -1)
        regression_point_i = regressor.predict(point).reshape(1)
        regression_points.append(regression_point_i)

    print('regression_points', regression_points)

    return points, regression_points


def create_years_ndvi_point_graphic(data, task, row, regressor, way, file_name, vector=True):
    '''Процедура получает информацию и создает гарфик изменения ndvi и урожайности в разные годы'''
    #   [maximums, prods, target_arg, years]

    YEARS = copy.deepcopy(data[3])  # Заводим массив годов # years
    NDVI_YEARS = copy.deepcopy(data[0])  # Заводим переменную с макс знач ndvi # maximums
    PROD = copy.deepcopy(data[1])  # График урожайности # prods
    P_YEARS = copy.deepcopy(data[3])  # Годы под урожайность # years

    print('YEARS     ', YEARS, len(YEARS))
    print('NDVI_YEARS', NDVI_YEARS, len(NDVI_YEARS))
    print('P_YEARS', P_YEARS, len(P_YEARS))
    print('PROD   ', PROD, len(PROD))
    print('str YEARS     ', YEARS, len(YEARS))
    print('str_P_YEARS', P_YEARS, len(P_YEARS))

    YEARS.reverse()  # Разворачиваем все листы, чтобы соблюсти порядок
    P_YEARS.reverse()
    NDVI_YEARS.reverse()
    PROD.reverse()

    x = NDVI_YEARS
    y = PROD

    points_arg, points_regression = get_regression_line_for_graph(NDVI_YEARS, PROD, regressor)

    plt.figure(figsize=(10, 6))  # Гарфик линейной регрессии
    plt.scatter(x, y, label=' NDVI ', color='blue')
    plt.plot(points_arg, points_regression, label=' Линейная регрессия ', color='darkred')
    plt.xlabel('NDVI max', size=11)
    plt.ylabel('Урожайность (ц/га) ', size=11)

    if str(row['district']) == 'nan':
        plt.title('Линейная регрессия: ' + row['region'] , size=14)
    else:
        plt.title('Линейная регрессия: ' + row['region'] + ' - ' + row['district'], size=14)
    plt.grid(True)

    if vector:
        pdf = PdfPages(way + file_name + '.pdf')
        pdf.savefig()
        pdf.close()

    else:
        plt.savefig(way + file_name + '.jpg')
        plt.close()


def make_graphs_to_illustrate_simple_linear_model(row, regressor, task, data):
    district = row['district']
    region = row['region']
    region_id = row['id_region']
    district_id = row['id_district']

    if str(district_id) == 'nan':
        district_id = '-'

    if str(district) == 'nan':
        district = '-'

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'line_graph' + '_' + 'SLM'

    way = 'Report_files\\Graphs\\PDF\\'.replace('\\', os.sep)
    create_years_ndvi_line_graphic(data, task, row, way=way, file_name=linear_graph_name, vector=True)

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'line_graph' + '_' + 'SLM'

    way = 'Report_files\\Graphs\\JPG\\'.replace('\\', os.sep)
    create_years_ndvi_line_graphic(data, task, row, way=way, file_name=linear_graph_name, vector=False)

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'point_graph' + '_' + 'SLM'
    way = 'Report_files\\Graphs\\PDF\\'.replace('\\', os.sep)
    create_years_ndvi_point_graphic(data, task, row, regressor, way=way, file_name=linear_graph_name, vector=True)

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'point_graph' + '_' + 'SLM'
    way = 'Report_files\\Graphs\\JPG\\'.replace('\\', os.sep)
    create_years_ndvi_point_graphic(data, task, row, regressor, way=way, file_name=linear_graph_name, vector=False)


class Simple_linear_model():
    """Класс простой линейно-регрессионной модели"""

    def predict(self, data, task, row):
        print('predict is on', data)

        # if data[0].shape[0] >= 2:
        if len(data[0]) >= 2:
            # [maximums, prods, target_arg]

            # x_train_scaled = data['x_train']
            # y_train = data['y_train']
            # x_target_scaled = data['x_target_rows']

            x_train_scaled = np.array(data[0]).reshape(-1, 1)
            y_train = np.array(data[1]).reshape(-1, 1)
            x_target_scaled = np.array(data[2]).reshape(-1, 1)

            # print('')
            # print('x_train_scaled', x_train_scaled.shape)
            # print('y_train', y_train.shape)
            # print('x_target_scaled', x_target_scaled.shape)

            regressor = LinearRegression()
            regressor.fit(x_train_scaled, y_train)

            y_pred_s = regressor.predict(x_target_scaled)
            y_pred = regressor.predict(x_train_scaled)

            print('argument', x_target_scaled)
            print('answer', y_pred_s)

            x, y = x_train_scaled, y_train

            answer = Answer(verdict=y_pred_s)

            answer = print_metrics(y_train, y_pred, x_train_scaled, answer)
            answer = print_settlement_information(answer, data, x_target_scaled, task)

            make_graphs_to_illustrate_simple_linear_model(row, regressor, task, data)

        else:
            print('Недостаточно точек для построения регрессий!')
            answer = Answer(verdict=None)

        return answer


class Complex_linear_model():
    '''Класс многопараметрической линейной модели'''

    def predict(self, data, task, row) -> Answer:
        print('predict is on', data)

        x_train_rows = data['x_train']
        y_train = data['y_train'].reshape(-1, 1)
        x_target_rows = data['x_target']
        y_train_mean_prods = data['y_train_mean_prods']
        y_train_deviation = data['deviation'].reshape(-1, 1)
        target_mean = data['target_mean']

        print('Is x_target in x_train?', x_target_rows.tolist() in x_train_rows.tolist())
        print('x_target_rows', x_target_rows)
        print('x_target_rows.shape', x_target_rows.shape)
        print('x_train_rows', x_train_rows)
        print('x_train_rows.shape', x_train_rows.shape)

        scaler = preprocessing.StandardScaler().fit(x_train_rows)
        x_train_rows = scaler.transform(x_train_rows)
        x_target_rows = scaler.transform(x_target_rows)

        print('')
        print('x_train_rows', x_train_rows.shape)
        print('y_train', y_train_deviation.shape)
        print('y_train_deviation', y_train_deviation)
        print('x_target_rows', x_target_rows.shape)

        # regressor = LinearRegression()      # Инициализируем модель
        # regressor.fit(x_train_rows, y_train_deviation)    # Обучаем модель

        depth = 200
        regressor = RandomForestRegressor(n_estimators=50, max_depth=depth, random_state=0)
        regressor.fit(x_train_rows, y_train_deviation)  # Обучаем модель

        y_pred_s = regressor.predict(x_target_rows)   # Предсказываем конкретное значение
        y_pred = regressor.predict(x_train_rows)    # Предсказываем все прочие решения, чтобы оценить точность

        print(y_pred.shape, y_train_mean_prods.shape)

        # y_pred = y_pred.reshape(y_train_mean_prods.shape) * y_train_mean_prods
        y_pred = y_train_mean_prods + y_pred.reshape(y_train_mean_prods.shape) * y_train_mean_prods

        print('y_pred_after *', y_pred)
        print('y_pred_after *', y_pred.shape)

        print('y_pred_model', y_pred)
        print('y_train_model', y_train_deviation)

        print('argument', x_target_rows)
        print('answer', y_pred_s)

        print('y_train', y_train)

        y_pred_s = target_mean + y_pred_s * target_mean

        answer = Answer(verdict=y_pred_s)       # Инициализируем экземпляр класса Ответ
        answer = print_metrics(y_train, y_pred, None, answer)     # Прикладываем к ответу дополнительную информацию

        return answer


class Simple_exp_model():
    """Класс простой линейно-регрессионной модели"""

    def predict(self, data):
        # [maximums, prods, target_arg]
        print('data[0]', data[0])
        print('data[1]', data[1])
        x_train_scaled = np.array(data[0]).reshape(-1, 1)
        y_train = np.array(data[1]).reshape(-1, 1)

        # x_train_scaled = np.log(x_train_scaled)
        y_train = np.log(y_train)

        print('y_train_log', y_train)
        print('x_train_scaled', x_train_scaled)

        x_test_scaled = np.array([2]).reshape(-1, 1)

        print('x_train_scaled.shape', x_train_scaled.shape)

        regressor = LinearRegression()
        regressor.fit(x_train_scaled, y_train)

        y_pred = regressor.predict(x_test_scaled)
        # y_pred = np.exp(y_pred)
        y_pred = 2.718281828**y_pred

        return y_pred


def write_inf(data, file_name):
    data = json.dumps(data)
    data = json.loads(str(data))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
         return json.load(file)

def scenario_1(task):
    directory = task.directory
    files = os.listdir(directory)

    while True:
        print('Укажите файл с расчетным заданием: ')
        print(files)
        print('или')
        print("['Назад']")

        phrase = input()

        if phrase.replace("'", '') in files:
            print('Имя файла записано!', phrase)
            task.task_file = phrase
            # task.write_task_file(phrase)
            print('Gaga', task.task_file)
            break
        elif phrase.replace(' ', '').replace("'", '') == 'Назад':
            break
        else:
            'Команда не распознана! \n'

    return task


def make_design(phrase: [str], task):       # Обработка входящих команд

    command_1 = '     Сделать прогноз по расчетному заданию '
    command_2 = '     Завершить работу'

    while True:

        if phrase.replace(' ', '') == command_1.replace(' ', ''):   # На ввод файла с заданием
            task = scenario_1(task)
            break

        elif phrase.replace(' ', '') == command_2.replace(' ', ''): # Завершение работы
            # task = scenario_2(task)
            break
        else:
            print('Команда не распознана...', '\n')
    return task


def сonduct_user_survey():
    availability_survey = True
    task = CalculationTask()
    command_2 = '     Завершить работу'

    while availability_survey:
        print('Введите команду:') # \n')
        print('     Сделать прогноз по расчетному заданию ')
        print('     Завершить работу')

        phrase = input()
        if phrase != command_2:
            task = make_design(phrase, task)
            break
        elif phrase == command_2:
            break
        else:
            print('Команда не распознана...', '\n')
    return task


def fast_user_survey():
    """Инициализация экземпляра класса задания с указанием расчетного файла"""
    task = CalculationTask(task_file='Destination template.xlsx', login='fki22_SB',
                            password='MAILoveforever2022', culture='озимая пшеница',
                           years_to_back=7, current_day=171, avegage_last=5,
                           years_for_correct=4, correction=False)
    return task


def get_models_list(df_destination) -> list:
    """Получение списка моделей, упомянутых в файле"""
    column_names = df_destination.columns.tolist()
    column_predict = [] # Список заголовков под предсказание
    column_models = [] # Список моделей
    years = [] # Список годов
    print('Заголовки в файле задания', column_names)

    for name in column_names:
        if '_predict_' in str(name):
            column_predict.append(name)
    print('Заголовки под предсказание', column_predict)

    for name in column_predict:
        model_name = name.split('_predict_')[0] # Записываем текст до маякового слова '_predict_'
        year = name.split('_predict_')[-1] # Записываем текст после маякового слова '_predict_'
        column_models.append(model_name)
        years.append(year)

    print('Указанные модели', column_models)
    print('Указанные годы', years, '\n')

    return column_predict, column_models, years


def make_reserw_copy_destination(task, keyword = '_reserw'):
    """Создание резервной копии задания"""
    way = task.directory + task.task_file
    df_destination = pd.read_excel(way, index_col='Unnamed: 0')

    extension = '.xlsx'
    reserw_way = task.directory + task.task_file.replace(extension, '') + keyword + extension
    df_destination.to_excel(reserw_way)

    task.task_file_reserw = task.task_file.replace(extension, '') + keyword + extension       # Записываем в класс сведения о положении резервной копии

    return task


def destination_analysis(task):
    'Предварительный анализ файла с заданием. Выяснение года предсказания, получение списка моделей, упомянутых в файле'
    way = task.directory + task.task_file       # Путь файлв str
    df_destination = pd.read_excel(way, index_col='Unnamed: 0')     # Загрузка файла с заданием
    # print(df_destination)
    task.column_predict, task.column_models, task.prediction_years = get_models_list(df_destination)    # Вычитываем в файле указанные модели

    task = make_reserw_copy_destination(task)      # Создаем резервную копию на всякий случай

    return task


def model_Loader(task):
    """Создаются и сохраняются в экземпляр класса задания списки экземпляров класса, служащего оберткой для моделей"""
    models_name_list = task.column_models
    models_list = []

    for model_name in models_name_list:
        model = ModelWrapper(model_name)
        models_list.append(model)

    task.models_list = models_list

    return task


def chek_file(parm, row, task):           # Потом допишу...
    way = 'Temporary_information\\TimeRows\\'.replace('\\', os.sep)

    if str(row['id_district']) != 'nan':
        id_dist = str(int(row['id_district']))
    else:
        id_dist = str(row['id_district'])

    if str(row['id_region']) != 'nan':
        id_region = str(int(row['id_region']))
    else:
        id_region = str(row['id_region'])

    if 'historical' in parm:
        year = 'interannual'
    else:
        year = task.prediction_years[task.prediction_years_iteration]

    if 'historical' in parm:
        file_name = str(parm).replace('_historical', '') + '_' + str(id_region) + '_' + str(id_dist) + '_' + str(year) #+ '.json'
    else:
        file_name = str(parm) + '_' + str(id_region) + '_' + str(id_dist) + '_' + str(year) #+ '.json'


    print('parm', parm)
    print('parm_file', file_name)

    files = os.listdir(way)

    print('file_name', file_name)
    print('way', way)
    # print('files', files)

    file_name = file_name + '.json'

    if file_name in files:
        file = read_inf(way + file_name)
        try:
            open_file_for_chek = file['data']['1']['xy'][0]

            return True
        except:
            return False

    else:
        print('Файл', file_name, 'не найден!')
        return False


def read(parm, row, task, time_rows_size=365):
    directory = 'Temporary_information\\TimeRows\\'.replace('\\', os.sep)

    if str(row['id_district']) != 'nan':
        id_dist = str(int(row['id_district']))
    else:
        id_dist = str(row['id_district'])

    if str(row['id_region']) != 'nan':
        id_region = str(int(row['id_region']))
    else:
        id_region = str(row['id_region'])

    if 'hist' in parm:
        year = 'interannual'
        file_name = str(parm).replace('_historical', '') + '_' + str(id_region) + '_' + str(id_dist) + '_' + str(year)
        print('Открытие файл...', file_name)

    else:
        year = task.prediction_years[task.prediction_years_iteration]
        file_name = str(parm) + '_' + str(id_region) + '_' + str(id_dist) + '_' + str(year)
        print('Открытие файл...', file_name)

    time_data, _ = TimeRow(directory + file_name).get_array_by_size(my_size=time_rows_size,
                                                                    # Открытие файла и приведение к нужному размеру
                                                                    padding='closest_value')

    return time_data

def vega_product(parm: str, ndvi: bool, historical: bool, model, task, row) -> str:
    """Функция возвращает имя продукта веги"""

    pd_file = model.pd_masks
    culture = task.culture
    veg_prod = 'None'

    if str(row['id_region']) != 'nan':
        reg_uid = str(int(row['id_region']))
    else:
        reg_uid = str(row['id_region'])

    if ndvi:    # Загрузка маски ndvi из файла в зависимости от культуры
        res_ndvi = pd_file[(pd_file['id_region'].astype(str).str.contains(str(reg_uid)))]

        print('res_ndvi', res_ndvi)

        mask = res_ndvi[culture].values[0]

        veg_prod = str(mask)
        print('Загружена маска', mask)

    if ndvi and historical:
        flag = 'n_n'
        splited_mask = veg_prod.split(flag)
        veg_prod = splited_mask[0] + 'n_interannual_n' + splited_mask[-1]

    if not ndvi and historical:
        # print('parm###', parm)
        veg_prod = parm.replace('_historical', '').replace('mean_', 'mean_interannual_')

    if not ndvi and not historical:
        veg_prod = parm

    print('Продукт веги -', veg_prod)
    return veg_prod


def write_time_row_json(json, row, task, parm, directory='Temporary_information\\TimeRows\\'.replace('\\', os.sep), hist=False):

    if str(row['id_district']) != 'nan':
        id_dist = str(int(row['id_district']))
    else:
        id_dist = str(row['id_district'])

    if str(row['id_region']) != 'nan':
        id_region = str(int(row['id_region']))
    else:
        id_region = str(row['id_region'])

    if hist == False:
        year = task.prediction_years[task.prediction_years_iteration]
    else:
        year = 'historical'

    file_name = str(parm).replace('_historical', '') + '_' + str(id_region) + '_' + str(id_dist) + '_' + str(year)
    write_inf(json, directory + file_name +  '.json')
    print('Записан файл', directory + file_name +  '.json' )

    return file_name, directory


def find_in_vega(parm, row, vega_api, model, task, make_file=True, hist=False):

    product_type = parm
    year = task.prediction_years[task.prediction_years_iteration]
    masks_pd = model.pd_masks

    if str(row['id_district']) != 'nan':
        district_uid = str(int(row['id_district']))
    else:
        district_uid = str(row['id_district'])

    if str(row['id_region']) != 'nan':
        reg_uid = str(int(row['id_region']))
    else:
        reg_uid = str(row['id_region'])

    json, success = None, False

    veg_prod = vega_product(parm, ('ndvi' in parm), ('historical' in parm), model, task, row)

    print('district_uid', district_uid, 'reg_uid', reg_uid)

    if str(district_uid) == 'nan':
        json, success = vega_api.region(veg_prod, year, reg_uid)
        file_name, directory = write_time_row_json(json, row, task, parm, hist=hist)

    else:
        json, success = vega_api.district(veg_prod, year, district_uid)
        file_name, directory = write_time_row_json(json, row, task, parm, hist=hist)

    return json, veg_prod, success, file_name, directory


def empty_json(json):
    """Проверка пустоты загруженного файла"""
    xy = json["data"]["1"]["xy"]
    if xy == []:
        return True
    else:
        return False


def get_not_hist_row(parm, row, vega_api, model, task,          # Функция возвращает массив временного ряда нужной длины
                     directory='Temporary_information\\TimeRows\\'.replace('\\', os.sep),
                     time_rows_size=365, hist=False):

    json, veg_prod, success, file_name, directory = find_in_vega(parm, row, vega_api, model, task, hist=hist)    # Поиск в веге
    # print(json)
    print('directory + file_name', directory + file_name)
    time_data, _ = TimeRow(directory + file_name).get_array_by_size(my_size=time_rows_size, # Открытие файла и приведение к нужному размеру
                                                                    padding='closest_value')

    return time_data, file_name, directory


def save_like_json(self, array, file_name, first_data = "2000-01-01 00:00:00",
                   last_data = "2000-12-31 23:59:59", directory = 'Dist_info_actual\\Average_long_term_data\\'.replace('\\', os.sep)):
        # file_name = directory + str(product) + "_" + str(dist_id) + "_" + str(year) + ".json"
        print('file_name_at_seving', file_name)
        file_name = directory + file_name
        target_array = []
        step = 365/(len(array)-1)
        for i in range(len(array)):
            time_i = round(array[i], 4)
            value_i = str(round(i * step, 2))
            target_array.append([value_i, time_i])
        json_file = {"data": {"1": {"xy": target_array,"labels": {target_array[0][0]: first_data, target_array[-1][0]: last_data}}}}
        write_inf(json_file, file_name)

        pass


def get_hist_row(parm, row, vega_api, model, task, directory='Temporary_information\\TimeRows\\'.replace('\\', os.sep),
                 time_rows_size=365):

    print('task.years_to_back', task.years_to_back)
    print('task.prediction_years', task.prediction_years)

    for i in range(task.years_to_back): # Загружаем несколько рядов за прошлые годы

        print(' int(task.prediction_years[0])',  int(task.prediction_years[task.prediction_years_iteration]))
        year_i = int(task.prediction_years[task.prediction_years_iteration]) - i
        task_fiction = copy.deepcopy(task)
        task_fiction.prediction_years[task.prediction_years_iteration] = year_i
        parm_i = parm.replace('_historical', '')
        print('year', task_fiction.prediction_years[task.prediction_years_iteration])

        time_data, file_name_last, directory = get_not_hist_row(parm_i, row, vega_api, model, task_fiction)

        print('New iteration', file_name_last)

    print('Iterations end')
    # id_region = str(row['id_region'])  # Запись полученного json в директорию
    # id_dist = str(row['id_district'])

    if str(row['id_district']) != 'nan':
        id_dist = str(int(row['id_district']))
    else:
        id_dist = str(row['id_district'])

    if str(row['id_region']) != 'nan':
        id_region = str(int(row['id_region']))
    else:
        id_region = str(row['id_region'])

    file_name_last_splited = file_name_last.split('_')
    print('file_name_last', file_name_last)
    file_name = file_name_last.replace(file_name_last_splited[-1], 'interannual')

    print('file_name mean ', file_name)

    print('Активируется класс поиска среднего')

    op = Operations()
    time_data, _ = op.mean_in_district(product=parm, dist_id=id_dist, file_name=file_name,
                                       size=time_rows_size, directory=directory, padding='closest_value')

    return time_data, file_name, directory



def find_information(parm, model, row, vega_api, task):     # Если нужны исторические данные, их расчитывет отдельная функция

    print('_parm_', parm)
    succes = False

    if not 'historical' in parm:  # Функция возвращает json по запросу, но многолетние данные следует рассчитать самостоятельно
        time_data, file_name, directory = get_not_hist_row(parm, row, vega_api, model, task)

        if time_data.shape[0] == 0:
            print('Данные за текущий год не найдены. Они будут заменены на многолетние...')
            time_data, file_name, directory = get_hist_row(parm, row, vega_api, model, task)
        else:
            succes = True

    elif 'historical' in parm:
        print('историч')
        print('year', task.prediction_years, 'parm', parm)
        time_data, file_name, directory = get_hist_row(parm, row, vega_api, model, task)

    if str(time_data) == 'None':
        print('Данные не найдены...')
    else:
        succes = True

    return time_data


def get_new_line_timerow(parm, task, row, model, vega_api, df_new_matrix):

    print('parm', parm, 'year for predict', task.prediction_years)
    file_success = chek_file(parm, row, task)  # Если файл уже есть на компе, он будет загружен
    print('file_success', file_success)

    if file_success:
        time_row_parm = read(parm, row, task)

    else:  # Если нет, информация будет найдена в интернете
        vega_api.connect()
        time_row_parm = find_information(parm, model, row, vega_api, task)

    print('time_row_parm.shape', time_row_parm.shape)
    print('parm', parm)
    # print('df_new_matrix', df_new_matrix)

    if time_row_parm.shape[0] == 0:  # Если матрица пустая,
        df_new_matrix, success = df_new_matrix, False
    else:
        if df_new_matrix.shape[0] == 0:
            df_new_matrix = np.array([time_row_parm])
        else:
            df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])

    return df_new_matrix


def get_raw_years_and_prods(row, task, all_years=False, including_current=False) -> tuple:
    '''Функция возвращает два списка. Список продуктивностей и соответствующие годы'''

    list_of_years = []
    list_of_prods = []
    heads = row.index

    print('heads', heads)
    counter = 0

    for head in heads:
        try:
            year = int(head)
            if int(year) <= int(task.prediction_years[task.prediction_years_iteration]) or including_current:
                if counter <= task.avegage_last or all_years:
                    year_i = int(year)
                    prod_i = float(str(row[year]).replace(',', '.'))
                    list_of_years.append(year_i)
                    list_of_prods.append(prod_i)
                    counter += 1
        except:
            pass

    print('list_of_years', list_of_years, 'list_of_prods', list_of_prods)

    return list_of_years, list_of_prods


def get_trend_parms(x_points: [list], y_points: [list]) -> tuple:
    '''Функция принимает массивы точек, строит линейную регрессию по методу наименьших квадратов. Возвращает параметры регрессии,
    такие как коэффициент наклона прямой и свдиг'''

    x_train = np.array(x_points).reshape(-1, 1)
    y_train = np.array(y_points).reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    coeff = regressor.coef_.reshape(-1, 1)[0][0]
    shift = regressor.intercept_.reshape(-1, 1)[0][0]

    return coeff, shift #, singular


def find_average_trend_of_prod_in_row(row, task) -> float:
    '''Функция получает строку и задание, собирает похдохящие значения из строки задания, строит регрессию и возвращает тренд.'''
    list_of_years, list_of_prods = gel_years_list_with_prods(task, row, all_points=False)

    list_of_years.reverse(), list_of_prods.reverse()

    print('finding trend...')

    try:
        trend, _ = get_trend_parms(list_of_years, list_of_prods)
    except:
        trend = 0

    print('year', task.prediction_years)
    print('list_of_years', list_of_years)
    print('list_of_prods', list_of_prods)
    print('trend', trend)

    return trend

def get_rows_by_parms(parm, task, row, vega_api, model, size_data_time):

    if not (('prod' in parm) or (
            'trend' in parm)):  # Если параметр не является характеристикой статистики...

        print('parm', parm, 'year', task.prediction_years)
        file_success = chek_file(parm, row, task)  # Если файл уже есть на компе, он будет загружен
        print('file_success', file_success)

        if file_success:
            time_row_parm = read(parm, row, task)

        else:  # Если нет, информация будет найдена в интернете
            vega_api.connect()
            time_row_parm = find_information(parm, model, row, vega_api, task)

    if 'prod' in parm:  # Добавляем в матрицу строку с продуктивностью
        mean_productive = find_average_prod_in_row(row, task)
        ones_line = np.ones(size_data_time)
        time_row_parm = ones_line * mean_productive  # Добавлена строка со средней продуктивностью
        pass

    elif 'trend' in parm:  # Добавляем в матрицу строку с трендом
        mean_trend = find_average_trend_of_prod_in_row(row, task)
        ones_line = np.ones(size_data_time)
        time_row_parm = ones_line * mean_trend  # Добавлена строка со средней продуктивностью
        pass

    return time_row_parm




def inf_loader(model, row, vega_api, task) -> Tuple[np.array, bool]:     # Итерирование по параметрам и формирование таблицы входных данных.

    success = True
    # model.activate()
    parms = model.parms
    df_new_matrix = np.array([])
    productive = False

    print('Параметры для модели', parms)
    print('')

    if ('LM' in model.model_name or 'EXM' in model.model_name) and not('KML' in model.model_name):       # Если модель линейная
        parm = parms[0]     # Загружается параметр для регрессии
        years, _ = gel_years_list_with_prods(task, row)
        print('years', years, 'prediction_year', task.prediction_years[task.prediction_years_iteration])

        for year in years:
            if year != task.prediction_years[task.prediction_years_iteration]:
                task_fiction = copy.deepcopy(task)  # Создаем фиктивное задание
                task_fiction.prediction_years[task.prediction_years_iteration] = year
                df_new_matrix = get_new_line_timerow(parm, task_fiction, row, model, vega_api, df_new_matrix)

        df_new_matrix = get_new_line_timerow(parm, task, row, model, vega_api, df_new_matrix)

    elif 'KML' in model.model_name:
        years, prods = gel_years_list_with_prods(task, row, all_points=True)
        print('years', years, 'prediction_year', task.prediction_years[task.prediction_years_iteration])

        try:
            size_data_time = matrix_norm_cut.shape[1]
        except:
            size_data_time = 365  # Использовать стандартный размер ряда

        years_in_df = []
        prod_list = []
        mean_prods_list = []

        matrix = np.array([])

        years.reverse()
        prods.reverse()

        print('years', years)
        print('prods', prods)


        for i in range(len(years)):     # итерации по годам и урожайностям
            year = years[i]
            prod = prods[i]

            df_line = np.array([])

            print('year', year)
            print('task.prediction_years_secret', task.prediction_years_secret)
            print('int(year) != int(task.prediction_years_secret)', int(year) != int(task.prediction_years_secret))

            if int(year) != int(task.prediction_years_secret):  # если год не соответсвует расчетному году, собираем статистику
                years_in_df.append(year)
                prod_list.append(prod)

                print('parms', parms)

                for parm in parms:  # итерации по параметрам
                    """Создаем фиктивное задание (копия экземпляра класса с заданием, 
                    чтобы передать ее в функцию, которая собирает исходные данные для прошлых лет)"""
                    task_fiction = copy.deepcopy(task)
                    task_fiction.prediction_years[task.prediction_years_iteration] = year
                    line_i = get_rows_by_parms(parm, task_fiction, row, vega_api, model, size_data_time)    # получаем вектор (график одного из параметров)

                    # print("'prod' in parm", 'prod' in parm)

                    if 'prod' in parm:      # Если параметр = продуктивность, запишем его в отдельную переменную, которую потом приложим к словарю с данными для расчетов.
                        mean_prods_list.append(line_i[0])

                    if df_line.shape[0] == 0:
                        df_line = np.array([line_i])
                    else:
                        df_line = np.vstack([df_line, [line_i]])

                if matrix.shape[0] == 0:
                    matrix = np.array([df_line])
                else:
                    matrix = np.vstack([matrix, [df_line]])

            else:
                years_in_df.append(year)
                print('parms_target_year', parms)

                for parm in parms:
                    task_fiction = copy.deepcopy(task)  # Создаем фиктивное задание
                    task_fiction.prediction_years[task.prediction_years_iteration] = year
                    line_i = get_rows_by_parms(parm, task_fiction, row, vega_api, model, size_data_time)

                    print('parm_target_year', parm)
                    print("'prod' in parm", 'prod' in parm)

                    if 'prod' in parm:
                        print('target_mean_prod = line_i[0] +')
                        target_mean_prod = line_i[0]

                    if df_line.shape[0] == 0:
                        df_line = np.array([line_i])
                    else:
                        df_line = np.vstack([df_line, [line_i]])
                rows = df_line


        # print('rows_1', rows.shape)
        # print('target_mean_prod_1', target_mean_prod)

        if not task.prediction_years_secret in years:

            df_line = np.array([])

            years_in_df.append(task.prediction_years_secret)
            print('parms_target_year', parms)

            for parm in parms:
                task_fiction = copy.deepcopy(task)  # Создаем фиктивное задание
                task_fiction.prediction_years[task.prediction_years_iteration] = task.prediction_years_secret
                line_i = get_rows_by_parms(parm, task_fiction, row, vega_api, model, size_data_time)

                print('parm_target_year', parm)
                print("'prod' in parm", 'prod' in parm)

                if 'prod' in parm:
                    print('target_mean_prod = line_i[0] +')
                    target_mean_prod = line_i[0]

                if df_line.shape[0] == 0:
                    df_line = np.array([line_i])
                else:
                    df_line = np.vstack([df_line, [line_i]])

            rows = df_line


        mean_prods = np.array(mean_prods_list)
        prods = np.array(prod_list)
        Deviation = (prods - mean_prods)/mean_prods

        print('years_in_df', years_in_df)

        print('')
        print('Проверка размера собранных данных', '\n')
        print('Deviation', Deviation)
        print('x_train.shape', matrix.shape)
        print('y_train len', len(prods))
        print('target_mean.shape', target_mean_prod.shape)
        print('y_train_mean_prods', mean_prods)
        print('x_target.shape', rows.shape)
        print('')

        print('years_in_df', years_in_df)
        df_new_matrix = {'x_train': matrix,
                         'y_train': prods,
                         'deviation': Deviation,
                         'target_mean': target_mean_prod,
                         'y_train_mean_prods': mean_prods,
                         'x_target': rows}

        print('df_new_matrix', df_new_matrix)
        print('df_new_matrix для KML успешно собран!')
        # print(heheh)

    else:
        for parm in parms:          # Если речь о нелинейной модели
            print('parn in progress', parm)

            try:
                size_data_time = matrix_norm_cut.shape[1]
            except:
                size_data_time = 365    # Использовать стандартный размер ряда

            if not(('prod' in parm) or ('trend' in parm)):  # Если параметр не является характеристикой статистики...

                print('parm', parm, 'year', task.prediction_years)
                file_success = chek_file(parm, row, task)  # Если файл уже есть на компе, он будет загружен
                print('file_success', file_success)

                if file_success:
                    time_row_parm = read(parm, row, task)

                else:   # Если нет, информация будет найдена в интернете
                    vega_api.connect()
                    time_row_parm = find_information(parm, model, row, vega_api, task)

                print('time_row_parm.shape', time_row_parm.shape)
                print('parm', parm)
                # print('df_new_matrix', df_new_matrix)
                print('df_new_matrix.shape', df_new_matrix.shape)


            if not(('LM' in model.model_name) or ('EXM' in model.model_name)):  # Если речь не идет о линейных моделях

                if 'prod' in parm:         # Добавляем в матрицу строку с продуктивностью
                    mean_productive = find_average_prod_in_row(row, task)
                    ones_line = np.ones(size_data_time)
                    time_row_parm = ones_line * mean_productive  # Добавлена строка со средней продуктивностью
                    pass

                elif 'trend' in parm:             # Добавляем в матрицу строку с трендом
                    mean_trend = find_average_trend_of_prod_in_row(row, task)
                    ones_line = np.ones(size_data_time)
                    time_row_parm = ones_line * mean_trend  # Добавлена строка со средней продуктивностью
                    pass

            if time_row_parm.shape[0] == 0:    # Если матрица пустая,
                df_new_matrix, success = None, False
            else:
                if df_new_matrix.shape[0] == 0:     # Происходит набор строк в одну матрицу
                    df_new_matrix = np.array([time_row_parm])
                else:
                    df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])

    # print('np_array_data.shape', df_new_matrix.shape)
    # print('')
    # print('np_array_data', df_new_matrix)
    # print('')

    return df_new_matrix, success


def normalize_matrix(matrix: [np.array], normalization_parms: [list]) -> [np.array]:       # Примитивная нормировка
    print('Длина массива и листа нормализации', matrix.shape, len(normalization_parms))
    df_new_matrix = np.array([])
    for i in range(matrix.shape[0]):
        line = matrix[i]
        norm_factor = normalization_parms[i]
        time_row_parm = line / norm_factor

        if df_new_matrix.shape[0] == 0:
            df_new_matrix = np.array([time_row_parm])
        else:
            df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])

    return df_new_matrix


def matrix_cut(matrix, first_day, last_day, model):
    '''Функция обрезает временные ряды, оставляя только заданный диапазон'''

    if not('KML' in model.model_name):
        print('not KML!')
        df_new_matrix = np.array([])
        for i in range(matrix.shape[0]):
            line = matrix[i]
            time_row_parm = line[first_day:last_day]
            if df_new_matrix.shape[0] == 0:
                df_new_matrix = np.array([time_row_parm])
            else:
                df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])
        matrix_norm_cut = df_new_matrix

    else:
        matrix_init = copy.deepcopy(matrix)
        key_list = list(matrix_init.keys())
        print('key_list', key_list)

        x_target = matrix_init['x_target']
        x_train = matrix_init['x_train']

        x_target = np.array([x_target])
        matrix_list = [x_train, x_target]

        for j in range(len(matrix_list)):
            matrix = matrix_list[j]
            df_new_big_matrix = np.array([])
            for i in range(matrix.shape[0]):    # перебор по примерам
                matrix_i = matrix[i]
                df_new_matrix = np.array([])
                for k in range(matrix_i.shape[0]):
                    line = matrix_i[k]
                    time_row_parm = line[first_day:last_day]
                    if df_new_matrix.shape[0] == 0:
                        df_new_matrix = np.array([time_row_parm])
                    else:
                        df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])

                if df_new_big_matrix.shape[0] == 0:
                    df_new_big_matrix = np.array([df_new_matrix])
                else:
                    df_new_big_matrix = np.vstack([df_new_big_matrix, [df_new_matrix]])

            matrix_init[key_list[-j]] = df_new_big_matrix

            print('matrix_init x_target', matrix_init['x_target'].shape)
            print('matrix_init x_train', matrix_init['x_train'].shape)

        matrix_norm_cut = matrix_init

    return matrix_norm_cut


def get_raw_years_list(row, task):
    years = []
    heads = row.index
    # print('heads', heads)

    for head in heads:
        try:
            year = int(head)
            if int(year) <= int(task.prediction_years[task.prediction_years_iteration]):
                years.append(int(year))
        except:
            pass

        # print('years_hehe', years)
    # print('years_list', years)

    return years


def get_raw_prod_list(row, years_list, task):
    prods = []

    years_in_attention = []
    years_list.sort()
    year_now = task.prediction_years_secret

    i = 0
    for j in range(len(years_list)):
        year_i = years_list[-j-1]
        prod_i = row[int(year_i)]

        if int(year_now) > int(year_i):
            if i < task.avegage_last:
                if str(prod_i) != 'nan':
                    if int(year_now) != int(year_i):
                        years_in_attention.append(year_i)
                        prod_i = float(str(prod_i).replace(',', '.'))
                        prods.append(prod_i)
                        i += 1


    if prods == []:
        i = 0
        for j in range(len(years_list)):
            year = years_list[-j - 1]
            prod_i = row[int(year)]
            if str(prod_i) != 'nan':
                if i <= task.avegage_last:
                    years_in_attention.append(year)
                    prod_i = float(str(prod_i).replace(',', '.'))
                    prods.append(prod_i)
                    i += 1

    print('')
    print('years_in_attention', years_in_attention)
    print('years_list', years_list)
    print('prod_i', prod_i)
    print('prods', prods)
    print('')

    return prods


def find_average_prod_in_row(row, task):
    """Функция принимает на вход строку из задания и возвращает среднюю продуктивность.
    Либо продуктивность дана, либо она будет подсчитана по приведенной статистикой"""

    avg_colm = row['Average productivity']
    print('avg_colm', avg_colm)

    if True: #str(avg_colm) == 'nan':
        years_list = get_raw_years_list(row, task)
        list_actual_prods = get_raw_prod_list(row, years_list, task)
        print('list_actual_prods', list_actual_prods)
        average_prod = np.array(list_actual_prods).mean()

    return average_prod


def NN_CN_v1_winter_wheat_prepeaer(matrix_norm_cut, model, task, mean_productive):
    """Функция принимает на вход массив переменных, продуктивность, нормирует ее
    и возвращает входные данные в формате, пригодном для загрузки в модель"""       # От данной функции лучше отказаться

    matrix_norm_cut.shape

    ones_line = np.ones((matrix_norm_cut.shape[1]))
    prod_line = ones_line * mean_productive             # Добавлена строка со средней продуктивностью
    # print('prod_line', prod_line)

    prod_idx = model.parms.index("mean_productive")
    print('prod_idx', prod_idx)

    prod_line_norm = prod_line / model.normalization_parms[prod_idx]
    # print('prod_line_norm', prod_line_norm)

    matrix_norm_cut = np.vstack([matrix_norm_cut, [prod_line_norm]])

    print('matrix_norm_cut', matrix_norm_cut.shape)

    return matrix_norm_cut


def roughen_up_to_a_month(line, points):
  counter = 0
  target_array = []
  month_array = []

  for i in range(line.shape[0]):
    # print(counter, (points), i, line.shape[0])
    # print('bool', counter < (points), i != line.shape[0])

    if counter <= (points):
      value_i = line[i]
      month_array.append(value_i)
      counter += 1

    if counter > points or i == line.shape[0] - 1:
      value_i = line[i]               # ???
      month_array.append(value_i)

      month_value = np.array(month_array).mean()
      target_array.append(month_value)

      month_value = np.array(month_array).max()
      target_array.append(month_value)

      month_value = np.array(month_array).min()
      target_array.append(month_value)

      month_value = sum(month_array)
      target_array.append(month_value)

      month_array = []
      counter = 0

  return target_array


def RF_v1_winter_wheat_prepeaer(matrix_mod_new_size, model, task, mean_productive):
    """Функция принимает на вход массив переменных, от функции, подготавливающей данные для нейросети
     и возвращает входные данные в формате, пригодном для загрузки в модель случайного леса"""

    mean_month = 30

    print('matrix_mod_new_size', matrix_mod_new_size.shape)
    points = math.ceil(matrix_mod_new_size.shape[1]/mean_month)
    print('points', points)

    list_of_parms = model.parms

    new_line = []
    for j in range(matrix_mod_new_size.shape[0]):  # Перебор по параметрам
        parm_j = list_of_parms[j]

        if parm_j != 'mean_productive':
            line = matrix_mod_new_size[j]

            # print('line', line.shape)
            points_array = roughen_up_to_a_month(line, mean_month)

            # print('points_array.shape', len(points_array))

            for point_i in points_array:
                new_line.append(point_i)

            additional_values = [np.array(line).max(), np.array(line).min(), sum(line)]

        else:
            new_line.append(matrix_mod_new_size[j][0])

    new_line = np.array(new_line)
    print('new_line', new_line.shape)

    return new_line


def SLM_v1_prepeaer(matrix_norm_cut, model, task, mean_productive, row):

    maximums = []
    for i in range(matrix_norm_cut.shape[0]-1):
        print('matrix_norm_cuti', matrix_norm_cut[i])
        maximums.append(matrix_norm_cut[i].max())

    print('matrix_norm_cut.shape', matrix_norm_cut.shape)
    # idx = matrix_norm_cut.shape[0]

    print('matrix_norm_cut', matrix_norm_cut)
    target_arg = matrix_norm_cut[-1].max()

    years, prods = gel_years_list_with_prods(task, row)
    print('years', years, 'prods', prods, 'target_arg', target_arg)

    print('[maximums, prods, target_arg]', [maximums, prods, target_arg])

    return [maximums, prods, target_arg, years]


def Averager_of_lists(vector, model, first_day, last_day) -> list:
    days_in_point = model.parm_file_info["averaging"]  # Параметр загрубления
    additional_features = model.parm_file_info["features_pross"]  # Спиcок дополнительных фич
    target_vector = []

    temporary_list = []
    averaging_count = 0
    count = 0

    # print('vector', vector)

    for i in range(vector.shape[0]):
        # if i >= first_day and i <= last_day:
        value_i = vector[i]
        if averaging_count < days_in_point:
            temporary_list.append(value_i)
            averaging_count += 1
            # print('ioy 1')

        if averaging_count == days_in_point:
            temporary_list.append(value_i)
            # print('ioy 2')

            for additional_feature in additional_features:

                # print('ioy 3')
                if additional_feature == 'mean':
                    target_vector.append(np.array(temporary_list).mean())
                    count += 1
                if additional_feature == 'max':
                    target_vector.append(np.array(temporary_list).max())
                    count += 1
            temporary_list = []
            averaging_count = 0

    if temporary_list != []:
        temporary_list.append(value_i)
        # print('ioy 2')
        for additional_feature in additional_features:
            # print('ioy 3')
            if additional_feature == 'mean':
                target_vector.append(np.array(temporary_list).mean())
                count += 1
            if additional_feature == 'max':
                target_vector.append(np.array(temporary_list).max())
                count += 1

    # print('target_vector', target_vector)
    # print('counter', count)

    return target_vector


def Averager(x_train_data, first_day, last_day, model, task) -> np.array:
    '''Список матриц необходимо преобразовать в список векторов.
    Строки матриц, которые относятся к временным рядам нужно усреднить в соответсвии с настройкой,
    а строки, относящиеся к скалярным значениям, необходимо записать в вектор единожды. '''

    parm_list = model.parms     # Загружаем список параметров
    scalars_parms = model.parm_file_info["scalars"]     # Загружаем список скалярных параметров

    print('scalars_parms', scalars_parms)

    target_matrix = np.array([])

    print('x_train_data.shape', x_train_data.shape)

    for i in range(x_train_data.shape[0]):    # Перебор по списку примеров для обучения регрессии
        exsample_matrix = x_train_data[i]
        exsample_vector = []

        print('exsample_matrix', exsample_matrix)
        print('exsample_matrix.shape', exsample_matrix.shape)
        print('parm_list', parm_list)

        for j in range(exsample_matrix.shape[0]):    # Перебор по параметрам
            vector = exsample_matrix[j]
            parm = parm_list[j]

            if parm in scalars_parms:
                new_vector_fragment = [vector[0]]
            else:
                new_vector_fragment = Averager_of_lists(vector, model, first_day, last_day)
            exsample_vector = exsample_vector + new_vector_fragment

        np_exsample_vector = np.array(exsample_vector)

        if target_matrix.shape[0] == 0:
            target_matrix = np.array([np_exsample_vector])
        else:
            target_matrix = np.vstack([target_matrix, [np_exsample_vector]])

    return target_matrix


def save_json_for_interest(dict_data_norm_cut):

    print("x_train", type(dict_data_norm_cut['x_train']))
    print("y_train", type(dict_data_norm_cut['y_train']))
    print("deviation", type(dict_data_norm_cut['deviation']))
    print("target_mean", type(dict_data_norm_cut['target_mean']))
    print("y_train_mean_prods", type(dict_data_norm_cut['y_train_mean_prods']))
    print("x_target", type(dict_data_norm_cut['x_target']))

    dict_data_norm_cut['x_train'] = dict_data_norm_cut['x_train'].tolist()
    dict_data_norm_cut['y_train'] = dict_data_norm_cut['y_train'].tolist()
    dict_data_norm_cut['deviation'] = dict_data_norm_cut['deviation'].tolist()
    dict_data_norm_cut['target_mean'] = float(dict_data_norm_cut['target_mean'])
    dict_data_norm_cut['y_train_mean_prods'] = dict_data_norm_cut['y_train_mean_prods'].tolist()
    dict_data_norm_cut['x_target'] = dict_data_norm_cut['x_target'].tolist()






    write_inf(dict_data_norm_cut, 'Data')


def Complex_linear_model_v1_prepeaer(dict_data_norm_cut, first_day, last_day, task, model):
    """Пока пустая функция для обработки данныех всеядной модели"""

    # save_json_for_interest(copy.deepcopy(dict_data_norm_cut))

    x_train_data = dict_data_norm_cut['x_train']
    y_train_data = dict_data_norm_cut['y_train']
    x_train_shape = x_train_data.shape
    x_target_data = dict_data_norm_cut['x_target']

    print('')
    print('Подготовка данных', '\n')
    print('x_train_data.shape', x_train_data.shape)
    print('y_train_data.shape', y_train_data.shape)
    print('x_target_data', x_target_data.shape)
    print('')

    x_train_data = Averager(x_train_data, first_day, last_day, model, task)

    print('')
    print('x_train_data.shape', dict_data_norm_cut['x_train'].shape)
    print('x_target_data.shape', x_target_data.shape)
    print('')

    x_target_data = Averager(x_target_data, first_day, last_day, model, task)

    dict_data_norm_cut['x_train'] = x_train_data
    dict_data_norm_cut['x_target'] = x_target_data

    print('')
    print('Подготовка данных_2', '\n')
    print('x_train_data.shape', x_train_data.shape)
    print('y_train_data.shape', y_train_data.shape)
    print('x_target_data', x_target_data.shape)
    print('')

    save_json_for_interest(copy.deepcopy(dict_data_norm_cut))

    return dict_data_norm_cut


def inf_processor(matrix, model, task, row):  # Предварительная подготовка временного ряда

    print('')
    current_day = task.current_day  # Текущий день
    parms_list = model.parms

    first_day = model.first_day
    last_day = model.last_day
    normalization_parms = model.normalization_parms

    print('model.model_name', model.model_name)

    if ('LM' in model.model_name) or ('EXM' in model.model_name):
        matrix_norm = matrix
        print('normalize_matrix_no')

    else:
        print('normalize_matrix_go')
        print('')
        print('matrix before norm shape', matrix.shape)
        print('matrix before norm', matrix)
        matrix_norm = normalize_matrix(matrix, normalization_parms)
        print('')
        print('matrix_norm shape', matrix_norm.shape)
        print('matrix_norm', matrix_norm)
        print('')

    print('matrix_norm', matrix_norm, '\n')

    matrix_norm_cut = matrix_cut(matrix_norm, first_day, last_day, model)
    print('matrix_norm_cut', matrix_norm_cut, '\n')

    mean_productive = find_average_prod_in_row(row, task)

    print('model.model_name', model.model_name)
    print('mean_productive', mean_productive)

    if 'NN_CN' in model.model_name: # model.model_name == 'NN_CN_v1_winter_wheat':
        # np_array_data_processed = NN_CN_v1_winter_wheat_prepeaer(matrix_norm_cut, model, task, mean_productive)
        np_array_data_processed = matrix_norm_cut
        pass

    if 'RF' in model.model_name: # model.model_name == 'RF_v1_winter_wheat':
        # np_array_data_processed = NN_CN_v1_winter_wheat_prepeaer(matrix_norm_cut, model, task, mean_productive)
        np_array_data_processed = matrix_norm_cut
        print('np_array_data_processed.shape', np_array_data_processed.shape)
        np_array_data_processed = RF_v1_winter_wheat_prepeaer(np_array_data_processed, model, task, mean_productive)
        pass

    if ('LM' in model.model_name or 'EXM' in model.model_name) and not('KML' in model.model_name):
        np_array_data_processed = SLM_v1_prepeaer(matrix_norm_cut, model, task, mean_productive, row)
        pass

    if 'KML' in model.model_name:
        np_array_data_processed = Complex_linear_model_v1_prepeaer(matrix_norm_cut, first_day, last_day, task, model)


    success = True
    return np_array_data_processed, success, mean_productive


def predictor(np_array_data_processed, model_class, task, row):        # Запуск модели, сохранение прогноза

    model_name = model_class.model_name
    model = model_class.model

    if 'NN_CN' in model_name:   #model_name == 'NN_CN_v1_winter_wheat':
        predicts = []
        for i in range(100):     # Цикл, в котором нейросеть делает несколько предсказаний.
            # Каждый раз они немного отличаются из-за модуля Дропаута
            # np.savetxt("Enter_data.csv", np_array_data_processed)
            enter = np_array_data_processed
            enter = enter.reshape(1, 18, -1)
            enter = torch.from_numpy(enter)
            enter = enter.to(torch.float32)
            prediction = model(enter)
            predicts.append(float(prediction[0]))

        anwers = np.array(predicts)

        dispersion = np.var(anwers)
        verdict = anwers.mean()
        verdict = Answer(verdict = verdict, inf_unconditional={'dispersion': dispersion})

    if 'RF' in model_name: # model_name == 'RF_v1_winter_wheat':
        print('np_array_data_processed', np_array_data_processed.shape)
        verdict = model.predict(np_array_data_processed.reshape(1, -1))
        verdict = Answer(verdict=verdict)

    if ('LM' in model_name or 'EXM' in model_name) and not('KML' in model_name):
        # print('np_array_data_processed_LM', np_array_data_processed)
        verdict = model.predict(np_array_data_processed, task, row)

    if 'KML' in model_name:
        # print('np_array_data_processed_LM', np_array_data_processed)
        verdict = model.predict(np_array_data_processed, task, row)

    print('verdict', verdict)
    success = True

    return verdict, success


def get_one_verdict(model, row, vega_api, task) -> Tuple[Answer, bool]:
    '''Функция получает модель, строку из документа с заданием, загрузчик данных и задание, а возвращает готовый ответ'''

    success = True
    if success:  # Проверка успеха предыдущего действия (Данное место нужно переписать...)
        np_array_data, success = inf_loader(model, row, vega_api, task)  # Скачиваем информацию
    else:
        verdict = 'unexpected error'  # Если случится ошибка, система вернет комментарий ошибки вместо вердикта
    if success:
        np_array_data_processed, success, average = inf_processor(np_array_data, model, task, row)  # Производим предварительную подготовку
    else:
        verdict = 'inf loader error'
    if success:
        # print('np_array_data_processed', np_array_data_processed, model.model_name)
        verdict, success = predictor(np_array_data_processed, model, task, row)  # Запуск модели, сохранение прогноза
    else:
        verdict = 'inf processor error'

    if success == False:
        verdict = 'No data'

    return verdict, success


def gel_years_list_with_prods(task, row, all_points=False):
    '''Функция возвращает n последних урожайностей и соответсвующих им лет, либо все возможные урожайнсти и годы, если all_points=True'''
    years_list = get_raw_years_list(row, task)
    years_list.sort(reverse=True)

    years_available = []
    prods_available = []

    # print('row', row)

    print('task.avegage_last', task.avegage_last)
    print('all_points', all_points)

    if all_points:
        for i in range(len(years_list)):
            year_i = int(years_list[i])
            prod_i = float(str(row[int(year_i)]).replace(',', '.'))
            if str(prod_i) != 'nan':
                if year_i < int(task.prediction_years[task.prediction_years_iteration]):
                    years_available.append(year_i)
                    prods_available.append(prod_i)
    else:
        counter = 0
        for i in range(len(years_list)):
            year_i = int(years_list[i])
            prod_i = float(str(row[int(year_i)]).replace(',', '.'))
            if counter <= task.avegage_last:
                if str(prod_i) != 'nan':
                    years_available.append(year_i)
                    prods_available.append(prod_i)
                    counter += 1

    if not all_points:
        years_available_short = []
        prods_available_short = []

        counter = 0
        for i in range(len(years_available)):

            if int(years_available[i]) < int(task.prediction_years[task.prediction_years_iteration]):
                if counter <= task.avegage_last:
                    years_available_short.append(years_available[i])
                    prods_available_short.append(prods_available[i])
                    counter += 1

        years_available = years_available_short
        prods_available = prods_available_short

    return years_available, prods_available


def get_correct_factor(prods_available, predicts_for_past):
    prods_available = np.array(prods_available)
    predicts_for_past = np.array(predicts_for_past)

    difference = prods_available - predicts_for_past

    correct_factor = difference.mean()
    return correct_factor


def exponental_filter(diff: np.array) -> np.array:
    print('diff', diff)
    return diff


def verdict_normalization(prods_available, predicts_for_past, verdict_bc):      # Происходит нормировка вердикта.
    predicts = np.append(np.array(predicts_for_past), np.array(verdict_bc))
    real = np.array(prods_available)

    print('predicts', predicts)
    print('predicts_for_past', predicts_for_past)
    print('real', real)

    diff = (real - np.array(predicts_for_past))

    diff_foltred = exponental_filter(diff)          # Функция не дописана.
    diff_foltred = diff_foltred.mean()

    print('diff_foltred', diff_foltred)
    preds_norm_shift = predicts + diff_foltred

    print('preds_norm_shift', preds_norm_shift)

    predicts_for_past_corrected = predicts + diff_foltred

    predicts_for_past_corrected = predicts_for_past_corrected[0:predicts_for_past_corrected.shape[0]-1]

    return preds_norm_shift[-1], predicts_for_past_corrected


def predict_and_correction_real(model, row, vega_api, task):
    """Функция делает один прогноз. Функция может просто сделать прогноз, либо сделать прогноз для нескольких прошлых лет, для
    которых известны продуктивности, подсчитать систематическую ошибку и вычесть ее для прогноза на заданый год. Кроме того, длагодаря прогнозам на
    прошлые годы, программа оценивает ско и прочие метрики качества для данного района/региона"""

    years_available, prods_available = gel_years_list_with_prods(task, row)     # Получим список продуктивностей в соответствующие годы

    print('years_available_1', years_available)
    print('prods_available_1', prods_available)

    predicts_for_past = []

    print('task.years_to_back', task.years_to_back)
    print('years_available', years_available)

    if task.correction and (not ('M' in model.model_name)):     # Для линейных моделей коррекция не предполагается

        years_for_correct_avalible = min(task.years_for_correct, len(years_available))

        # for i in range(task.years_for_correct):       # Делаем предсказания на N прошлых доступных лет
        for i in range(years_for_correct_avalible):  # Делаем предсказания на N прошлых доступных лет

            print('counter', i)
            year_i = years_available[i]                 # Получаем год
            task_fiction = copy.deepcopy(task)          # Создаем фиктивное задание
            task_fiction.prediction_years[task.prediction_years_iteration] = year_i
            task_fiction.prediction_years_secret = task.prediction_years[task.prediction_years_iteration]
            verdict_i, succses = get_one_verdict(model, row, vega_api, task_fiction)
            print('verdict_i_type', type(verdict_i))

            print('verdict = ', verdict_i)
            print('verdict_i.verdict = ', verdict_i.verdict)

            try:
                verdict_i.verdict.astype(float)
                print('astype(float) is done!')
            except:
                print('verdict_i_type', type(verdict_i.verdict))

            predicts_for_past.append(float(verdict_i.verdict))
            print('verdict_i ', verdict_i)

        task.prediction_years_secret = task.prediction_years[task.prediction_years_iteration]
        verdict_bc, succses = get_one_verdict(model, row, vega_api, task)

        average = find_average_prod_in_row(row, task)
        prods_available = prods_available[0: len(predicts_for_past)]

        print('prods_available', prods_available)
        print('predicts_for_past', predicts_for_past)
        print('years_available', years_available[0: len(predicts_for_past)])

        verdict_corrected = copy.deepcopy(verdict_bc)

        verdict_corrected.verdict, predicts_for_past_corrected = verdict_normalization(prods_available, predicts_for_past, copy.deepcopy(verdict_bc.verdict))

        verdict_corrected = print_metrics(prods_available, predicts_for_past_corrected, None, verdict_corrected)

        print('verdict_bc', verdict_bc.verdict)
        print('average', average)
        print('verdict_corrected', verdict_corrected.verdict)

    else:
        task.prediction_years_secret = task.prediction_years[task.prediction_years_iteration]
        verdict_bc, succses = get_one_verdict(model, row, vega_api, task)
        verdict_corrected, average = verdict_bc, find_average_prod_in_row(row, task)


    return verdict_corrected, average


def predict_array(row, task) -> Tuple[list, float]:   # Получаем строку и задание. Каждая модель из списка должна получить данные и отработать.
    """Функция принимает на вход задание, pd строку метаданных. Из задания принимает список оберток моделей.
    Те в свою очередь хранят список входных данных, которые нужно загрузить из интернета, и инструменты для
    предварительной обработки. Данные скачиваются, обрабатываются, пропускаются через модель и возвращается
    список прогнозов"""

    model_list = task.models_list
    column_models = task.column_models

    verdicts = []
    success = True

    vega_api = VegaAPI(task.login, task.password)   # Инициализируем экземпляр класса, с где будет храниться сеанс подключения к сайту для загрузки рассчетных данных

    for i in range(len(column_models)):         # Итерации по моделям
        years_list_in_row, _ = gel_years_list_with_prods(task, row, all_points=True)    # Получим список годов, для которых у нас имеются продуктивности в обрабатываемой строчке документа
        print('years_list_in_row', years_list_in_row)
        print('len(years_list_in_row)', len(years_list_in_row))
        print('column_models', column_models)
        task.prediction_years_iteration = i

        if len(years_list_in_row) != 0:
            print('Длины ', len(model_list), len(column_models))
            print(model_list, column_models)
            model, model_name = model_list[i], column_models[i]
            model.activate()    # Активируем модель (произойдут предварительные приготовления)

            year = task.prediction_years[task.prediction_years_iteration]     # Загрузим год, для которого делается предсказание
            model_name_in_cap = model_name + '_predict_' + year
            print('model_name_in_cap', model_name_in_cap, 'years', year)

            if str(row['id_region']) == 'nan' and str(row['id_district']) == 'nan':     # Если строка полностью пустая, пропускаем ее
                verdicts, average = None, None    # Заполним целевые переменные None. Они будут возвращены в таком виде
                print('Пустрая строка в задании!')

            else:
                if str(row[model_name_in_cap]) == 'nan':     # Если в данной строке столбец с ответом все еще не заполнен, запускается расчет
                    verdict, average = predict_and_correction_real(model, row, vega_api, task)  # Запускается расчет
                else:
                    verdict = Answer(verdict='nan')
                    print('Ведрикт уже записан!', verdict)
                    average = row['Average productivity']


            verdicts.append(verdict)

        else:
            verdict = Answer(verdict='')
            average = ''

    return verdicts, average


def record_additional_information(df_destination, index, answer):

    unconditional_dict = answer.inf_unconditional
    dict_keys_list = unconditional_dict.keys()

    for key in dict_keys_list:
        print('unconditional_dict[key]', key, unconditional_dict[key])
        df_destination.loc[index, str(key)] = unconditional_dict[key]

    additional_dict = answer.additional_inf
    dict_keys_list = list(additional_dict.keys())

    for i in range(len(dict_keys_list)):
        key = dict_keys_list[-i-1]
        df_destination.loc[index, str(key)] = additional_dict[key]


    return df_destination


def download_and_prediction(task):  # Идет перебор по стокам задания, получаем ответы всех моделей и записываем

    way = task.directory + task.task_file_reserw
    way_2 = task.directory + task.task_file
    print('Путь к резервной копии', way, '\n')
    df_destination = pd.read_excel(way, index_col='Unnamed: 0')

    interest_head = ['id_country', 'region', 'district', 'culture', 'id_region', 'id_district', 'Average productivity']
    column_predict = task.column_predict

    with alive_bar(df_destination.shape[0], force_tty=True) as bar:
        for index, row in df_destination.iterrows():  # Итерации по списку задания

            row_predict, average = predict_array(row, task)     # Передаем функции строку и экземпляр класса задания
            print('average', average)
            df_destination.loc[index, 'Average productivity'] = average

            for i in range(len(row_predict)):
                if row_predict[i].verdict != 'nan':
                    df_destination.loc[index, column_predict[i]] = row_predict[i].verdict
                    df_destination = record_additional_information(df_destination, index, row_predict[i])

            df_destination.to_excel(way_2)
            bar()
    return True


def main():
    print('Добро пожаловать в Nerual Rabbit!', '\n')

    task = fast_user_survey()       # Функция возвращает экземпляр класса с заданием. Позже следует заменить эту часть функцией, реализующей интерфейс.
    print('Получено задание: ', task.task_file, '\n')

    task = destination_analysis(task)
    task = model_Loader(task)  # Загрузка выбранной модели
    succes = download_and_prediction(task)  # Запуск цикла загрузок данных и предсказания.


    pass

if __name__ == '__main__':
    # cProfile.run('main()')
    main()


