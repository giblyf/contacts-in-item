from typing import Tuple, Union
import logging
import lightgbm as lgb
import sys
from data_transformation import DataTransformation

# Определение класса для обучения модели


class ModelTrainer:
    def __init__(self) -> None:
        self.logger = self._get_logger()

    def _get_logger(self):
        # Создание и настройка логгера
        logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger
    # Метод класса для обучения модели

    def initiate_model_trainer(self, X_train, y_train, X_val, y_val):
        self.logger.info(f'Начало обучения модели LightGBM...')

        # Параметры модели LightGBM
        params = {
            'boosting_type': 'gbdt',
            'colsample_bytree': 1.0,
            'importance_type': 'split',
            'learning_rate': 0.1,
            'max_depth': 7,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'n_estimators': 200,
            'n_jobs': None,
            'num_leaves': 100,
            'objective': 'binary',
            'random_state': None,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'subsample': 1.0,
            'subsample_for_bin': 200000,
            'subsample_freq': 0,
            'metric': 'binary_logloss',
            'feature_fraction': 1.0
        }

        # Создание датасетов LightGBM для обучения и валидации
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        # Обучение модели LightGBM
        classifier = lgb.train(
            params, lgb_train, valid_sets=[
                lgb_train, lgb_val])

        self.logger.info(f'Конец обучения модели LightGBM...')

        return classifier

# Функция для выполнения задачи 1


def task1(train_df, val_df, test_df) -> float:
    # Инициализация объекта для предварительной обработки данных
    preprocessor = DataTransformation()

    # Преобразование данных для обучения, валидации и теста
    X_train = preprocessor.initiate_data_transformation(train_df).values
    X_val = preprocessor.initiate_data_transformation(val_df).values
    X_test = preprocessor.initiate_data_transformation(test_df).values

    # Извлечение целевых переменных
    y_train = train_df['is_bad'].values
    y_val = val_df['is_bad'].values

    # Инициализация и обучение модели
    trainer = ModelTrainer()
    model = trainer.initiate_model_trainer(X_train, y_train, X_val, y_val)

    # Возвращение прогнозов модели для тестовых данных
    return model.predict(X_test)

# Функция для выполнения задачи 2


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)

    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
