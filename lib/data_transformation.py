import re
import logging
import sys
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from razdel import tokenize
import string
import fasttext

# Определение класса по трансформации данных


class DataTransformation:
    def __init__(self):
        # Инициализация логгера для вывода информации
        self.logger = self._get_logger()

        # Загрузка предобученной модели FastText для русского языка
        self.logger.info(
            f'Загрузка предобученной модели FastText для русского языка')
        self.preprocessor = fasttext.load_model('data/cc.ru.300.bin')

        # Компиляция регулярного выражения для поиска телефонных номеров
        self.phone_pattern = re.compile(
            r'(\+7|7|8)?[\s\-]?\(?[489][0-9]{2}\)?[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}')
        self.translation_table = str.maketrans(
            {"ъ": "ь", "ё": "е", '\n': ' ', '\t': ' '})

        # Загрузка стоп-слов и пунктуации для фильтрации
        self.logger.info(f'Загрузка стоп-слов и пунктуации для фильтрации')
        self.stop_words = set(stopwords.words('russian'))
        self.punctuation = set(string.punctuation)

    def _get_logger(self):
        # Создание и настройка логгера
        logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def clean_text(self, text):
        # Преобразование текста: приведение к нижнему регистру,
        # замена телефонных номеров, удаление лишних символов и множественных
        # пробелов
        text = text.lower()
        text = self.phone_pattern.sub(' контакты ', text)
        text = text.translate(self.translation_table)
        text = re.sub('[^а-я\\s]', '', text)
        text = re.sub(' +', ' ', text)

        return text

    def tokenize_text(self, text):
        # Токенизация текста и фильтрация стоп-слов и пунктуации
        tokens = list(tokenize(text))
        tokens = [
            word.text for word in tokens if word.text not in self.stop_words and word.text not in self.punctuation and len(
                word.text) >= 3]

        return tokens

    def match_phone(self, text):
        # Поиск телефонных номеров в тексте
        return bool(self.phone_pattern.search(text)) * 1

    def get_embedding(self, word):
        # Получение векторного представления слова из FastText
        try:
            embedding = self.preprocessor[word]
        except BaseException:
            embedding = np.zeros((300,))
        return embedding

    # Метод класса для трансформации данных

    def initiate_data_transformation(self, df):
        self.logger.info(f'Начало предобработки данных...')

        # Создание целевой переменной 'info' путем конкатенации столбцов
        # "Категория", "Заголовок", "Описание"
        df['info'] = df['category'].astype(
            str) + ' ' + df['title'].astype(str) + ' ' + df['description'].astype(str)

        # Очистка текста
        self.logger.info(f'Очистка текста...')
        df['info_cleaned'] = df['info'].apply(self.clean_text)

        # Токенезация текста
        self.logger.info(f'Токенизация текста...')
        df['info_tokenized'] = df['info_cleaned'].apply(self.tokenize_text)

        # Генерация дополнительной переменной "есть ли номер телефона в
        # объявлении"
        df['phone_catch'] = df['info'].apply(self.match_phone)

        # Получение векторных представлений для каждого объявления
        self.logger.info(f'Векторизация токенов...')
        X_embeddings = [np.mean(np.array(list(map(self.get_embedding, tok_sent))), axis=0)
                        for tok_sent in df['info_tokenized'].values]
        X_embeddings = np.array(X_embeddings)

        # Объединение векторных представлений и переменной "есть ли номер
        # телефона" в единый DataFrame
        X = pd.concat([pd.DataFrame(X_embeddings), df['phone_catch']], axis=1)

        self.logger.info(f'Конец предобработки данных...')

        return X
