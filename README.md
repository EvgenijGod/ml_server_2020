# ml_server_2020

## Основные команды

Команда, осуществляющая сборку докер-образа:

$ sudo docker build -t ml_server_2020 .

Команда, запускающая сервер:

$ sudo docker run --rm -p 1234:1234 -v "$PWD/server/data:/root/server/data" ml_server_2020

## Краткое описание

На титульной странице находятся 3 ссылки:
- Первая позволяет создать модель градиентного бустинга
- Вторая позволяет создать модель случайного леса
- Третья содержит интерфейс взаимодействия с обученными моделями и просмотром информации об их обучении и гиперпараметрах

Создание модели состоит из этапов:
- Задание гиперпараметров
- Выбор обучающего множества
- Создание и обучение модели

На странице с использованием моделей нужно выбрать из списка обученную модель и произвести на ней обучение, 
предварительно указав подходящий по формату файл. Результат будет сохранен в папке по адресу: 
**$PWD/server/data/predict** в формате имя поданного файла + _predict в формате .csv

Сервер поддерживает работу только с файлами формата .csv

Также предусмотрена возможность просмотреть информацию о модели. Для этого нужно выбрать ее в cписке на этой же странице.




