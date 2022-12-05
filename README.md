# Описание модулей
## Tracker soft
[Модуль](https://github.com/GlazkoAE/object-tracking-assignment/tree/develop/soft_tracker) реализован на алгоритме трекинга Sort. В процессе работы ID треков присваиваются, если объект был продетектирован на двух последовательных кадрах. 

Алгоритм основан на венгерском алгоритме и фильтре Калмана. Фильтр Калмана предсказывает bbox на текущем кадре, основываясь на предыдуших. Этот результат сравнивается с каждым продетектированным объектом по метрике IoU и заносится в матрицу. При помощи венгерского алгоритма ID объектов и треков сопоставляются и находятся наиболее вероятные соответствия. 

## Tracker strong
[Модуль](https://github.com/GlazkoAE/object-tracking-assignment/tree/feature/add_strong_tracker) не готов. Существующая реализация основана на алгоритме DeepSort. В классический Sort встроена нейронная сеть для определения визуального соответствия объекта на текущем кадре с объектами на предыдущих. В упомянутую выше матрицу вносятся не значения IoU, полученные от фильтра Калмана, а взвешанная сумма IoU и расстояния Махаланобиса между эмбэддингами двух объектов (или между объектом и центроидом трека). Далле венгерским алгоритмом находятся соответствия объектов и треков.

Вместо классической нейросети DeepSort была встроена ResNet34 без FC слоя, предобученная на ImageNet. Для улучшения качества трекинга планируется дообучить сеть на классификацию countryball-ов.

# Оценка качества
Оценка качества работы трекеров производится следующим образом:

После завершения работы собираются все ID треков для каждого объекта. Формируется метрика, равная отношению количества самого часто встречающегося ID к длине трека. Такая метрика считается отдельно для каждого объекта и далее усредняется для всего набора данных. Причем в случае, если чаще всего встречается None, берется второй по частоте ID.

# Результаты
Во всех наборах данных bbox отклонялся в пределах 10 пикселей
| Количество объектов | Вероятность детекции | Tracker soft |
|---------------------|----------------------|--------------|
| 5                   | 1                    | 0.737        |
| 5                   | 0.95                 | 0.685        |
| 5                   | 0.90                 | 0.527        |
| 10                  | 1                    | 0.808        |
| 10                  | 0.95                 | 0.598        |
| 10                  | 0.90                 | 0.367        |
| 20                  | 1                    | 0.601        |
| 20                  | 0.95                 | 0.425        |
| 20                  | 0.90                 | 0.237        |
