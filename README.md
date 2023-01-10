# <p align="center">AML test task</p>
## Задача

<p align="justify"> По набору обезличенных признаков клиента необходимо определить вероятность, 
что клиент занимается отмыванием денежных средств</p>

## Описание структуры
1. eda.ipynb - анализ данных, сопровождающийся различным проверками, построением графиков, препроцесингом и т.д;
2. CheckerClass.py - класс с функциями для проверки датасета на пропуски, корреляции и т.д.
3. VisualisationClass.py - класс с функциями для построения графиков
4. DataPreprocessingClass.py - класс для препроцессинга данных
5. FeatureGeneratorClass.py - класс для генерации новых признаков
6. FeatureSelectorClass.py - класс для выбора признаков
7. artefacts - место хранения моделей, графиков и т.д.
8. config.py - файл для хранения конфигурационных данных для моделей и т.д.
9. pipeline.py - основной файл для запуска всего алгоритма
```
│   .gitignore
│   README.md
├───data
│   ├───prepared_data
│   │       features_test.pickle
│   │       features_train.pickle
│   │       submit.csv
│   │       target_train.pickle
│   │
│   └───raw_data
│           .gitkeep
│           sample_submission.csv
│           test.csv
│           train.csv
├───exploration
│   │   config.py
│   │   pipeline.py
│   ├───artefacts
│   │       features_target.png
│   │       model.pickle
│   │       train_features_target.png
│   │       train_test_features.png
│   ├───Modules
│   │   │   CheckerClass.py
│   │   │   DataPreprocessingClass.py
│   │   │   FeatureGeneratorClass.py
│   │   │   FeatureSelectorClass.py
│   │   │   VisualisationClass.py
│   ├───notebooks
│   │   │   eda.ipynb
```