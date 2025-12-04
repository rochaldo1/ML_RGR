# Reinforcement Learning Project (CartPole + MountainCar)

## Описание

Этот проект демонстрирует базовые алгоритмы обучения с подкреплением на примерах классических сред из OpenAI Gym/Gymnasium:

1. **CartPole-v1**

   * **Hill Climbing (Восхождение на вершину)**
   * **Policy Gradient (Градиент стратегии)**

2. **MountainCar-v0**

   * **Policy Gradient (Градиент стратегии)**

Программа включает GUI на PyQt5 для настройки гиперпараметров, запуска обучения и отображения результатов.

---

## Структура проекта

```
ML_RGR/
├── cartpole/
│   ├── cartpole_hill_climbing.py
│   ├── cartpole_policy_gradient.py
│   └── __init__.py
├── mountaincar/
│   ├── mountaincar_policy_gradient.py
│   └── __init__.py
├── gui/
│   ├── gui_runner.py
│   └── __init__.py
├── utils/
│   ├── plots.py
│   └── __init__.py
└── README.md
```

---

## Требования

* Python 3.10+
* PyTorch
* Gymnasium
* PyQt5
* Matplotlib
* NumPy

Установка зависимостей:

```bash
pip install torch gymnasium pyqt5 matplotlib numpy
```

---

## Запуск

1. Активируйте виртуальное окружение (опционально):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux / macOS
```

2. Запуск GUI:

```bash
python -m gui.gui_runner
```

3. В GUI:

* Настройте гиперпараметры для каждого алгоритма.
* Нажмите кнопку запуска соответствующего алгоритма.
* Для MountainCar PG будет отображаться прогресс внизу окна.

---

## Реализованные алгоритмы

### Hill Climbing (CartPole)

* Случайная инициализация параметров.
* Добавление случайного шума к политике.
* Выбор параметров с наилучшей суммарной наградой.

### Policy Gradient (CartPole и MountainCar)

* REINFORCE алгоритм.
* Нормализация вознаграждений (returns) для стабильности обучения.
* Для MountainCar добавлен **reward shaping**: бонус за скорость, бонус за позицию.

---

## Особенности GUI

* Поля для ввода количества эпизодов и скорости обучения.
* QLabel для отображения текущего прогресса обучения (для MountainCar PG).
* После завершения обучения строится график с суммарной наградой за эпизод.

---

## Примечания

* MountainCar-v0 имеет очень медленное начальное обучение, так как базовая награда -1 за шаг.
* Для ускорения обучения добавлено reward shaping (бонус за скорость).
