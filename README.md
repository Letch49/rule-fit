# rule-fit
based on imodels (https://github.com/csinva/imodels)

## Установка

```
!pip install rulefit_vvsu
```

## Пример задачи классификации

```python

from rulefit_vvsu.main import RuleFitClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
# Загрузка данных о диабете из онлайн-источника
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Определение признаков и целевой переменной
X = df.drop('Outcome', axis=1)
y = df['Outcome']

feature_names = X.columns.tolist()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Создание и обучение модели RuleFitClassifier
model = RuleFitClassifier(max_rules=10)
model.fit(X_train, y_train, feature_names=feature_names)

# Объединение коэффициентов и правил
rules = model.rules_
coefficients = model.coef

# Сортировка правил по абсолютному значению коэффициентов
sorted_rules = sorted(zip(rules, coefficients), key=lambda x: abs(x[1]), reverse=True)

# Вывод отсортированных правил
for rule, coef in sorted_rules:
    print(f"Rule: {rule}, Coefficient: {coef}")

#>>>
#Rule: Age <= 41.5 and BMI <= 48.25 and Glucose <= 154.5, Coefficient: -0.06588467118949785
#Rule: Age <= 30.5 and Glucose <= 127.5, Coefficient: 0.0
#Rule: Age <= 41.5 and Glucose <= 127.5, Coefficient: 0.0
#Rule: Age <= 30.5 and Glucose <= 165.5, Coefficient: 0.0
#Rule: Age <= 41.5 and BMI <= 37.75 and Glucose <= 159.0, Coefficient: 0.0
#Rule: Age <= 42.5 and BMI <= 40.75 and Glucose <= 157.5, Coefficient: 0.0
#Rule: Glucose <= 127.5, Coefficient: 0.0
#Rule: BMI <= 33.25 and Glucose <= 154.5, Coefficient: 0.0
#Rule: BMI <= 46.55 and Glucose <= 123.5, Coefficient: 0.0
```

## Пример задачи регрессии
```python
from rulefit_vvsu.main import RuleFitRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных о ценах на жилье из онлайн-источника
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Определение признаков и целевой переменной
X = df.drop('medv', axis=1)
y = df['medv']

feature_names = X.columns.tolist()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Создание и обучение модели RuleFitRegressor
model = RuleFitRegressor(max_rules=10)
model.fit(X_train, y_train, feature_names=feature_names)

# Объединение коэффициентов и правил
rules = model.rules_
coefficients = model.coef

# Сортировка правил по абсолютному значению коэффициентов
sorted_rules = sorted(zip(rules, coefficients), key=lambda x: abs(x[1]), reverse=True)

# Вывод отсортированных правил
for rule, coef in sorted_rules:
    print(f"Rule: {rule}, Coefficient: {coef}")

# >>>
#Rule: nox <= 0.659 and rm > 6.6635, Coefficient: 0.15526345174185835
#Rule: lstat > 9.005, Coefficient: -0.0
#Rule: lstat > 9.715, Coefficient: 0.0
#Rule: lstat > 9.95, Coefficient: -0.0
#Rule: lstat > 9.01, Coefficient: 0.0
#Rule: lstat <= 7.865, Coefficient: -0.0
#Rule: lstat <= 9.005, Coefficient: -0.0
```