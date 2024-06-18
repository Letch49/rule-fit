from rule_fit import RuleFitClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('df_model.csv')

X = df[['sex_male', 'age', 'diabets', 'smoking', 'ОХС']]
y = df['y']

feature_names = ['sex_male', 'age', 'diabets', 'smoking', 'ОХС']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)


model = RuleFitClassifier(max_rules=10)

model.fit(X_train, y_train, feature_names=feature_names)

print(model.rules_)