import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

DATA = pandas.read_csv('../data/titanic.csv', index_col='PassengerId')

nan_filtered_data = DATA[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
input_objects = nan_filtered_data[['Pclass', 'Fare', 'Age', 'Sex']].replace({'Sex': {'male': 1, 'female': 0}})
answers = nan_filtered_data[['Survived']]

classifier = DecisionTreeClassifier(random_state=241)
classifier.fit(input_objects, answers)

print('feature importances:', classifier.feature_importances_)

columnIndexes = np.argsort(classifier.feature_importances_)[::-1]
print(input_objects.columns[columnIndexes])
