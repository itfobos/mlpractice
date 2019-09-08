import pandas
from titanic_extract_female_name import to_female_name

DATA = pandas.read_csv('../data/titanic.csv', index_col='PassengerId')
PASSENGER_COUNT = len(DATA)

print('Male/female passengers')
print(DATA['Sex'].value_counts().sort_index())

print('\nSurvived/not percents')
SURVIVED = 1
NOT_SURVIVED = 0

survivedAndNotAmount = DATA['Survived'].value_counts() / PASSENGER_COUNT * 100
print("SURVIVED: %.2f" % survivedAndNotAmount[SURVIVED])
print("NOT_SURVIVED: %.2f" % survivedAndNotAmount[NOT_SURVIVED])

print('\nAge')
ageDimension = DATA['Age']
print("Age mean: %.2f" % ageDimension.mean())
print("Age median: %.2f" % ageDimension.median())

print('\nFirst class part')
FIRST_CLASS = 1
firstClassAmount = DATA['Pclass'].value_counts()[FIRST_CLASS] / PASSENGER_COUNT * 100
print("Age mean: %.2f" % firstClassAmount)

print('\nPearson correlation')
print("SibSp/Parch correlation: %.2f" % DATA.filter(['SibSp', 'Parch']).corr()['Parch']['SibSp'])

print('\nMost popular female name')
print('Name: ' + DATA[DATA['Sex'] == 'female']['Name'].map(to_female_name).mode()[0])
