import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
Y = newsgroups.target

param_grid = {'C': np.power(10.0, np.arange(-5, 6))}
crossValidationGenerator = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(random_state=241, kernel='linear')

gridSearch = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', cv=crossValidationGenerator)
gridSearch.fit(X, Y)

# Best C is: 1.0
print('\n\nBest C is: %.1f' % gridSearch.best_params_['C'])
