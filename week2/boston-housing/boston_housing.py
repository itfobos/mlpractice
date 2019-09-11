from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.utils import Bunch

dataContainer: Bunch = load_boston()
X_scaled = scale(dataContainer.data)

crossValidationGenerator = KFold(n_splits=5, shuffle=True, random_state=42)

P_MIN = 1
P_MAX = 10
STEPS_AMOUNT = 200
STEP_SIZE = (P_MAX - P_MIN) / STEPS_AMOUNT

pValues = []
for stepNumber in range(0, 200):
    pValues.append(P_MIN + stepNumber * STEP_SIZE)

tuned_parameters = {
    'n_neighbors': [5],
    'weights': ['distance'],
    'metric': ['minkowski'],
    'p': pValues
}

clf = GridSearchCV(estimator=KNeighborsRegressor(),
                   param_grid=tuned_parameters,
                   cv=crossValidationGenerator,
                   scoring='neg_mean_squared_error')

clf.fit(X_scaled, dataContainer.target)

print('\n\nBest P is: %.1f' % clf.best_params_['p'])
