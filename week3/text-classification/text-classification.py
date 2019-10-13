import pandas
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
Y = newsgroups.target

# C=1.0 found via c-param-search
resultClf = SVC(random_state=241, kernel='linear', C=1.0)
resultClf.fit(X, Y)

coefsFrame = pandas.DataFrame(data=resultClf.coef_.data, index=resultClf.coef_.indices, columns=['coef'])
valuableWordsFrame = coefsFrame.applymap(lambda x: abs(x)).nlargest(10, 'coef')

feature_names = vectorizer.get_feature_names()
valuableWords = list(map(lambda ind: feature_names[ind], valuableWordsFrame.index.values))
valuableWords.sort()

print('Most valuable words: ', ' '.join(valuableWords))
