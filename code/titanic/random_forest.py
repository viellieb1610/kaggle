import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# get data
train = pd.read_csv('../../data/titanic/train.csv')
test = pd.read_csv('../../data/titanic/test.csv')

# prep data
le = LabelEncoder()
for col in ['Sex', 'Embarked']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

X_train = train.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train['Survived']

test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# create classifier and predict output
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(test)

result = test[['PassengerId']].copy()
result['Survived'] = y_pred

result.to_csv('../../data/titanic/random_forest.csv', index=False)