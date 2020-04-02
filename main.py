import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# read file data
file = pd.read_csv('wine_dataset.csv')

# convert string to number
file['style'] = file['style'].replace('red', 0)
file['style'] = file['style'].replace('white', 1)

# split predict and target variable
y = file['style']
x = file.drop('style', axis = 1)

# classifier test and training variables
x_training, x_test, y_training, y_test = train_test_split(x, y, test_size = .3)

# Set ExtraTreesClassifier algorithm for predict
model = ExtraTreesClassifier(n_estimators = 100)
model.fit(x_training, y_training)

# calculate accuracy
print "\nACCURACY:"
accuracy = model.score(x_test, y_test) * 100
print accuracy, "%"


print"\nPREDICTED:"
result = model.predict(x_test[200:205])
print result

print "\nTRUE RESULT:"
trueResult = y_test[200:205]
print trueResult


