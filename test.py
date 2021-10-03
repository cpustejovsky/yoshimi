from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3], [11, 12, 13]] # 2 samples, 3 features
y = [0, 1]  # classes of each sample
initialState = clf.fit(X, y)
print("initialState", initialState)

predictTrainingData = clf.predict(X) # predict classes of the training data
print("predictTrainingData", predictTrainingData)

predictNewData = clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
print("predictNewData", predictNewData)