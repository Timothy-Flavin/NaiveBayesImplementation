import NaiveBayes

x_train = [
  [0,1,1],
  [1,2,2],
  [0,0,1],
  [1,0,0],
  [2,2,1],
  [3,0,1]
]
y_train = [0,0,0,1,1,1]
bayes = NaiveBayes.NBClassifier()
bayes = bayes.fit(x_train, y_train, verbose=True)
print(bayes.predict(x_train[0], verbose=True))
print(bayes.predict(x_train[1], verbose=True))
print(bayes.predict(x_train[2], verbose=True))
print(bayes.predict(x_train[3], verbose=True))
