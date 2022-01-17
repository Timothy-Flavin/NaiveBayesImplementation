import NaiveBayes

x_train = [
  [0,1,2],
  [1,2,2],
  [0,0,1],
  [1,0,0],
  [2,2,0],
  [3,0,1]
]
y_train = [0,0,0,1,1,1]
bayes = NaiveBayes.NBClassifier(nb_type="multinomial")
bayes = bayes.fit(x_train, y_train, verbose=True)
print(f"Final class predicion: {bayes.predict(x_train[0], verbose=True)}")
print(f"Final class predicion: {bayes.predict(x_train[1], verbose=True)}")
print(f"Final class predicion: {bayes.predict(x_train[2], verbose=True)}")
print(f"Final class predicion: {bayes.predict(x_train[3], verbose=True)}")
print(f"Final class predicion: {bayes.predict(x_train[4], verbose=True)}")
print(f"Final class predicion: {bayes.predict(x_train[5], verbose=True)}")
