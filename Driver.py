from cgitb import text
import NaiveBayes

x_train = [
  [0,2,3],
  [1,3,4],
  [1,0,1],
  [2,1,0],
  [3,2,1],
  [0,0,1]
]

x_train_s = [
  "Hi friend, what a nice day it is today. Good luck on your exam!",
  "Dear john, we want you to enjoy your time here at the market. Have fun!",
  "Hey Jake, I was wondering if you are enjoying your time here at the company. Good luck!",
  "Free cash here at brainboob.com. Get your cash and credit now by signing up here!",
  "Loans and free credit score reports here! get a report quickly for free now!",
  "Invest at cash kind.com for a new report. please enter your number below."
]

y_train = [0,0,0,1,1,1]

bayes = NaiveBayes.NBClassifier(nb_type="multinomial")
bayes = bayes.fit(x_train, y_train, verbose=True)
print(f"Final class predicion: {bayes.predict(x_train, verbose=False)}")
bayes.score(x_train, y_train, verbose=True)


bayes = NaiveBayes.NBClassifier(nb_type="bernouli")
x_train_s = bayes.format_docs(x_train_s, verbose=False)
bayes.fit(x_train_s, y_train, verbose=True, text=True)
print(f"Final class predicion: {bayes.predict(x_train_s, verbose=False)}")
bayes.score(x_train_s, y_train, verbose=True)
