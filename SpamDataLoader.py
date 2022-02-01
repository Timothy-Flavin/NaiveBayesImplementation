import pandas as pd

def load_train_test(split=0.75, verbose = False):
  spam_data = pd.read_csv("./SPAM/SPAMData.csv")
  
  if(verbose):
    print(spam_data.head())
  y = spam_data['Category'].to_list()
  x = spam_data['Message'].to_list()

  x_train = x[0:int(len(x)*split)]
  x_test = x[int(len(x)*split):-1]
  y_train = y[0:int(len(y)*split)]
  y_test = y[int(len(y)*split):-1]
  print(y_train[0])
  if(verbose):
    print(f"x_train: {x_train[0:5]}")
    print(f"y_train: {y_train[0:5]}")
    print(f"x_test: {x_test[0:5]}")
    print(f"y_test: {y_test[0:5]}")
  
  return x_train, y_train, x_test, y_test

if __name__ == "__main__":
  load_train_test(verbose=True)