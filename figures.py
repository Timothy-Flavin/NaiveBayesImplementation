import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

mnd = pd.read_csv("MNIST_Results_final")
txt = pd.read_csv("SPAM_Results_final.csv")

def makePlots(df, title, lentrain, lentest):
  print(f"Making plots for {title}")
  df["num_train"] = df["num_train"].replace(-1,lentrain)
  df["num_test"] = df["num_test"].replace(-1,lentest)
  
  dfb = df[df["nb_type"]=="bernouli"]
  dfm = df[df["nb_type"]=="multinomial"]

  #Plot training times of each
  dfb1 = dfb[dfb["k"]==1]
  dfm1 = dfm[dfm["k"]==1]
  plt.grid()
  plt.figure(1)
  plt.plot(dfb1['num_train'], dfb1['fit_time'])
  plt.plot(dfm1['num_train'], dfm1['fit_time'])
  plt.legend(["Bernouli", "Multinomial"])
  plt.xlabel("Number of Training Examples")
  plt.ylabel("Fit Time (s)")
  plt.title(f"Naive Bayes {title} Model Fit Time")
  plt.show()

  #Plot score times
  plt.figure(2)
  plt.grid()
  plt.plot(dfb1['num_train'], dfb1['train_s_time'])
  plt.plot(dfm1['num_train'], dfm1['train_s_time'])
  plt.legend(["Bernouli", "Multinomial"])
  plt.xlabel("Number of Training Examples")
  plt.ylabel("Train Score Time (s)")
  plt.title(f"Naive Bayes {title} Model Classification Time")
  plt.show()

  #Plot Accuracy
  plt.figure(3)
  plt.grid()
  plt.plot(dfb1['num_train'], dfb1['train_acc'])
  plt.plot(dfm1['num_train'], dfm1['train_acc'])
  plt.plot(dfb1['num_train'], dfb1['test_acc'])
  plt.plot(dfm1['num_train'], dfm1['test_acc'])
  plt.legend(["Bernouli Train Score", "Multinomial Train Score","Bernouli Test Score", "Multinomial Test Score"])
  plt.xlabel("Number of Training Examples")
  plt.ylabel("(Accuracy % )/100")
  plt.title(f"Naive Bayes {title} Model Classification Accuracy")
  plt.show()


  plt.figure(4)
  plt.grid()
  ks = pd.unique(dfb['k'])
  legendlist = list()
  colors = list()
  c=np.linspace(0.0,1,len(ks))
  for i in c:
    colors.append((i*0.8+0.2,0,i*0.9+0.10))
  for i in c:
    colors.append((0.5+i*0.5,1-0.2*i,0))
  for ii,i in enumerate(ks):
    dfm2 = dfm[dfm["k"]==i]
    dfb2 = dfb[dfb["k"]==i]
    plt.plot(np.log(dfb2['num_train']), dfb2['test_acc'], color=colors[ii])
    plt.plot(np.log(dfm2['num_train']), dfm2['test_acc'], color=colors[len(ks)+ii])
    legendlist.append(f"Bernouli k={i}")
    legendlist.append(f"Multinom k={i}")
  plt.xlabel("Ln (Number of Training Examples)")
  plt.ylabel("(Accuracy % )/100")
  plt.title(f"Naive Bayes {title} Effect of Laplace Smoothing")
  plt.legend(legendlist)
  plt.title(f"Naive Bayes {title} Effect of Laplace Smoothing")
  plt.show()


makePlots(mnd, "MNIST", 60000, 10000)
makePlots(txt, "Spam", 5572, 5572-4179)