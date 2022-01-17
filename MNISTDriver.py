import NaiveBayes
import MNISTDataLoader
import time
import pandas as pd

x,y,xt,yt = MNISTDataLoader.load_train_test()

x_samps = [50,100,500,2000,10000,60000]
y_samps = [50,100,500,1000,2000,10000]
ks = [1,5,30,100]
nb_types=['bernouli', 'multinomial']

job_results = pd.DataFrame(columns=['nb_type','num_train', 'num_test', 'k', 'train_acc','test_acc', 'fit_time', 'train_s_time','test_s_time'])
for t in nb_types:
  for k in ks:
    for i,j in zip(x_samps, y_samps): 
      start = time.time()
      bayes = NaiveBayes.NBClassifier(k=k,nb_type=t)
      bayes = bayes.fit(x[0:i], y[0:i], verbose=False)
      fit = time.time()-start
      #print(f"Final class predicion: {bayes.predict(x[0:100], verbose=False)}")
      acc1 = bayes.score(x[0:i], y[0:i], verbose=True)
      score1 = time.time()-(fit+start)
      acc2 = bayes.score(xt[0:j], yt[0:j], verbose=True)
      score2 = time.time()-(score1+fit+start)
      job_results.loc[len(job_results.index)] = [t,i,j,k,acc1,acc2,fit,score1,score2]
      print(job_results)
      job_results.to_csv("MNIST_Results_in_progress")
      #print(f"Num Train Samples: {i}, Num Test Samples: {j}, k: {k}")
      #print(f"Train accuracy: {acc1}, Test Accuracy: {acc2}")
      #print(f"Fit time: {fit}, Score Train time: {score1}, Score Test Time: {score2}")
job_results.to_csv("MNIST_Results_final")