import matplotlib.pyplot as plt
import numpy as np
whether_plot_train_loss = False
train_data = "train_data/Loss_record"
eval_data = "train_data/DNN_result"
if whether_plot_train_loss:
    data = np.loadtxt(train_data+".txt")  # frame, error, predict, truth
    print(data.shape)
    plt.plot(data[:, 1])
    plt.savefig(train_data+'.png')
    plt.show()
else:
    data = np.loadtxt(eval_data+".txt")  # frame, error, predict, truth
    # data = data[220:239, :]
    print(data.shape)
    predict_max = np.amax(data[:,2])
    indice = np.where(data[:,2] == predict_max)
    print(predict_max, data[indice])
    truth_max = np.amax(data[:,3], axis=0)
    indice = np.where(data[:,3] == truth_max)
    print(truth_max, data[indice])
    # print(np.where(np.logical_and(data[:,3]>0.60, data[:,3]<=0.70)))
    print("mean error: ", abs(data[:, 1]).mean(axis=0))
    plt.scatter(data[:,3], data[:, 2])
    # plt.scatter(data[:,3], data[:, 2])
    plt.savefig(eval_data+".png")
    plt.show()


