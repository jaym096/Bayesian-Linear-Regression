import matplotlib.pyplot as plt
import numpy as np
import random

def getData(filename):
    data_mtx = []
    with open(filename,'r') as f:
        lines = [lines for lines in f.read().split("\n")][:-1]
        for each_line in lines:
            each_line = [float(value) for value in each_line.split(",")]
            data_mtx.append(each_line)
    #return data_mtx
    return np.asmatrix(data_mtx)

def getData_regressors(filename):
    data_mtx = []
    with open(filename,'r') as f:
        lines = [lines for lines in f.read().split("\n")][:-1]
        for each_line in lines:
            data_mtx.append(float(each_line))
    #return data_mtx
    return np.asmatrix(data_mtx)

#------------------------ Experiment 1 --------------------------- 
def convert_to_list_1(narray):
    arr = []
    for i in range(len(narray)):
        arr.append(narray[i].tolist())
    return arr

def calculate_MSE_t1(W,dataSamples,target):
    mse = []
    for i in range(len(W)):
        phi = dataSamples
        N = len(dataSamples)
        #phi_transpose = np.matrix.transpose(dataSamples)
        phi_W = np.matmul(phi,W[i])
        t = target
        diff = phi_W - t
        sqr = np.square(diff)
        sqr_err = np.sum(sqr)/N
        mse.append(sqr_err)
    return mse

def calculate_MSE_t2(W,dataSamples,target):
    phi = dataSamples
    N = len(dataSamples)
    #phi_transpose = np.matrix.transpose(dataSamples)
    phi_W = np.matmul(phi,W)
    t = target
    diff = phi_W - t
    sqr = np.square(diff)
    mse = np.sum(sqr)/N
    return mse

def calculate_W_t1(lambda_r,train_x,train_y):
    W = []
    for i in range(len(lambda_r)):
        phi_transpose = np.matrix.transpose(train_x)
        phi = train_x
        I = np.identity(len(np.matmul(phi_transpose,phi)))
        W_part1 = np.linalg.inv((lambda_r[i]*I) + (np.matmul(phi_transpose,phi)))
        W_part2 = np.matmul(phi_transpose,train_y)
        W_star = np.matmul(W_part1,W_part2)
        W.append(W_star)
    return W
    
def calculate_W_t2(lambda_r,train_x,train_y):
    phi_transpose = np.matrix.transpose(train_x)
    phi = train_x
    phi_t_phi = np.matmul(phi_transpose,phi)
    I = np.identity(len(phi_t_phi))
    W_part1 = np.linalg.inv((lambda_r*I) + (phi_t_phi))
    W_part2 = np.matmul(phi_transpose,train_y)
    W_star = np.matmul(W_part1,W_part2)
    return W_star

def getRandomSample(train_x,train_y,n):
    samp_x = []
    samp_y = []
    sample_size = int(len(train_x) * n)
    temp_array = list(range(0,len(train_x)))
    c = random.sample(temp_array,sample_size)
    for index in c:
        samp_x.append(train_x[index])
        samp_y.append(train_y[index])
    samp_x = np.stack(samp_x, axis=0)
    samp_y = np.matrix.transpose(np.stack(samp_y, axis=1))
    return samp_x,samp_y,sample_size

def getLambda(lambda_r, test_mse):
    #get index of minimum and maximum mse
    index_min = test_mse.index(min(test_mse))
    index_max = test_mse.index(max(test_mse))
    just_right_lambda = lambda_r[index_min]
    too_small_lambda = 1
    too_large_lambda = lambda_r[index_max]
    #too_large_lambda = 100
    lambdas = [too_small_lambda] + [just_right_lambda] + [too_large_lambda]
    return lambdas

def getAverageMSE(lambda_mtx):
    list_avgMSE_Lambda = []
    for list_mtx in lambda_mtx:
        mtrx = np.stack(list_mtx, axis=0)
        summ = np.mean(mtrx, axis=0)
        list_avgMSE_Lambda.append(summ)
    return list_avgMSE_Lambda

def calculate_SD(lambda_mtx):
    list_sd_forech_lambda = []
    for listmtx in lambda_mtx:
        mtrx = np.stack(listmtx, axis=0)
        sd = np.std(mtrx, axis=0)
        list_sd_forech_lambda.append(sd)
    return list_sd_forech_lambda

def plot_t1(train_MSE, test_MSE, lambda_r):
    plt.plot(lambda_r,train_MSE,'-r',label='train_MSE')
    plt.plot(lambda_r,test_MSE,'-g',label='test_MSE')
    plt.xlim(min(lambda_r)-10,max(lambda_r)+10)
    plt.legend(loc='upper left')
    plt.xlabel('Values of Lambda')
    plt.ylabel('Mean Squared Error')
    plt.title("Plot of training set MSE and test set MSE as a function regularization patameter λ \n",
              fontsize="small")
    plt.grid(True)
    plt.show()

def plot_t2(Avg_mse,sd,dataset_size,represent_lambda):
    plt.errorbar(dataset_size,Avg_mse[0],sd[0],label='too small')
    plt.errorbar(dataset_size,Avg_mse[1],sd[1],label='just right')
    plt.errorbar(dataset_size,Avg_mse[2],sd[2],label='too large')
    plt.xlim(min(dataset_size),max(dataset_size))
    plt.legend(loc='upper right')
    plt.xlabel('size of sample data set')
    plt.ylabel('Mean Squared Error')
    plt.title("learning curve plots as a function of size of training set for different values of λ",
              fontsize="small")
    plt.grid(True)
    plt.show()
    
#------------------------ Experiment 2 ---------------------------
#-------------------------- Task 3.1 -----------------------------
def convert_to_list(train_x,train_y):
    t_x = []
    t_y = []
    for i in range(len(train_x)):
        t_x.append(train_x[i].tolist()[0])
        t_y.append(train_y[i].tolist()[0])
    return t_x,t_y

def generateKfold(size,no_of_fold,train_x,train_y):
    t_x, t_y = convert_to_list(train_x,train_y)
    final_fold_x = []
    final_fold_y = []
    i_fold_x = []
    i_fold_y = []
    for i in range(len(train_x)):
        if((i%size==0) and (i>0)):
            final_fold_x.append(i_fold_x)
            final_fold_y.append(i_fold_y)
            i_fold_x = []
            i_fold_y = []
            i_fold_x.append(t_x[i])
            i_fold_y.append(t_y[i])
        else:
            i_fold_x.append(t_x[i])
            i_fold_y.append(t_y[i])
    final_fold_x.append(i_fold_x)
    final_fold_y.append(i_fold_y)
    return final_fold_x, final_fold_y

def kfold_train_test(kfold,k):
    test_data = kfold[k]
    train_set = []
    for i in range(len(kfold)):
        if(i==k):
            continue
        else:
            train_set += kfold[i]
    train_data = np.stack(train_set, axis=0)
    test_data = np.stack(test_data, axis=0)
    return train_data, test_data

def calculateMS(alpha, beta, phi_t_phi, phi_transpose, t):
    s_part1 = beta*phi_t_phi
    I = np.identity(len(phi_t_phi))
    s_part2 = alpha * I
    sn_inv = (s_part1 + s_part2)
    sN = np.linalg.inv(sn_inv)
    mN1 = beta*(np.linalg.multi_dot([sN,phi_transpose,t]))
    return sN,mN1