import supporting_functions as func
import numpy as np
import random
import timeit
import sys
     
#=============================== Experiment 1 ===================================
#-----------------------------------Task 1--------------------------------------- 
def Exp1_task1(lambda_r,train_x,train_y,test_x,test_y):
    #Task1: Effects of regularization parameters 
    W = func.calculate_W_t1(lambda_r,train_x,train_y)
    training_mse = func.calculate_MSE_t1(W,train_x,train_y)
    test_mse = func.calculate_MSE_t1(W,test_x,test_y)
    min_index = test_mse.index(min(test_mse))
    print("minimum mse: ",min(test_mse))
    lamb = lambda_r[min_index]
    print("lambda value: ",lamb)
    func.plot_t1(training_mse,test_mse,lambda_r)
#--------------------------------------------------------------------------------

#-----------------------------------Task 2--------------------------------------- 
def Exp1_task2(lambda_r,train_x,train_y,test_x,test_y):
    #Task2: Effects of No.of examples and Features
    #from Experiment1 task1
    W = func.calculate_W_t1(lambda_r,train_x,train_y)
    test_mse = func.calculate_MSE_t1(W,test_x,test_y)
    #select 3 representative values of lambda
    lambda_mtx = []
    represent_lambda = func.getLambda(lambda_r,test_mse)
    sampling_factor = np.arange(0.1,1.1,0.1)
    #for each selected lambda
    for rl in represent_lambda:
        print(rl)
        mse_mtx = []
        #we loop through 0 to 10
        for i in range(0,10):
            iter_array = []
            dataset_size = []
            #generating random subsets using sampling factor
            for n in sampling_factor:
                #get Sample subset
                samp_x, samp_y, sample_size = func.getRandomSample(train_x,train_y,n)
                dataset_size.append(sample_size)
                w = func.calculate_W_t2(rl,samp_x,samp_y)
                mse = func.calculate_MSE_t2(w,test_x,test_y)
                iter_array.append(mse)
            mse_mtx.append(iter_array)
        lambda_mtx.append(mse_mtx)
    #calcualting average MSE
    Avg_mse = func.getAverageMSE(lambda_mtx)
    #calculate Standard deviation
    sd = func.calculate_SD(lambda_mtx)
    #convert to list of list
    sd = func.convert_to_list_1(sd)
    Avg_mse = func.convert_to_list_1(Avg_mse)
    func.plot_t2(Avg_mse,sd,dataset_size,represent_lambda)
#--------------------------------------------------------------------------------

def Experiment1(train_x,train_y,test_x,test_y,task):
    """
    study the effects of no.of examples, no.of features and regularization parameter
    on the performance of the corresponding algorithms
    """
    lambda_r = np.array(np.arange(0,151,1))
    if(task=='1'):
        #Task1: Effects of regularization parameters
        Exp1_task1(lambda_r,train_x,train_y,test_x,test_y)
    if(task=='2'):
        #Task2: Effects of No.of examples
        Exp1_task2(lambda_r,train_x,train_y,test_x,test_y)
#================================================================================
    
#=============================== Experiment 2 ===================================
#-----------------------------------Task 1---------------------------------------
#------------------------------Cross Validation----------------------------------      
def Exp2_t1(train_x,train_y,test_x,test_y):
    no_of_fold = 10
    size = round(len(train_x)/(no_of_fold))
    kfold_x, kfold_y = func.generateKfold(size,no_of_fold,train_x,train_y)
    lambda_r = np.array(np.arange(0,151,1))
    list_of_MSE = [] #holds avg MSE for a particular lambda
    for l in lambda_r:
        kfold_mse = [] #holds mse for each fold for a particular value of lambda
        for i in range(len(kfold_x)):
            samp_train_x, samp_test_x = func.kfold_train_test(kfold_x,i)
            samp_train_y, samp_test_y = func.kfold_train_test(kfold_y,i)
            w = func.calculate_W_t2(l, samp_train_x, samp_train_y)
            mse = func.calculate_MSE_t2(w,samp_test_x,samp_test_y)
            kfold_mse.append(mse)
        avg_mse = np.average(kfold_mse)
        list_of_MSE.append(avg_mse)
    #Get the index of the minimum mse from the list
    min_index = list_of_MSE.index(min(list_of_MSE))
    final_lambda = lambda_r[min_index]
    print("best lambda we got is: ",final_lambda)
    print("The associated MSE is: ",min(list_of_MSE))
    #retrain on the entire set
    W = func.calculate_W_t2(final_lambda, train_x, train_y)
    mse = func.calculate_MSE_t2(W,test_x,test_y)
    print("After retraining and evaluating on test set MSE we get is: ",mse)
#--------------------------------------------------------------------------------

#-----------------------------------Task 2---------------------------------------
#------------------------------Evidence Function---------------------------------
def Exp2_t2(train_x,train_y,test_x,test_y):
    phi = train_x
    t = train_y
    max_iter = 100
    N = len(phi)
    beta = random.uniform(1,10)
    alpha = random.uniform(1,10)
    phi_transpose = np.matrix.transpose(phi)
    phi_t_phi = np.matmul(phi_transpose,phi)
    iteration = 0
    eigenValues_0 = np.linalg.eigvalsh(phi_t_phi)
    for i in range(0,max_iter):
        iteration = iteration + 1
        eigenValues = eigenValues_0 * beta
        sN,mN = func.calculateMS(alpha, beta, phi_t_phi, phi_transpose, t)
        gamma = np.sum(eigenValues/(eigenValues + alpha))
        mN_transpose = np.matrix.transpose(mN)
        mN_t_mN = np.sum(np.matmul(mN_transpose,mN))
        alpha_next = gamma/mN_t_mN
        #phi_t_mN = np.matmul(phi_transpose,mN)
        phi_mN = np.matmul(phi,mN)
        sqr_err = np.square(t-phi_mN)
        beta_inv = (1/(N-gamma))*(np.sum(sqr_err))
        beta_next = 1/beta_inv
        if(abs(beta-beta_next)>0.000001 and abs(alpha-alpha_next)>0.000001):
            alpha = alpha_next
            beta = beta_next
        else:
            alpha = alpha_next
            beta = beta_next
            print("converged after iteration no: ",iteration)
            break
    lamb = alpha/beta
    w = func.calculate_W_t2(lamb, train_x, train_y)
    mse = func.calculate_MSE_t2(w, test_x, test_y)
    print("value of alpha: ",alpha)
    print("value of beta: ",beta)
    print("value of lambda: ",lamb)
    print("MSE obtained:",mse)
#--------------------------------------------------------------------------------    

def Experiment2(train_x,train_y,test_x,test_y,filename,task):
    """
    Investigate two methods for model selection in linear regression (evidence maximization and 
    cross validation)
    """
    if(task=='1'):
        #task: 3.1 Model selection using cross validation
        t1 = timeit.default_timer()
        print("file: ",filename)
        Exp2_t1(train_x,train_y,test_x,test_y)
        t2 = timeit.default_timer()
        t = t2-t1
        print("run time: ",t)
    if(task=='2'):
        #task: 3.2 Model selection using Evidence function
        t1 = timeit.default_timer()
        print("file: ",filename)
        Exp2_t2(train_x,train_y,test_x,test_y)
        t2 = timeit.default_timer()
        t = t2-t1
        print("run time: ",t)
#================================================================================
    
if __name__ == "__main__":
    filename = sys.argv[1]
    train_x = func.getData("pp2data/train-"+filename+".csv")
    train_y = func.getData_regressors("pp2data/trainR-"+filename+".csv")
    test_x = func.getData("pp2data/test-"+filename+".csv")
    test_y = func.getData_regressors("pp2data/testR-"+filename+".csv")
    """Since t is read as a row vector we have to convert it to a column vector 
       (which is what it's actual shape should be)"""
    train_y = np.matrix.transpose(train_y)
    test_y = np.matrix.transpose(test_y)
    exp = sys.argv[2]
    task = sys.argv[3]
    if(exp == '1'):
        Experiment1(train_x,train_y,test_x,test_y,task)
    if(exp == '2'):
        Experiment2(train_x,train_y,test_x,test_y,filename,task)