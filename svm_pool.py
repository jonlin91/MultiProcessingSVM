def svm_fit(item):
    '''
    function performs model fit for one fold and validates on other fold.
    It return the validation score
    item: parameter combination it uses for the model
    '''
    k = item[0]
    c = item[1]
    g = item[2]
    
    train_index = item[3][0]
    val_index = item[3][1]
    
    #from the fold given, define the training fold and validation fold
    X_cv_tr = X_train1[train_index]
    y_cv_tr = y_train1[train_index]
    X_cv_val = X_train1[val_index]
    y_cv_val = y_train1[val_index]

    model_svc = SVC(C= c, kernel=k, gamma = g)
    model_svc.fit(X_cv_tr, y_cv_tr)
    val_score = model_svc.score(X_cv_val, y_cv_val)
    
    new_score_entry = (k, c, g, train_index, val_index, val_score)

    return new_score_entry

def svm_pool(X_train,y_train, X_test, y_test, param, n_pool):
    '''
    function does the multiprocessing
    X_train: training inputs
    y_train: training labels
    X_test: testing inputs
    y_test: testing outputs
    param: parameter dictionary that defines that parameter search space
    n_pool: number of processors to use
    '''
    #defines the KFold the same way as for single-processing to ensure reproducibility
    kf = KFold(n_splits=2)
    dict_kf = {'Train':[], 'Test':[], 'cv':[]}

    #stores the indices of the different combinations of training vs test folds in CV. 
    #Here, 'Test' actually refers to the validation set and not the final test set
    for train_index, test_index in kf.split(X_train):
        dict_kf['Train'].append(train_index)
        dict_kf['Test'].append(test_index)
        dict_kf['cv'].append([train_index, test_index])
        
    #do multiprocessing on cartesian product of all parameter value but also all CV folds
    with multiprocessing.Pool(processes = n_pool) as pool:
        result=pool.map(svm_fit, product(param['kernel'], param['C'], param['gamma'], dict_kf['cv']), chunksize=1) #chunksize??
    
    scores = pd.DataFrame(data = result, columns=['kernel','cost','gamma','train','val','score'])
    
    #do a group by to be able to take the mean validation score across the different folds for one parameter combination
    scores_cv = scores.groupby(['kernel','cost','gamma'])['score'].mean()
    scores_cv = scores_cv.reset_index()
    
    #find the maximum score
    max_score_param = scores_cv.iloc[scores_cv['score'].idxmax()]
    
    #score model
    model = SVC(kernel=max_score_param[0],C=max_score_param[1],gamma=max_score_param[2])
    model.fit(X_train,y_train)
    test_acc = model.score(X_test1,y_test1)
    
    print('Best model parameters:', max_score_param)
    print('Accuracy on test set:', model.score(X_test1,y_test1))
    
    return max_score_param, scores_cv, test_acc

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import datetime as dt
    from itertools import product
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import sys
    import multiprocessing
    
    #take as a minimum 1 processor or whatever is given on the command line
    try:
        n_pool = int(sys.argv[1])
    except:
        n_pool = 1

    k = ['poly', 'rbf', 'sigmoid']
    c = [10**x for x in range(-5,5)]
    g = [10**x for x in range(-5,5)]

    parameters = {'kernel':k, 'C':c, 'gamma':g}

    articles = pd.read_csv("datasets/SportsArticles/features.csv")
    X_origin = articles.iloc[:,3:]
    y_origin = articles.iloc[:,2]

    X_standardized = preprocessing.scale(X_origin) 
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_standardized, y_origin, test_size=0.3, random_state=42)
    y_train1 = y_train1.reset_index()
    y_train1 = y_train1['Label']
    y_test1 = y_test1.reset_index()
    y_test1 = y_test1['Label']

    results_pool = svm_pool(X_train1, y_train1, X_test1, y_test1, parameters, n_pool)
    np.save('result_pool%s'%n_pool, results_pool)