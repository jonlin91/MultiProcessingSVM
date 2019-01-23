#gridsearch
def gridsearch(X_train, y_train, X_test, y_test, parameters, n_pool, cv):
    '''
    function uses GridSearchCV and trains/scores final model
    X_train: training inputs
    y_train: training labels
    X_test: testing inputs
    y_test: testing outputs
    param: parameter dictionary that defines that parameter search space
    n_pool: number of processors to use
    '''
    from itertools import product
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    model_svc = SVC()
    model = GridSearchCV(model_svc, parameters, cv=cv, n_jobs = n_pool)
    model.fit(X_train, y_train)
    max_score_param = [model.best_estimator_.kernel,model.best_estimator_.C,model.best_estimator_.gamma, model.best_score_]
    print('Best model parameters:', max_score_param)
    
    model_scores = model.cv_results_
    
    #score model
    model = SVC(kernel=max_score_param[0],C=max_score_param[1],gamma=max_score_param[2])
    model.fit(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    print('Accuracy on test set:', test_acc)
    return max_score_param, model_scores, test_acc

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import datetime as dt
    from itertools import product
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    from sklearn import preprocessing
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

    cv = KFold(n_splits=2, random_state=0, shuffle=False)

    articles = pd.read_csv('datasets/SportsArticles/features.csv')
    X_origin = articles.iloc[:,3:]
    y_origin = articles.iloc[:,2]

    X_standardized = preprocessing.scale(X_origin)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_standardized, y_origin, test_size=0.3, random_state=42)

    result_grid = gridsearch(X_train1, y_train1, X_test1, y_test1, parameters, n_pool,cv)
    np.save('result_grid%s'%n_pool,result_grid)

