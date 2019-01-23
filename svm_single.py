'''
Method 1: Single-processing using sklearn

'''
@profile
def svm_single(X_train, y_train, X_test, y_test, param, cv):
    '''
    Loops through the different parameter combinations and gets the best score
    X_train: training inputs
    y_train: training labels
    X_test: testing inputs
    y_test: testing outputs
    param: parameter dictionary that defines that parameter search space
    cv: KFolds
    '''
    from itertools import product
    import pandas as pd
    from sklearn import metrics
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    #initialise parameters
    scores = []
    max_score = -1 #starting point since a score cannot negative
    
    #create the cartesian product of the parameters to search over
    kernel_param = product(param['kernel'], param['C'], param['gamma'])
    
    #loop through every parameter combination
    for item in kernel_param:
        k = item[0]
        c = item[1]
        g = item[2]
        
        #initialise model
        model_svc = SVC(C= c, kernel=k, gamma=g)
        new_score = cross_val_score(model_svc, X_train, y_train, cv=cv)
        new_score_entry = [k, c, g, new_score.mean()]
        scores.append(new_score_entry)
        
        #if the score is greater than previous maximum score, keep as best parameter option
        if new_score.mean() > max_score:
            max_score = new_score.mean()
            max_score_param = new_score_entry
            
    
    #train and score model with best parameters
    model = SVC(kernel=max_score_param[0],C=max_score_param[1],gamma=max_score_param[2])
    model.fit(X_train,y_train)
    
    test_acc = model.score(X_test,y_test)
    print('Best model parameters:', max_score_param)
    print('Accuracy on test set:', test_acc)

    return max_score_param, scores, test_acc

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import datetime as dt
    from itertools import product
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

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

    result_single = svm_single(X_train1, y_train1, X_test1, y_test1, parameters, cv)

    np.save('result_single',result_single)