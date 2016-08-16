def random_forest(X, y_, k):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split
    import numpy as np
    search_grid = [
        {'criterion':['gini'],  'max_features':[1,2,3,4,5,6],
         'max_depth':[1,2,3,4,5,6], 'n_estimators':np.arange(2,31,1),
         'n_jobs':[4], 'random_state':[1]},
        {'criterion':['entropy'], 'max_features':[1,2,3,4,5,6],
         'max_depth':[1,2,3,4,5,6], 'n_estimators':np.arange(2,31,1), 
         'n_jobs':[4], 'random_state':[1]}
    ]
    X_train, X_test, y_train, y_test = train_test_split(
                X, y_, test_size=.3, random_state=1)
    rfclf = GridSearchCV(RandomForestClassifier(), 
                                 search_grid, cv=k, error_score=0)
    rfclf.fit(X_train, y_train)
    search_scores = rfclf.grid_scores_
    best_score = rfclf.best_score_
    best_params = rfclf.best_params_
    prediction = rfclf.predict(X_test)
    return search_scores, best_score, best_params, prediction, y_test
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(
                X, y_, test_size=.3, random_state=1)
    brfclf = RandomForestClassifier(criterion=best_params['criterion'],
                                    max_features=best_params['max_features'],
                                    max_depth=best_params['max_depth'],
                                    n_estimators=best_params['n_estimators'],
                                    random_state=1)
                                    
                                    
                                    
    brfclf.fit(X_train, y_train)
    prediction = brfclf.predict(X_test)
    feat_importance = brfclf.feature_importances_
