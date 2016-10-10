from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import numpy as np

if __name__ == '__main__':
    import argparse
    defstr = '(default %(default)s)'
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--datadir', type=str, help="directory for data, X and y must be in the same directory")
    parser.add_argument('-x', '--xname', type=str, help='predictors')
    parser.add_argument('-y', '--yname', type=str, help='response')
    args=parser.parse_args()
    datadir = args.datadir
    xname = args.xname
    yname = args.yname

imports_data = ['import os',
                'import pandas as pd',
                'import numpy as np']

imports_model = ['import os',
                 'import numpy as np',
                 'from sklearn import linear_model',
                 'from sklearn.model_selection import StratifiedKFold',
                 'from sklearn.metrics import roc_curve, auc, roc_auc_score']


def data(datadir, xname, yname):
    X = pd.read_csv(os.path.join(datadir, xname))
    y = np.genfromtxt(os.path.join(datadir, yname))
    X = X.values
    y[y == 1.] = 0
    y[y == 2.] = 1
    return X, y

Data = pe.Node(name='Data',
              interface=Function(
        input_names=['datadir','xname', 'yname'],
        output_names=['X','y'],
        function=data,
        imports=imports_data
    ))

Data.inputs.datadir = datadir
Data.inputs.xname = xname
Data.inputs.yname = yname

def check_assume(X):
    if not(np.allclose(X.mean(0), [0. for x in range(X.shape[1])])):
        raise Exception("Data are not demeaned")
    if not(np.allclose(X.std(0), [1. for x in range(X.shape[1])])):
        raise Exception("Data do not have standard deviation of 1")

Check_Assume = pe.Node(name='Check_Assume',
                      interface=Function(
        input_names=['X'],
        output_names=[''],
        function=check_assume,
        imports=imports_data
    ))        
        
def logistic_l1(X, y, C, n_splits=10, max_iter=1000):
    '''logistic regression, l1 penalty'''
    classifier_params = {'C': [], 'coef_':[], 'intercept_':[], 
                         'score': [], 'k':[], 'roc_auc_score': [],'n_iter_':[]}
    stratkfold = StratifiedKFold(n_splits=n_splits, shuffle=False)
    model = linear_model.LogisticRegression(C=C, penalty='l1',
                                            solver='liblinear',max_iter=max_iter)
    for k, (train, test) in enumerate(startkfold.split(X, y)):
        model.fit(X[train], y[train])
        classifier_params['C'].append(C)
        classifier_params['score'].append(model.score(X[test], y[test]))
        classifier_params['coef_'].append(model.coef_)
        classifier_params['k'].append(k)
        classifier_params['intercept_'].append(model.intercept_)
        classifier_params['n_iter_'].append(model.n_iter_)
        classifier_params['roc_auc_score'].append(roc_auc_score(y[test], model.predict(X[test])))  
    return classifier_params

LogisticL1 = pe.Node(name='LogisticL1',
                        interface=Function(
        input_names=['X', 'y', 'C',
                     'n_splits', 'max_iter'],
        output_names=['classifier_params'],
        function=logistic_l1,
        imports=imports_model
    ))

Iternode = pe.Node(niu.IdentityInterface(fields=['C']), name='Iternode')
Iternode.iterables = [('C', np.arange(0,1,.0005))]

def save(classifier_params, datadir):
    os.chdir(datadir)
    os.makedirs('l1_out')
    out = pd.DataFrame(classifier_params)
    os.chdir('l1_out')
    out.to_csv('{}_logreg_model.csv'.format(classifier_params['C']), index=None)

Save = pe.Node(name='Save',
              interface=Function(
        input_names=['classifier_params','datadir'],
        output_names=[''],
        funciton=save,
        imports=imports_data
    ))   

Save.inputs.datadir = datadir
    
wf = pe.Workflow(name='log_test')
wf.connect(Data, 'X', Check_Assume, 'X')
wf.connect(Data, 'X', LogisticL1, 'X')
wf.connect(Data, 'y', LogisticL1, 'y')
wf.connect(Iternode, 'C', LogisticL1, 'C')
wf.connect(LogisticL1, 'classifier_params', Save, 'classifier_params')
