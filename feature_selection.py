from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

def data():
    import scipy.io
    import pandas as pd
    y = scipy.io.loadmat('/data/petryshen/yoel/annealed_pgc/ml/scz_eur_pgc_annealed_centered_normed.mat')['y']
    y_ = y.T
    data_1 = pd.read_hdf('/data/petryshen/yoel/annealed_pgc/ml/scz_eur_pgc_annealed.hdf5','freesurfer_data')
    X = data_1.values
    return X, y_, data_1

Data = pe.Node(name='Data',
              interface=Function(input_names=[''],
                                output_names=['X', 'y_', 'data_1'],
                                function=data))

def demean_scale(X):
    import pandas as pd
    import numpy as np
    X = X - X.mean(0)
    X = X * 1./X.std(0)
    return X

DemeanScale = pe.Node(name='DemeanScale',
                     interface=Function(input_names=['X'],
                                       output_names=['X'],
                                       function=demean_scale))

def svc_crossval(k, X, y_):
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, 
                cv=StratifiedKFold(y_.flatten(), k, random_state = 1),
                scoring="accuracy")               
    rfecv.fit(X, y_.flatten())
    rank = rfecv.ranking_
    prediction = rfecv.predict(X)  
    n_feat = rfecv.n_features_                  
    return rank, prediction, n_feat, k

SvcCrosVal = pe.Node(name='SvcCrosVal',
                    interface=Function(input_names=['k', 'X', 'y_'],
                                      output_names=['rank', 'prediction', 'n_feat','k'],
                                      function=svc_crossval))                      
SvcCrosVal.plugin_args={'bsub_args':'-q big'}

Iternode = pe.Node(niu.IdentityInterface(fields=['k']), name='Iternode')
Iternode.iterables = [('k',[2,3,4,5,6,7,8,9,10])]                  

def get_outputs(rank, prediction, data_1, y_, n_feat, k):
    import numpy as np
    import pandas as pd
    import os                 
    indices = np.where(rank == 1)
    data_reduced = data_1.ix[:,list(indices[0])]                 
    percent_correct = sum(np.where(prediction == y_.flatten(), 1,0)) / float(len(y_.flatten()))  
    fc = "The number of chosen featuers: {}".format(n_feat)
    pc = "Number of correct case / control predictions: {}".format(percent_correct)
    to_save = np.array([fc,pc])
    os.chdir('/data/petryshen/yoel/annealed_pgc/ml')
    data_reduced.to_csv('{}_{}_{}_df.csv'.format(percent_correct,
					n_feat, k), index=None)                 
    np.savetxt('{}_{}_{}_svc.txt'.format(percent_correct,
                                        n_feat, k), to_save, fmt="%s")
                     
GetOutputs = pe.Node(name='GetOutPuts',
                    interface=Function(input_names=['rank','prediction','data_1',
                                                   'y_','n_feat','k'],
                                      output_names=[''],
                                      function=get_outputs))
                     
wf = pe.Workflow(name='wf')
wf.connect(Data,'X',DemeanScale,'X')
wf.connect(Data,'y_',SvcCrosVal,'y_')                     
wf.connect(DemeanScale,'X',SvcCrosVal,'X')
wf.connect(Iternode,'k',SvcCrosVal,'k')
wf.connect(Data,'data_1',GetOutputs,'data_1')
wf.connect(SvcCrosVal,'rank',GetOutputs,'rank')
wf.connect(SvcCrosVal,'prediction',GetOutputs,'prediction')
wf.connect(SvcCrosVal,'n_feat',GetOutputs,'n_feat')
wf.connect(SvcCrosVal,'k',GetOutputs,'k')
wf.connect(Data,'y_',GetOutputs,'y_')
wf.base_dir = '/data/petryshen/yoel/annealed_pgc/ml'
wf.run(plugin='LSF', plugin_args={'bsub':'-q big'})     
