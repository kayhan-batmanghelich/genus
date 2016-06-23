from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio


def set_directory(d_dir):
    d_dir = str(d_dir)
    if d_dir[-1] != '/':
        return d_dir + '/'
    else:
        return d_dir

SetDirectory = pe.Node(name='SetDirectory',
                      interface=Function(input_names=['d_dir'],
                                        output_names=['d_dir'],
					overwrite=True,
                                        function=set_directory))
                                        
def data_files(dir_):
    lhCortexThicknessFn = dir_+'/ADNI_lh_cortex_thick.csv' 
    rhCortexThicknessFn = dir_+'/ADNI_rh_cortex_thick.csv'
    subCorticalVolFn = dir_+'/ADNI_aseg_vols.csv'
    classLabelFn = dir_+'/ADNI_diagnosis.txt'
    apoeVariantFn = dir_+'/APOE.csv'
    natureSnpImputedListADvsCN_fam = dir_+'/nature.snp_imputedList_onlyADvsCN.fam'
    natureSnpImputedListADvsCN_bim = dir_+'/nature.snp_imputedList_onlyADvsCN.bim'
    natureSnpImputedListAdvsCN_h5 = dir_+'/nature.snp_imputedList_onlyADvsCN.h5'
    lowPValSnporgListAdvsCN_fam = dir_+'/lowPVal.snp_orgList_onlyADvsCN.fam'
    lowPValSnporgListAdvsCN_bim = dir_+'/lowPVal.snp_orgList_onlyADvsCN.bim'
    #lowPValSnporgListAdvsCN_h5 = ''
    indSnpImputedListADvsCN_fam = dir_+'/independent.snp_imputedList_onlyADvsCN.fam'
    indSnpImputedListADvsCN_bim = dir_+'/independent.snp_imputedList_onlyADvsCN.bim'
    indSnpImputedListADvsCN_h5 = dir_+'/independent.snp_imputedList_onlyADvsCN.h5'
    return ([lhCortexThicknessFn, rhCortexThicknessFn, subCorticalVolFn, apoeVariantFn], classLabelFn,
           [natureSnpImputedListADvsCN_fam, natureSnpImputedListADvsCN_bim, natureSnpImputedListAdvsCN_h5])

DataFiles = pe.Node(name='DataFiles',
                   interface=Function(input_names=['dir_'],
                                     output_names=['files','classlabel','snp'],
                                     function=data_files))
def brain_thickness_strings():
    
    # right and left sides should be the same
    # cts = cortical thickness string
    # sts = subcortical thickness string
    
    cts_headers = ['bankssts','caudalanteriorcingulate','caudalmiddlefrontal',
                   'cuneus','entorhinal','fusiform','inferiorparietal','inferiortemporal',
                   'isthmuscingulate','lateraloccipital','lateralorbitofrontal','lingual',
                   'medialorbitofrontal','middletemporal','parahippocampal','paracentral',
                   'parsopercularis','parsorbitalis','parstriangularis','pericalcarine',
                   'postcentral','posteriorcingulate','precentral','precuneus','rostralanteriorcingulate',
                   'rostralmiddlefrontal','superiorfrontal','superiorparietal','superiortemporal',
                   'supramarginal','frontalpole','temporalpole','transversetemporal','insula']
    
    sts_headers = ['Left-Cerebral-White-Matter','Left-Cerebral-Cortex','Left-Lateral-Ventricle',
                   'Left-Inf-Lat-Vent','Left-Cerebellum-White-Matter','Left-Cerebellum-Cortex',
                   'Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','3rd-Ventricle',
                   '4th-Ventricle','Left-Hippocampus','Left-Amygdala','Right-Cerebral-White-Matter',
                   'Right-Cerebral-Cortex','Right-Lateral-Ventricle','Right-Inf-Lat-Vent',
                   'Right-Cerebellum-White-Matter','Right-Cerebellum-Cortex','Right-Thalamus-Proper',
                   'Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','ICV'] 
    
    return (['Left-%s' % i for i in cts_headers],['Right-%s' % i for i in cts_headers],sts_headers)

BrainThickness = pe.Node(name='BrainThickness',
                        interface=Function(input_names=[],
                                          output_names=['headers'],
                                          function=brain_thickness_strings))

def class_label(classlabel):
    # diag.iloc[:,0] and diag.iloc[:,1] == diagnosisLblID
    # diag.iloc[:,4] == diagnosisLbl
    import numpy as np
    import pandas as pd
    classlabel = pd.read_csv(classlabel)
    diag = classlabel[classlabel['dxBase'] == classlabel['dxCurrent']]
    diag = diag[diag['dxBase'] != 2]
    diag['diagnoisiLBl'] = np.where(diag['dxBase'] == 1, 0,1)
    return diag

ClassLabel = pe.Node(name='ClassLabel',
                    interface=Function(input_names=['classlabel'],
                                      output_names=['diag'],
				      overwrite=True,
                                    function=class_label))

def read_data(files, diag):
    import pandas as pd
    import numpy as np
    data_frames = []
    for i in files:
        in_file = pd.read_csv(i)
        subs = pd.concat([in_file.iloc[:,0], diag.iloc[:,0]], axis=1) 
        ids_and_matrix_matched = in_file[in_file.iloc[:,0] == subs.iloc[:,1]] # equivalent to matrix, rhIDs, lhIDs
        ids_and_matrix_not_matched = in_file[in_file.iloc[:,0] != subs.iloc[:,1]]
        ids_and_matrix_not_matched['idx'] = ids_and_matrix_not_matched.index # idx column == idx in matlab script
        matched_diag_idx = np.where(subs.iloc[:,0] == subs.iloc[:,1], subs.index, 'no_match')
        matched_diag = [] # equivalent to [lh,rh,apoe,aseg]_diag in matlab script
        for i in matched_diag_idx:
            if i != 'no_match':
                matched_diag.append(diag['diagnoisiLBl'][int(i)])
        matched_diag = pd.DataFrame({'diag':matched_diag})
        data_frames.append((ids_and_matrix_matched, ids_and_matrix_not_matched, matched_diag))
    return data_frames

ReadData = pe.Node(name='ReadData',
                  interface=Function(input_names=['files','diag'],
                                    output_names=['data_frames'],
                                    function=read_data))
def covs(classlabel, dir_):
    from scipy.io import loadmat
    import numpy as np
    import pandas as pd
    covList = {'age':[], 'educ':[], 'sex':[], 'handedness':[]}
    mertData = loadmat(dir_+'adni_thickness_data_for_kayhan.mat')
    covCell = {}
    covData = np.zeros((len(classlabel.iloc[:,0]), len(covList)))
    for y in covList:
        for i in mertData['adni_data'][y]:
            for x in i:
                for j in x:
                        covList[y].append(str(j[0]))
    sid = []                    
    for i in mertData['adni_data']['sid_cell']:
            for x in i:
                for j in x:
                    for k in j:
                        sid.append(str(k[0]))
    cov_dataframe = pd.DataFrame(covList)
    cov_dataframe['sid_cell'] = sid
    sub_cov_df = pd.concat([cov_dataframe, classlabel.iloc[:,1]], axis=1)
    sub_cov_df = sub_cov_df[sub_cov_df['sid_cell'] == sub_cov_df['SID']]
    return sub_cov_df.drop('SID', 1)

Covs = pe.Node(name='Covs',
              interface=Function(input_names=['classlabel','dir_'],
                                output_names=['sub_cov_df'],
                                function=covs))

def endo_matrix(lh_mat, rh_mat, sub_mat, lh, rh, subc, diag):
    import pandas as pd
    headers=[lh[0],rh[1],subc[2]]
    endo_headers=[]
    for i in headers:
        endo_headers.extend(i)
    endoPhenMatrix = pd.concat([lh_mat[0][0].iloc[:,1:], 
                                rh_mat[1][0].iloc[:,1:], 
                                sub_mat[2][0].iloc[:,1:]], axis=1)
    endoPhenMatrix.columns=endo_headers
    endo_mat_mean = endoPhenMatrix[diag.iloc[:,4] == 0].mean()
    endoPhenMatrix = endoPhenMatrix.iloc[:,] - endo_mat_mean
    endo_mat_sd = 1 / endoPhenMatrix[diag.iloc[:,4] == 0].std()
    endoPhenMatrix = endoPhenMatrix.iloc[:,] * endo_mat_sd
    return endoPhenMatrix

EndoMatrix = pe.Node(name='EndoMatrix',
                    interface=Function(input_names=['lh_mat','rh_mat','sub_mat',
                                                   'lh', 'rh', 'subc','diag'],
                                      output_names=['endo_matrix'],
                                      function=endo_matrix))

# there are no files for block 3 and 4.0, starting on 4.1
# but it looks like the same funciton (below) will take care of those blocks

def genotype_data(classlabel, fam, bim, h5, output_dir):
    import pandas as pd
    import numpy as np
    import h5py
    import os
    output_dir = str(output_dir)
    if output_dir[-1] != '/':
        output_dir += '/'
    nature_imp = pd.read_csv(fam[0],sep=' ', 
                                    names=['familyID','individualID',
                                           'paternalID','maternalID',
                                           'sex','phenotype'])
    snp_nature_imp = pd.read_csv(bim[1],sep='\t', 
                                        names=['chrNum','snpID',
                                               'genDist','physloc',
                                                'A1','A2'])
    h5_nature_imp = h5py.File(h5[2])['genotype']
    h5_nature_imp_df = pd.DataFrame(np.transpose(h5_nature_imp),
                                          columns=[x for x in range(len(h5_nature_imp))])
    #h5_nature_imp_zeros = pd.DataFrame(np.zeros((len(data.iloc[:,0]), len(h5_nature_imp))),
                                      #columns=[x for x in range(len(h5_nature_imp))])
    #h5_nature_imp_zeros = h5_nature_imp_zeros.iloc[:].astype(int)
    nature_imp['par_id'] = nature_imp['individualID'].str[4:]
    nature_imp_matched = nature_imp.set_index('par_id', drop=False).loc[classlabel.iloc[:,1]]
    nature_imp_matched.index = [x for x in range(len(nature_imp_matched.iloc[:,1]))]
    h5_nature_imp_df['par_id'] = nature_imp['individualID'].str[4:]
    h5_nature_imp_df_matched = h5_nature_imp_df.set_index('par_id', drop=False).loc[classlabel.iloc[:,1]]
    h5_nature_imp_df_matched.index = [x for x in range(len(h5_nature_imp_df_matched.iloc[:,1]))]
    h5_nature_imp_df_matched = h5_nature_imp_df_matched.fillna(0) 
    h5_nature_imp_df_matched.to_hdf(output_dir+'h5_nature_imp_matched.h5',key='MATCHED', mode='w')
    return (nature_imp_matched)
    
GenoTypeData = pe.Node(name='GenoTypeData',
                      interface=Function(input_names=['classlabel','fam','bim','h5', 'output_dir'],
                                        output_names=['nature_imp_matched'],
                                        function=genotype_data))

def feature_selection(endo_mat, class_label, method):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    import pandas as pd
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.datasets import make_classification
    #% matplotlib inline
    import matplotlib.pyplot as plt
    class_label = class_label.iloc[:,4]
    if method == 1 or method == None:
        # method 1, univariate test
        X_new = SelectKBest(f_regression, k=10).fit_transform(endo_mat, class_label)
        return pd.DataFrame(X_new)
    elif method == 2:
        # method 2, l1-model based regression
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(endo_mat, class_label)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(endo_mat)
        return pd.DataFrame(X_new)
    elif method == 3:
        # method 3, tree based
        model = ExtraTreesClassifier()
        model.fit(endo_mat, class_label)
        model = SelectFromModel(model, prefit=True)
        X_new = model.transform(endo_mat)
        return pd.DataFrame(X_new) 
    elif method == 4:
        # method 4, recursive feature elimination
        endo_mat, class_label = make_classification(n_samples=406, n_features=94, n_informative=4,
                                   n_redundant=2, n_repeated=0, n_classes=8,
                                   n_clusters_per_class=1, random_state=0)
        svc = SVC(kernel="linear") # haven't tried polynomial
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(class_label, 2),
                      scoring='accuracy')
        rfecv.fit(endo_mat, class_label)
        print("Optimal number of features : %d" % rfecv.n_features_)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

FeatureSelection = pe.Node(name='FeatureSelection',
                          interface=Function(input_names=['endo_mat','class_label','method'],
                                            output_names=['features'],
                                            function=feature_selection)) 


def norm_and_join(endo_mat, features, covdata, apoedata, classlabel, h5_dir):
    import pandas as pd
    import h5py
    import numpy as np
    if str(h5_dir)[-1] != '/':
        h5_dir = str(h5_dir) + '/'
    h5 = h5py.File(h5_dir+'h5_nature_imp_matched.h5')['MATCHED']
    gendata = pd.DataFrame(np.transpose(h5['block0_values'])).T
    col_markers = endo_mat.ix[:0,].T
    feat_markers = features.ix[:0,].T
    nums = col_markers.iloc[:,0]
    fnums = feat_markers.iloc[:,0]
    col_feat_markers = col_markers[nums.isin(fnums)].T
    features.columns = col_feat_markers.columns # this is the endo_mat with reduced number of columns
    #gendata = gendata.drop('par_id', 1)
    apoedata[3][0].index = [x for x in range(len(apoedata[3][0].iloc[:,1]))]
    gendata[['e3','e4']] = apoedata[3][0][['e3','e4']]
    gendata = gendata.mean() - gendata
    classlabel.index = [x for x in range(len(classlabel.iloc[:,1]))]
    endo_mat.index = [x for x in range(len(endo_mat.iloc[:,1]))]
    endo_merge = pd.concat([endo_mat, classlabel.iloc[:,4]], axis=1)
    endo_mean = endo_merge[endo_merge.iloc[:,-1] == 0].mean()
    return endo_mean  
    
NormandJoin = pe.Node(name='NormandJoin',
		      interface=Function(input_names=['endo_mat', 'features',
		      				      'covdata', 'apoedata', 
		      				      'classlabel','h5_dir'],
		      			output_names=[''],
		      			function=norm_and_join))
                                            
def save_files(cov, endo, n_imp, output_dir):
    files = {'cov': cov, 'endo': endo, 'nature_imp': n_imp}
    import pandas as pd
    output_dir = str(output_dir)
    if output_dir[-1] != '/':
	for k,v in files.items():
            v.to_csv(output_dir+'/'+'%s.csv' % k)
    else:
	for k,v in files.items():
            v.to_csv(output_dir+'%s.csv' % k)

SaveFiles = pe.Node(name='SaveFiles',
                    interface=Function(input_names=['cov','endo','n_imp','output_dir'],
                                       output_names=[''],
				       overwrite=True,
                                       function=save_files))
 
if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='genus.py',
                                    description=__doc__)
    parser.add_argument('-m', '--method', type=int, default=1,
                       help='the method for feature selection \n 1 \
                       for  univariate test \n 2 for l1-moded based regression \n \
                       3 for tree ensemble \n 4 for rfe with cross validation')
    parser.add_argument('-d', '--data_dir', type=str, help='directory where the data files live' )
    parser.add_argument('-o', '--output_dir', type=str, help='directory to output files')
    args=parser.parse_args()
    method = args.method
    data_dir = args.data_dir
    output_dir = args.output_dir

FeatureSelection.inputs.method = method
SetDirectory.inputs.d_dir = data_dir
SaveFiles.inputs.output_dir = output_dir
GenoTypeData.inputs.output_dir = output_dir
NormandJoin.inputs.h5_dir = output_dir
    
wf = pe.Workflow(name='wf')
wf.base_dir = data_dir


# connections for script functions
wf.connect(SetDirectory, 'd_dir', DataFiles, 'dir_')
wf.connect(DataFiles, 'classlabel', ClassLabel, 'classlabel')
wf.connect(DataFiles, 'files', ReadData, 'files')
wf.connect(ClassLabel, 'diag', ReadData, 'diag')
wf.connect(SetDirectory, 'd_dir', Covs, 'dir_')
wf.connect(ClassLabel, 'diag', Covs, 'classlabel')
wf.connect(ReadData, 'data_frames', EndoMatrix, 'lh_mat')
wf.connect(ReadData, 'data_frames', EndoMatrix, 'rh_mat')
wf.connect(ReadData, 'data_frames', EndoMatrix, 'sub_mat')
wf.connect(BrainThickness, 'headers', EndoMatrix, 'lh')
wf.connect(BrainThickness, 'headers', EndoMatrix, 'rh')
wf.connect(BrainThickness, 'headers', EndoMatrix, 'subc')
wf.connect(ClassLabel, 'diag', EndoMatrix, 'diag')
wf.connect(ClassLabel, 'diag', GenoTypeData, 'classlabel')
wf.connect(DataFiles, 'snp', GenoTypeData, 'fam')
wf.connect(DataFiles, 'snp', GenoTypeData, 'bim')
wf.connect(DataFiles, 'snp', GenoTypeData, 'h5')
wf.connect(EndoMatrix, 'endo_matrix', FeatureSelection, 'endo_mat')
wf.connect(ClassLabel, 'diag', FeatureSelection, 'class_label')
wf.connect(EndoMatrix, 'endo_matrix', NormandJoin, 'endo_mat')
wf.connect(FeatureSelection, 'features', NormandJoin, 'features')
wf.connect(Covs, 'sub_cov_df', NormandJoin, 'covdata')
wf.connect(ReadData, 'data_frames', NormandJoin, 'apoedata')
# connection from gentoypedata to normandrank can't be made. pickle file cant handle size of dataframe
wf.connect(ClassLabel, 'diag', NormandJoin, 'classlabel')


# connections for outputs
wf.connect(EndoMatrix, 'endo_matrix', SaveFiles, 'endo')
wf.connect(Covs, 'sub_cov_df', SaveFiles, 'cov')
wf.connect(GenoTypeData, 'nature_imp_matched', SaveFiles, 'n_imp')

wf.run()
wf.write_graph()
