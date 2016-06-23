'''
currently runs steps 'bf' and 'normalize

next steps: 
  check fxvb step
  add nipype into script for parallel processing
'''

import matlab.engine
import os

eng = matlab.engine.start_matlab()
eng.run('/om/user/ysa/genus/bayes/basis/gpml/startup.m', nargout=0)
eng.addpath('/om/user/ysa/genus/bayes/Carbonetto_VBS/MATLAB')
eng.addpath('/om/user/ysa/genus/bayes/basis/bayesianImagingGenetics/src')
data = '/om/user/ysa/genus/adni/natureImput-ChrAll_Data94.mat'
os.chdir('/om/user/ysa/testdir/')
eng.deployEndoPhenVB('step','bf','inputMat',data,'colNum',2,'outFile','test_2.m', nargout=0)
