'''
currently runs step 'bf' and writes output file
next steps: 
  check normalize step
  create the cvs output
  check fxvb step
  add nipype into script for parrallel processing
'''

import numpy as np
import matlab.engine
import os

eng = matlab.engine.start_matlab()
eng.run('/om/user/ysa/genus/bayes/basis/gpml/startup.m', nargout=0)
eng.addpath('/om/user/ysa/genus/bayes/Carbonetto_VBS/MATLAB')
eng.addpath('/om/user/ysa/genus/bayes/basis/bayesianImagingGenetics/src')
data = '/om/user/ysa/genus/adni/natureImput-ChrAll_Data94.mat'
os.chdir('/om/user/ysa/testdir/')
eng.deployEndoPhenVB('step','bf','inputMat',data,'colNum',2,'outFile','t_test_2_m.m', nargout=0)
