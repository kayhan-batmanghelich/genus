import matlab.engine
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio

eng = matlab.engine.start_matlab()

def setup(vbvs, gpml, depvb, outpath):
    import os
    for i in [vbvs, gpml, depvb]:
        eng.addpath(i)
    if gpml[-1] == '/':
        eng.run(gpml+'startup.m', nargout=0)
    else:
        eng.run(gpml+'/startup.m', nargout=0)
    os.chdir(outpath)

def csv(colnums, matfiles):
    import pandas as pd
    df = pd.DataFrame()
    df.columns = ['colnum', 'matFn']
    for i in range(len(colnums)):
        df.ix[i] = [i, matfiles[i]]
    df.to_csv('BFResultsFileList.csv')
    
def runsteps(step, infile, outfile, data, colnum):
    # next
