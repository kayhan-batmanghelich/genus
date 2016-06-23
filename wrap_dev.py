import matlab.engine
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio

eng = matlab.engine.start_matlab()


def checkstr(string):
    if string[-1] == '/':
        return string
    else:
        return string + '/'

def setup(vbvs, gpml, depvb, comp, outpath):
    import os
    for i in [vbvs, gpml, depvb, comp]:
        eng.addpath(i)
    eng.run(checkstr(outpath), nargout=0)
    os.chdir(outpath)
    return output
    
Setup = pe.Node(name='Setup',
                interface=Function(input_names=['vbvs','gpml',
                                                'depvb','comp','outpath'],
                                        output_names=['outpath'],
                                        function=setup))

def csv(colnums, matfiles):
    import pandas as pd
    df = pd.DataFrame(columns=['colNum','matFn'])
    for i in range(colnums):
        df.loc[i] = [i, matfiles[i]]
    return df 
