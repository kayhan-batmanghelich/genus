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
    eng.run(checkstr(gpml) + 'startup.m', nargout=0)
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
    df.iloc[:,0] = df.iloc[:,0].astype(int)
    return df

Csv = pd.Node(name='Csv',
             interface=Function(input_names=['colnums','matfiles'],
                               output_names=['df'],
                               function=csv))
 
if __name__ == '__main__':
    import argparse
    defstr = '(default %(default)s)'
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--numrepeats', type=int)
    parser.add_argument('-cn', '--colnums', type=int)
    parser.add_argument('-s', '--step', type=str)
    parser.add_argument('-v', '--vbvs', type=str)
    parser.add_argument('-g', '--gpml', type=str)
    parser.add_argument('-d', '--depvb', type=str)
    parser.add_argument('-c', '--comp', type=str)
    parser.add_argument('-o', '--outpath', type=str)
    args=parser.parse_args()
    numrepeats = args.numrepeats
    colnums = args.colnums
    step = args.step
    vbvs = args.vbvs
    gpml = args.gpml
    depvb = args.depvb
    comp = args.comp
    outpath = args.outpath

Setup.inputs.vbvs = vbvs
Setup.inputs.gpml = gpml
Setup.inputs.depvb = depvb
Setup.inputs.comp = comp
Setup.inputs.outpath = outpath
Csv.inputs.colnums = colnums
