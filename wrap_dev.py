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
    out = checkstr(outpath) + 'bfout.mat'
    return out
    
Setup = pe.Node(name='Setup',
                interface=Function(input_names=['vbvs','gpml',
                                                'depvb','comp','outpath'],
                                        output_names=['out'],
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
                               
def runstep_bf(step, infile, outfile, data, colnum):
    eng.deployEndoPhenVB('step', step,
                        'inputMat', infile,
                        'colNum', colnum,
                        'outFile', outfile,
                        nargout = 0)

RunstepBF = pe.Node(name='Runstep',
                 interface=Function(input_names=['step',
                'infile','outfile','data','colnum'],
                output_names=[''],
                        function=runstep))  
                        
#RunstepBF.iterables = ("colnum", [x for x in range(94)])  

if __name__ == '__main__':
    import argparse
    defstr = '(default %(default)s)'
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--numrepeats', type=int, help='')
    parser.add_argument('-cn', '--colnums', type=int, help='the number of features in your data matrix')
    parser.add_argument('-s', '--step', type=str, help="either 'bf', 'normalize', or 'fxvb'")
    parser.add_argument('-v', '--vbvs', type=str, help='path to varbvs directory')
    parser.add_argument('-g', '--gpml', type=str, help='path to gpml directory')
    parser.add_argument('-d', '--depvb', type=str, help='path to where deployEndoPhenVB.m lives')
    parser.add_argument('-c', '--comp', type=str, help='path to where computeGPLnZHelper.m lives')
    parser.add_argument('-o', '--outpath', type=str, help='path for the output files')
    parser.add_argument('-i', '--infile', type=str, help='the input file')
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
RunstepBF.inputs.step = step

wf = pe.Workflow(name='wf')
wf.base_dir = outpath
wf.connect(Setup,'out', RunstepBF,'outfile')
wf.run()
wf.write_graph()
