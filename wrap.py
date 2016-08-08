from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu


if __name__ == '__main__':
    import argparse
    defstr = '(default %(default)s)'
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=str, help="either 'bf', 'normalize', or 'fxvb'")
    parser.add_argument('-o', '--outname', type=str, help='template name for output files')
    parser.add_argument('-i', '--infile', type=str, help='input data file')
    parser.add_argument('-csv', '--csvfile', type=str, help="a BFResults csv file created at the 'bf' step")
    args=parser.parse_args()
    step = args.step
    outname = args.outname
    infile = args.infile
    csvfile = args.csvfile


def runstep(step, infile, outname, csvfile=None, randomSeed=None, colnum=None):
    import nipype.interfaces.matlab as Matlab
    import os

    def outnames(col, outn):
        return outn + '{}.mat'.format(col)

    matlab = Matlab.MatlabCommand()

    matlab.inputs.paths = [
        '/data/petryshen/yoel/dependencies/Carbonetto_VBS/MATLAB',
        '/data/petryshen/yoel/dependencies/gpml-matlab-v3.4-2013-11-11',
        '/data/petryshen/yoel/dependencies/bayesianImagingGenetics/src',
        '/data/petryshen/yoel/dependencies/bayesianImagingGenetics/src/Utils'
    ]

    if step == 'bf':
        matlab.inputs.script = """deployEndoPhenVB('step','%s','inputMat','%s','colNum',%d,'outFile','%s')"""%(step, 
        infile, colnum, os.path.join('/data/petryshen/yoel/analysis/bayesianImagingGenetics/bf/',outnames(colnum, outname)))
    elif step == 'normalize':
        matlab.inputs.script = """deployEndoPhenVB('step','%s','inFile','%s','outFile','%s')"""%(step, 
        os.path.join('/data/petryshen/yoel/analysis/bayesianImagingGenetics/bf/',outnames(colnum, outname)),
        os.path.join('/data/petryshen/yoel/analysis/bayesianImagingGenetics/bf/',outnames(colnum, outname)))
    elif step == 'fxvb':
        matlab.inputs.script = """run('startup.m');
        deployEndoPhenVB('step','%s','csvFile','%s','randomSeed',%d,'inputMat','%s','outFile','%s','numRepeats',%d)"""%(
            step, csvfile, randomSeed, infile, os.path.join('/data/petryshen/yoel/analysis/bayesianImagingGenetics/fxvb/',
                                                    'fxvb_'+outnames(randomSeed, outname)), 20)

    matlab.inputs.mfile=True
    res = matlab.run()
    print matlab.inputs.script
    print res.runtime.stdout
    
Runstep = pe.Node(name='Runstep',
                 interface=Function(input_names=[
            'step','infile','colnum',
            'outname',
            'csvfile', 'randomSeed'],
                output_names=[''],
                        function=runstep))
Runstep.plugin_args = {'bsub_args': '-q big -We 120:00'}
Runstep.inputs.outname = outname
Runstep.inputs.infile = infile
Runstep.inputs.step = step
Runstep.inputs.csvfile = csvfile

if step == 'bf' or step == 'normalize':
    Iternode = pe.Node(niu.IdentityInterface(fields=['colnum']), name = 'Iternode')
    Iternode.iterables =[('colnum', [x for x in range(1, 102)])]
else:
    Iternode = pe.Node(niu.IdentityInterface(fields=['randomSeed']), name = 'Iternode')
    Iternode.iterables = [('randomSeed', [2134, 2111, 2354, 2132, 2564, 4321, 1123, 2001, 2234, 3234])]

def csv(colnum, outname):
    import pandas as pd
    import os
    df = pd.DataFrame(columns=['colNum','matFn'])
    for i in range(1,colnum):
        df.loc[i] = [i, os.path.join('/data/petryshen/yoel/analysis/bayesianImagingGenetics/bf/',
        outname+'{}.mat'.format(i))]
    df.iloc[:,0] = df.iloc[:,0].astype(int)
    df.to_csv('BFResultsFileList.csv', index=False)


if step == 'normalize':
    wf = pe.Workflow(name="wf_norm")
    wf.connect(Iternode, 'colnum', Runstep, 'colnum')
elif step == 'fxvb':
    wf = pe.Workflow(name="wf_fxvb")
    wf.connect(Iternode, 'randomSeed', Runstep, 'randomSeed')
else:
    csv(102, outname)
    wf = pe.Workflow(name="wf_bf")
    wf.connect(Iternode, 'colnum', Runstep, 'colnum')
wf.base_dir = '/data/petryshen/yoel/Analysis/'
wf.run(plugin='LSF', plugin_args={'bsub': '-q big -We 120:00'})
