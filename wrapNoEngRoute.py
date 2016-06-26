# seems to work now
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu


def runstep(step, infile, colnum, vbvs, gpml, depvb, comp, outname, csvfile=None):
    import nipype.interfaces.matlab as Matlab
    import os
    def outnames(col, outn):
        return outn + '{}.mat'.format(col)
    matlab = Matlab.MatlabCommand()
    matlab.inputs.paths = [vbvs, gpml, depvb, comp]
    if step == 'bf':
        matlab.inputs.script = """deployEndoPhenVB('step','%s','inputMat','%s','colNum',%d,'outFile','%s')"""%(step, 
        infile, colnum, os.path.join('/om/user/ysa/',outnames(colnum, outname)))
    elif step == 'normalize':
        matlab.inputs.script = """deployEndoPhenVB('step','%s','inFile','%s','outFile','%s')"""%(step, 
        os.path.join('/om/user/ysa/',outnames(colnum, outname)) , os.path.join('/om/user/ysa/',outnames(colnum, outname)))
    elif step == 'fxvb':
        matlab.inputs.scripts = """deployEndoPhenVB('step','%s','csvFile','%s','randomSeed',%d,'inputMat','%s','numRepeats',%d)"""%(
            step, csvfile, 2194, infile, os.path.join('/om/user/ysa/testdir/fxvb/', 'fxvb_'+outnames(colnum, outname)), 20)
    matlab.inputs.mfile=True
    res = matlab.run()
    
Runstep = pe.Node(name='Runstep',
                 interface=Function(input_names=[
            'step','infile','outfile','colnum',
            'vbvs', 'gpml', 'depvb', 'comp', 'outname'
            'csvfile'],
                output_names=[''],
                        function=runstep))

Iternode = pe.Node(niu.IdentityInterface(fields=['colnum']), name = 'Iternode')
Iternode.iterables =[('colnum', [x for x in range(1, 95)])]

def csv(colnum, outname):
    import pandas as pd
    import os
    df = pd.DataFrame(columns=['colNum','matFn'])
    for i in range(1,colnum):
        df.loc[i] = [i, os.path.join(os.getcwd(),outname+'{}.mat'.format(i))]
    df.iloc[:,0] = df.iloc[:,0].astype(int)
    df.to_csv('BFResults.csv', index=False)

if __name__ == '__main__':
    import argparse
    defstr = '(default %(default)s)'
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=str, help="either 'bf', 'normalize', or 'fxvb'")
    parser.add_argument('-v', '--vbvs', type=str, help='path to varbvs directory')
    parser.add_argument('-g', '--gpml', type=str, help='path to gpml directory')
    parser.add_argument('-d', '--depvb', type=str, help='path to where deployEndoPhenVB.m lives')
    parser.add_argument('-c', '--comp', type=str, help='path to where computeGPLnZHelper.m lives')
    parser.add_argument('-o', '--outname', type=str, help='template name for output files')
    parser.add_argument('-i', '--infile', type=str, help='input data file')
    parser.add_argument('-csv', '--csvfile', type=str, help="a BFResults csv file created at the 'bf' step")
    args=parser.parse_args()
    step = args.step
    vbvs = args.vbvs
    gpml = args.gpml
    depvb = args.depvb
    comp = args.comp
    outname = args.outname
    infile = args.infile
    csvfile = args.csvfile

Runstep.inputs.vbvs = vbvs
Runstep.inputs.gpml = gpml
Runstep.inputs.depvb = depvb
Runstep.inputs.comp = comp
Runstep.inputs.outname = outname
Runstep.inputs.infile = infile
Runstep.inputs.step = step
Runstep.inputs.csvfile = csvfile


if step == 'normalize':
    wf = pe.Workflow(name="wf_norm")
elif step == 'fxvb':
    wf = pe.Workflow(name="wf_fxvb")
else:
    csv(95, outname)
    wf = pe.Workflow(name="wf_bf")
wf.base_dir = '/om/user/ysa/testdir/new'
wf.connect(Iternode, 'colnum', Runstep, 'colnum')
wf.run('SLURM', plugin_args={'sbatch_args': '-c2 --mem=8G'})
