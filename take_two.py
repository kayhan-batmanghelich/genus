# runs smoothly, and quicker than other methods i've tried but does not write to file
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

def runstep(step, infile, colnum, vbvs, gpml, depvb, comp, outname):
    import os
    import matlab.engine
    eng = matlab.engine.start_matlab()
    def checkstr(string):
        if string[-1] == '/':
            return string
        else:
            return string + '/'
    def outnames(col, outn):
        return outn + '{}.mat'.format(col)
    for i in [vbvs, gpml, depvb, comp]:
        eng.addpath(i)
    eng.run(checkstr(gpml) + 'startup.m', nargout=0)
    eng.deployEndoPhenVB('step', step,
                        'inputMat', infile,
                        'colNum', colnum,
                        'outFile', os.path.join('/om/user/ysa/testdir/new/output',outnames(colnum, outname)),
                        nargout=0)

Runstep = pe.Node(name='Runstep',
                 interface=Function(input_names=[
            'step','infile','outfile','colnum',
            'vbvs', 'gpml', 'depvb', 'comp', 'outname'],
                output_names=[''],
                        function=runstep_bf))


Infosource = pe.Node(niu.IdentityInterface(fields=['colnum']), name = 'Infosource')
Infosource.iterables =[('colnum', [x for x in range(1, 95)])]

def csv(colnums, matfiles):
    import pandas as pd
    df = pd.DataFrame(columns=['colNum','matFn'])
    for i in range(colnums):
        df.loc[i] = [i, matfiles[i]]
    df.iloc[:,0] = df.iloc[:,0].astype(int)
    return df

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
    args=parser.parse_args()
    step = args.step
    vbvs = args.vbvs
    gpml = args.gpml
    depvb = args.depvb
    comp = args.comp
    outname = args.outname
    infile = args.infile

Runstep.inputs.vbvs = vbvs
Runstep.inputs.gpml = gpml
Runstep.inputs.depvb = depvb
Runstep.inputs.comp = comp
Runstep.inputs.outname = outname
Runstep.inputs.infile = infile
Runstep.inputs.step = step

wf = pe.Workflow(name="wf")
wf.base_dir = '/om/user/ysa/testdir/new'
wf.connect(Infosource, 'colnum', Runstep, 'colnum')
wf.run('SLURM', plugin_args={'sbatch_args': '-c2 --mem=8G'})
