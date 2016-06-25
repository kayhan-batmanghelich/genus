# seems to work now
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu


def runstep(step, infile, colnum, vbvs, gpml, depvb, comp, outname):
    import nipype.interfaces.matlab as Matlab
    import os
    def checkstr(string):
        if string[-1] == '/':
            return string
        else:
            return string + '/'
    def outnames(col, outn):
        return outn + '{}.mat'.format(col)
    matlab = Matlab.MatlabCommand()
    matlab.inputs.paths = [vbvs, gpml, depvb, comp]
    matlab.inputs.script = """deployEndoPhenVB('step','%s', 'inputMat','%s', 'colNum',%d, 'outfile','%s' );""" % (step, infile, colnum, os.path.join('/om/user/ysa/',outnames(colnum, outname))) 
    res = matlab.run()
    
Runstep = pe.Node(name='Runstep',
                 interface=Function(input_names=[
            'step','infile','outfile','colnum',
            'vbvs', 'gpml', 'depvb', 'comp', 'outname'],
                output_names=[''],
                        function=runstep))


Infosource = pe.Node(niu.IdentityInterface(fields=['colnum']), name = 'Infosource')
Infosource.iterables =[('colnum', [x for x in range(1, 95)])]

def csv(colnum, outname):
    import pandas as pd
    df = pd.DataFrame(columns=['colNum','matFn'])
    for i in range(1,colnum):
        df.loc[i] = [i, outname+'{}.mat'.format(i)]
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

csv(95, outname)

wf = pe.Workflow(name="wf")
wf.base_dir = '/om/user/ysa/testdir/new'
wf.connect(Infosource, 'colnum', Runstep, 'colnum')
wf.run('SLURM', plugin_args={'sbatch_args': '-c2 --mem=8G'})
