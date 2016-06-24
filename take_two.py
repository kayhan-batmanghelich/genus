
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

def runstep_bf(step, infile, colnum, vbvs, gpml, depvb, comp, outname):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    def outnames(col, outn):
        return outn + '{}.mat'.format(col)
    for i in [vbvs, gpml, depvb, comp]:
        eng.addpath(i)
    eng.run(gpml + '/startup.m', nargout=0)
    eng.deployEndoPhenVB('step', step,
                        'inputMat', infile,
                        'colNum', colnum,
                        'outFile', outnames(colnum, outname),
                        nargout=0)

RunstepBF = pe.Node(name='Runstep',
                 interface=Function(input_names=[
            'step','infile','outfile','colnum',
            'vbvs', 'gpml', 'depvb', 'comp', 'outname'],
                output_names=[''],
                        function=runstep_bf))


Infosource = pe.Node(niu.IdentityInterface(fields=['colnum']), name = 'Infosource')
Infosource.iterables =[('colnum', [x for x in range(1, 95)])]



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
    parser.add_argument('-i', '--infile', type=str, help='data file')
    args=parser.parse_args()
    step = args.step
    vbvs = args.vbvs
    gpml = args.gpml
    depvb = args.depvb
    comp = args.comp
    outfile = args.outname
    infile = args.infile

RunstepBF.inputs.vbvs = vbvs
RunstepBF.inputs.gpml = gpml
RunstepBF.inputs.depvb = depvb
RunstepBF.inputs.comp = comp
RunstepBF.inputs.outname = outname
RunstepBF.inputs.infile = infile
RunstepBF.inputs.step = step

wf = pe.Workflow(name="wf")
wf.base_dir = '/om/user/ysa/testdir/new'
wf.connect(Infosource, 'colnum', RunstepBF, 'colnum')
wf.run('SLURM', plugin_args={'sbatch_args': '-c2 --mem=8G'})
