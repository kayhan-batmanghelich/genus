# genus


*************************** 
### Steps needed to run wrap[something].py
***************************

#### Set up python

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    bash Miniconda2-latest-Linux-x86_64.sh
    conda create -p /PHShome/ysa2/genus_env python pip # this is just an example path
    cd /PHShome/ysa2/genus_env # just an example
    source activate .
    conda install pip pytables matplotlib scikit-learn seaborn pandas statsmodels cython sympy networkx 
    pip install nibabel nipy nipype 

#### Optional

    vi ~/.bashrc
    
    # add the line below
    
    alias genv="source activate /PHShome/ysa2/genus_env"
    
    
* save and exit. `source ~/.bashrc` and now `genv`


#### To be able to use and import the matlab engine

    cd "matlabroot/extern/engines/python"
    python setup.py build --build-base="builddir" install --prefix="installdir"

* "installdir" can be the path to a python environment created with anaconda `/PHShome/ysa2/genus_env`
* refer to the documentation http://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html



#### Other matlab dependencies

* download both from the links below, make sure to run the install.m file in the varbvs download. this will require a gcc version of 4.7.x

https://github.com/pcarbo/varbvs/tree/c24049a5bc9532dae4f7f2d81df50ca877a47a6d
http://www.gaussianprocess.org/gpml/code/matlab/doc/


#### Example use

    srun --mem=8G python wrap.py -v /om/user/ysa/genus/bayes/Carbonetto_VBS/MATLAB -g /om/user/ysa/genus/bayes/basis/gpml -d /om/user/ysa/genus/bayes/basis/bayesianImagingGenetics/src -c /om/user/ysa/genus/bayes/basis/bayesianImagingGenetics/src/Utils -s 'bf' -i /om/user/ysa/genus/adni/natureImput-ChrAll_Data94.mat -o 'natureimpdata94'

