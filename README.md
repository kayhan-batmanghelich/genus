# genus


*************************** 
### Steps needed to run wrap.py
***************************

#### Set up python

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    bash Miniconda2-latest-Linux-x86_64.sh
    conda create -p /PHShome/ysa2/genus_env python pip
    cd /PHShome/ysa2/genus_env # just an example
    source activate .
    conda install pip pytables matplotlib scikit-learn seaborn pandas statsmodels cython sympy networkx 
    pip install nibabel nipy nipype 

#### To be able to use and import the matlab engine

    cd "matlabroot/extern/engines/python"
    python setup.py build --build-base="builddir" install --prefix="installdir"

* "installdir" can be the path to a python environment created with anaconda `/PHShome/ysa2/genus_env`



#### Other matlab dependencies

* download both from the links below, make sure to run the install.m file in the varbvs download. this will require a gcc version of 4.7.x

https://github.com/pcarbo/varbvs/tree/c24049a5bc9532dae4f7f2d81df50ca877a47a6d
http://www.gaussianprocess.org/gpml/code/matlab/doc/


