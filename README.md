# genus


*************************** 
### Steps needed to run wrap.py
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

#### Matlab dependencies

* download both from the links below, make sure to run the install.m file in the varbvs download. this will require a gcc version of 4.7.x

https://github.com/pcarbo/varbvs/tree/c24049a5bc9532dae4f7f2d81df50ca877a47a6d
http://www.gaussianprocess.org/gpml/code/matlab/doc/
