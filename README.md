# genus


*************************** 
### Steps needed to run wrap.py
***************************

#### to be able to use and import the matlab engine

    cd "matlabroot\extern\engines\python"

    python setup.py build --build-base="builddir" install --prefix="installdir"

* "installdir" can be the path to python environment created with anaconda



#### other matlab dependencies

* download both from the links below, make sure to run the install.m file in the varbvs download. this will require a gcc version of 4.7.x

https://github.com/pcarbo/varbvs/tree/c24049a5bc9532dae4f7f2d81df50ca877a47a6d
http://www.gaussianprocess.org/gpml/code/matlab/doc/

#### python dependencies to be added soon
