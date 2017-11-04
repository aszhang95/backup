module load gsl
module load boost/1.63.0+gcc-6.2
module python

cd zbase
make -f Makefile

cd ..
make -f Makefile
