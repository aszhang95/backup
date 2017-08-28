#!/bin/bash

SMALL_EXAMPLE=1
LARGE_EXAMPLE=0


if [ $SMALL_EXAMPLE -eq 1 ] ; then


    #generate libraries
    ./test.sh -L 5300 -N 70 -B LIB0 -M model0.cfg
    ./test.sh -L 5800 -N 60 -B LIB1 -M model1.cfg
    ./test.sh -L 8200 -N 180 -B LIB2 -M model2.cfg

    #generate test data
    ./test.sh -L 5100 -N 15 -B TEST0 -M model0.cfg
    ./test.sh -L 5500 -N 15 -B TEST1 -M model1.cfg
    ./test.sh -L 5000 -N 15 -B TEST2 -M model2.cfg 


fi


if [ $LARGE_EXAMPLE -eq 1 ] ; then


    #generate libraries
    ./test.sh -L 53000 -N 270 -B LIB0 -M model_0.cfg
    ./test.sh -L 58000 -N 180 -B LIB1 -M model_1.cfg
    ./test.sh -L 52000 -N 190 -B LIB2 -M model_2.cfg

    #generate test data
    ./test.sh -L 51000 -N 15 -B TEST0 -M model_0.cfg
    ./test.sh -L 55000 -N 15 -B TEST1 -M model_1.cfg
    ./test.sh -L 50000 -N 15 -B TEST2 -M model_2.cfg 


fi
