#!/bin/bash

PRUN='../../../working_base/zutil_/bin/prun '
NUMEX=20
LIB='LIB'
MODEL=''
RUNS=10000

while getopts ":M:L:B:N:P:h" opt
do
    case $opt in
	h)
	    echo OPTIONS:
	    echo -M : model file '<--' required
	    echo -L : Data length  [10000]
	    echo -B : Library file name [LIB]
	    echo -N : Number of examples in library [20]
	    echo -P : path to prun [../../zutil_/bin/prun]
	    echo '============================'
	    exit 1;;
	L)
	    RUNS=$OPTARG;;
	M)
	    MODEL=$OPTARG;;
	B)
	    LIB=$OPTARG;;
	N)
	    NUMEX=$OPTARG;;
	P)
	    PRUN=$OPTARG;;
	\?)
	    echo "Invalid option: -$OPTARG" >&2
	    exit 1;;
	:)
	    echo "Option -$OPTARG requires an argument." >&2
	    exit 1;;
    esac
done

if [ "$MODEL" == "" ] ; then
    echo "mising model..."
    ./$0 -h
    exit
fi

for i in $(eval echo "{1..$NUMEX}")
do
    $PRUN $MODEL $RUNS | tail -n 1 >> tmpLIB
done
mv tmpLIB $LIB
rm *.dot >& /dev/null


