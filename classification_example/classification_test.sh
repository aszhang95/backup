#!/bin/bash

#../bin/smashmatch  -f TEST0 -F LIB0 LIB1 LIB2 -T symbolic -D row -L true true true  -o resx -n 2


for i in {0..2}
do 
    ../bin/smashmatch -f TEST"$i" -F LIB0 LIB1 LIB2  -n 10 -T symbolic -D row -L true true true -o res"$i" -d true -v on
done
