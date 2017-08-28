# Example of data smashing based classification

Three models are included to generate data:

+ model_0.cfg 
+ model_1.cfg
+ model_2.cfg

*model_0.cfg* actually is a model of medical history for female children who do not end up with any neuropsychiatric diagnosis. *model_1.cfg* is a model for female kids who do end up with such a diagnosis eventually, and *model_2.cfg* is for male kids who end up with a diagnosis.

## Generate data
	./gendata.sh

This generates the library files LIB0 LIB1 LIB2, and the test files TEST0 TEST1 TEST2 

## Classification test

	./classification_test.sh



