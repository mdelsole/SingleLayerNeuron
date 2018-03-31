# SingleNeuron
A one-neuron machine learning program made from scratch. It is trained to detect patterns in a series of 1's and 0's.

Examples:

[1,0,1] -> [1] 

[0,1,0] -> [0]

[1,1,1] -> [1]

[1,0,0] -> [0]

New Example:

[0,1,1] -> [?]

Using forward and backpropagation, the program successfully determines that there should be a 1 outputted by the new example, 
as the rule it should figure out is that there should be a 1 outputted if and only if there is a 1 in the rightmost column.
