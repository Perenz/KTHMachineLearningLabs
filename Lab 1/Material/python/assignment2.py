'''
Assignment 2: Explain entropy for a uniform distribution and a
non-uniform distribution, present some example distributions with
high and low entropy.
'''

##

'''
ENTROPY measures the uncertainty of a dataset.
If it's uniformly distributed the entropy is maxisimised because all the points (classes) are picked with the same probability.
So the uncertainty of the dataset can't be higher than that.

Differently, in a non-uniform distribution, the entropy will be lower since some class are represented by more samples than others.
Therefore, since the classes are not picked witb the same probability the uncertainty of the dataset is lower.


Examples:

High entropy -> Rollig a normal dice, tossing a coin, drawing a suit from a deck of cards
    Pi = 1/6, i = {1,2,3,4,5,6}

    E = -6 (1/6 * log (1/6)) = 2,58

Low entropy -> Rolling a fake dice, typing the right pin for a 4 digits code
    Pi = 1/10, i = {1,2,3,4,5}
    P6 = 1/2

    E = -5 (1/10 * log (1/10)) - (1/2 * log(1/2)) = 2,16


Find other examples..-.
'''