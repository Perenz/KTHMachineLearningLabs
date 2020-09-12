#Import dataset and functions
import monkdata as m
from dtree import entropy


print("Entropy monk1: ", entropy(m.monk1))
print("Entropy monk2: ", entropy(m.monk2))
print("Entropy monk3: ", entropy(m.monk3))

'''
('Entropy monk1: ', 1.0)
('Entropy monk2: ', 0.957117428264771)
('Entropy monk3: ', 0.9998061328047111)
'''

