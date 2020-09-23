'''
Assignment 0: Each one of the datasets has properties which makes
them hard to learn. Motivate which of the three problems is most
dicult for a decision tree algorithm to learn.
'''

##

'''
Considering the ENTROPY

('Entropy monk1: ', 1.0)
('Entropy monk2: ', 0.957117428264771)
('Entropy monk3: ', 0.9998061328047111)

MONK1 would be the hardest but ENTROPY is a value specialized on the TRAINING dataset so it doesn't catch the GENERAL problem.

The GENERAL problem is described by the boolean concept behind the datasets described in table 1.

MONK1: a3, a4 and a6 are not considered in the assignment of the class. So, starting with a5, the tree will have a depth of just 3.
One node to check the attribute a5, one level to check a1 and another level to check a2.

MONK2: 
This properties represent boolean PARITY which is hard to learn for a decision tree. 
Indeed, all the attributes are "indipendent" so the tree will have a depth of 6.

MONK3: Only a2, a4 and a5 impact the actual class of the sample. So, also this tree has a depth of 3.
However, the additional 5% noise (misclassification) on the training set make the database harder for the decision tree to learn
The noise affect the learning difficult but the model robustness could be increased pruning or using forest of decision tress

CONCLUSION: MONK2 is the most difficult to learn because of the dept it requires to query all the 6 attributes of the dataset.
'''