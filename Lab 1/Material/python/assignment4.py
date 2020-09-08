'''
Assignment 4: For splitting we choose the attribute that maximizes
the information gain, Eq.3. Looking at Eq.3 how does the entropy of
the subsets, Sk, look like when the information gain is maximized?
How can we motivate using the information gain as a heuristic for
picking an attribute for splitting? Think about reduction in entropy
after the split and what the entropy implies.
'''

####

'''
The entropy of the subsets Sk look minimized when the IG is maximized.
Because we want to split on the attribute that reduce as much as possible the uncertainty of the dataset.
That uncertainty is described by the Entropy.

So, using the IG as a heuristic for picking an attribute for splitting means that we can
choose for the attribute which split reduces as much as possible the entropy.
In other words, it let us choosing the attribute that minimizes the uncertainty of the remaining datasets
'''