#Import dataset and functions
import monkdata as m
from dtree import buildTree, check, misClasRate, allPruned
from drawtree_qt5 import drawTree
import random

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

'''
Assignment 5: Build the full decision trees for all three Monk
datasets using buildTree. Then, use the function check to mea-
sure the performance of the decision tree on both the training and
test datasets.
For example to built a tree for monk1 and compute the performance
on the test data you could use
import monkdata as m
import dtree as d
t=d.buildTree(m.monk1, m.attributes);
print(d.check(t, m.monk1test))
Compute the train and test set errors for the three Monk datasets
for the full trees. Were your assumptions about the datasets correct?
Explain the results you get for the training and test datasets.
'''

def prune(dec_tree, val_data):
    #Flag to keep memory of any best tree
    one_better = True

    while one_better:
        #Obtain all the pruned tress
        pruned_trees = allPruned(dec_tree)
        #print("%d pruned tress" % (len(pruned_trees)))
        dec_tree_perf = check(dec_tree, val_data)

        #Set local variables
        one_better=False
        maxPerf = dec_tree_perf

        #Compute performance evaluation and keep the best one
        for tree in pruned_trees:
            tree_perf = check(tree, val_data)
            #print("\t NEW(%f), OLD(%f)" % (tree_perf, maxPerf))
            if tree_perf >= maxPerf:
                maxPerf = tree_perf
                dec_tree= tree
                one_better = True
                #print("\tFound a better one: %f" % (tree_perf))

    return maxPerf, dec_tree


dataset = [m.monk1, m.monk2, m.monk3]
test_dataset = [m.monk1test, m.monk2test, m.monk3test]

for k in range(3):
    #For each dataset

    train_data, val_data = partition(dataset[k], 0.7)

    dec_tree = buildTree(train_data, m.attributes)
    #dec_tree = buildTree(dataset[k], m.attributes)



    #Try pruning
    perf, dec_tree = prune(dec_tree, val_data)

    #Evaluate performace

    #On test dataset
    test_perf = misClasRate(dec_tree, train_data)

    #On trainingt dataset
    train_perf = misClasRate(dec_tree, dataset[k])

    print("Dataset %d\n\tTest misc rate: %f\n\tTrain misc rate: %f" % (k+1, test_perf, train_perf))

    #Plot the Tree
    if k == 2:
        drawTree(dec_tree)


'''
Accuracy

Dataset 1
        Test perf: 0.828704
        Train perf: 1.000000
Dataset 2
        Test perf: 0.692130
        Train perf: 1.000000
Dataset 3
        Test perf: 0.944444
        Train perf: 1.000000
'''


'''
Misclassification rate

Dataset 1
        Test err: 0.171296
        Train err: 0.000000
Dataset 2
        Test err: 0.307870
        Train err: 0.000000
Dataset 3
        Test err: 0.055556
        Train err: 0.000000



PRUNING with f=0.7
Dataset 1
        Test misc rate: 0.083333
        Train misc rate: 0.040323
Dataset 2
        Test misc rate: 0.333333
        Train misc rate: 0.301775
Dataset 3
        Test misc rate: 0.027778
        Train misc rate: 0.065574


So dataset2 is the hardest one to learn.
'''
