#Import dataset and functions
import monkdata as m
from dtree import buildTree, check, misClasRate


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


dataset = [m.monk1, m.monk2, m.monk3]
test_dataset = [m.monk1test, m.monk2test, m.monk3test]

for k in range(3):
    #For each dataset
    dec_tree = buildTree(dataset[k], m.attributes)

    #Evaluate performace

    #On test dataset
    test_perf = misClasRate(dec_tree, test_dataset[k])

    #On trainingt dataset
    train_perf = misClasRate(dec_tree, dataset[k])

    print("Dataset %d\n\tTest perf: %f\n\tTrain perf: %f" % (k+1, test_perf, train_perf))


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
        Test perf: 0.171296
        Train perf: 0.000000
Dataset 2
        Test perf: 0.307870
        Train perf: 0.000000
Dataset 3
        Test perf: 0.055556
        Train perf: 0.000000
'''
