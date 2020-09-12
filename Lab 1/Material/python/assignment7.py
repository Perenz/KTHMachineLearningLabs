#Import dataset and functions
import monkdata as m
from dtree import buildTree, check, misClasRate, allPruned
import random
from matplotlib import pyplot as plt
import numpy as np


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

'''
Assignment 7: Evaluate the effect pruning has on the test error for
the monk1 and monk3 datasets, in particular determine the optimal
partition into training and pruning by optimizing the parameter
fraction. Plot the classication error on the test sets as a function
of the parameter fraction 2 f0:3; 0:4; 0:5; 0:6; 0:7; 0:8g.
Note that the split of the data is random. We therefore need to
compute the statistics over several runs of the split to be able to draw
any conclusions. Reasonable statistics includes mean and a measure
of the spread. Do remember to print axes labels, legends and data
points as you will not pass without them.
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


def best_partition(full_dataset):
    #Set local variables
    tmp_max_perf = 0
    max_partition = None
    plot_y = []


    for i in range(6): #[0,1,2,3,4,5]
        i = (float(i)+3)/10 #[0.3,0.4,0.5,0.6,0.7,0.8]

        monk_train, monk_val = partition(full_dataset, i)

        #Get the best pruning for that partition
        max_prune, pruned_tree = prune(buildTree(monk_train, m.attributes), monk_val)

        #Compute performance for pruned_tree on the test set
        max_prune = check(pruned_tree, test_set[k])

        #print("\t NEW(%f), OLD(%f)" % (max_prune, tmp_max_perf))

        #Store the results in a list
        plot_y.append(1-max_prune)

        #Compare perf with the best one
        if max_prune > tmp_max_perf:
            tmp_max_perf = max_prune
            max_partition = i

    return max_partition, tmp_max_perf, plot_y

#For dataset 1
'''
monk1_train, monk1_val = partition(m.monk1, 0.6)
'''

#Call the pruning for dataset 1
'''
maxPerf = prune(buildTree(monk1_train, m.attributes), monk1_val)
print(maxPerf)
'''

#Do it for different values of fraction = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
'''
#Set local variables
tmp_max_perf = 0
max_partition = None
for i in range(6):
    i = (float(i)+3)/10

    monk_train, monk_val = partition(m.monk1, i)

    #Get the best pruning for that partition
    max_prune, pruned_tree = prune(buildTree(monk_train, m.attributes), monk_val)

    #Compute performance for pruned_tree on the test set
    max_prune = check(pruned_tree, m.monk1test)
    print("\t NEW(%f), OLD(%f)" % (max_prune, tmp_max_perf))

    #Plot the classification error as a function for each value of fraction


    #Compare perf with the best one
    if max_prune > tmp_max_perf:
        tmp_max_perf = max_prune
        max_partition = i

print("Best partition %f with performace %f" % (i, tmp_max_perf))
'''

dataset = [m.monk1, m.monk2, m.monk3]
test_set = [m.monk1test, m.monk2test, m.monk3test]

dataset_limit = (0,2)

#Do it for MONK1 and MONK2
total_plot_y = {k:[] for k in dataset_limit}
total_plot_x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

avg_plot_y = {}
var_plot_y = {}

max_perf_dataset = {}
max_ind_dataset = {}

n_times=1000
for k in dataset_limit:
    #Do it for many times (n_times)
    test_error=list()
    for n in range(n_times):
        #Do it for different values of fraction = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
        max_partition, max_perf, plot_y = best_partition(dataset[k])
        
        #Add the results to the total plot values
        test_error.append(plot_y)
        total_plot_y[k].append(plot_y)

    #Compute average for total_plot_y[0] and [1] and a measure of spread
    avg_plot_y[k] = [np.mean(fract_err) for fract_err in zip(*total_plot_y[k])]
    var_plot_y[k] = [np.var(fract_err) for fract_err in zip(*total_plot_y[k])]
    #avg_plot_y[k] = ([sum(x)/n_times for x in zip(*total_plot_y[k])])

    #Now i can get the best partition
    max_perf_part = 1
    max_ind_part = -1

    for i in range(len(total_plot_x)):
        if avg_plot_y[k][i] < max_perf_part:
            #Update max and index
            max_perf_part = avg_plot_y[k][i]
            max_ind_part = i
    

    max_perf_dataset[k] = max_perf_part
    max_ind_dataset[k] = max_ind_part

    #Print the reesults
    print("%f best fraction for MONK%d, error: %f, variance: %f" % (total_plot_x[max_ind_dataset[k]], k+1, max_perf_dataset[k], var_plot_y[k][max_ind_dataset[k]]))
    print("All fract mean error for MONK%d: " % (k+1), avg_plot_y[k])
    print("All fract variance for MONK%d: " % (k+1), var_plot_y[k])



#Plot the classification error as a function for each value of fraction

#Using MatPlotLib
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('ERROR Mean and Variance plots')


#plt.plot(range(10), linestyle='--', marker='o', color='b')


#color_shape = ("r--", "bs", "g^")
colors = ['r', 'b', 'g']
plots = []
for k in dataset_limit:
    #plots.append(plt.plot(total_plot_x, avg_plot_y[k], label="MONK%d"%(k))[0])
    ax1.plot(total_plot_x, avg_plot_y[k], label="MONK%d"%(k+1), linestyle='--', marker='o', color=colors[k])
    ax2.plot(total_plot_x, var_plot_y[k], label="MONK%d"%(k+1), linestyle='--', marker='o', color=colors[k])

#print(plots)
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

plt.xlabel("Fraction")
plt.ylabel("Misclassification Err")

ax1.set(xlabel='Fraction', ylabel='MisC Error mean')
ax2.set(xlabel='Fraction', ylabel='MisC Error variance')

plt.show()

#Run it several times and get the mean


'''
Results fluctuate a LOT

Have to run the statistic multiple times and get mean and a measure of spread

Results:

09/09, 14:25, N=1000: 
    MONK1 best 0.6 -> ERR = 0.194981
    MONK3 best 0.6 -> ERR = 0.034021

09/09, 14:27, N=1000: 
    MONK1 best 0.6 -> ERR = 0.195836
    MONK3 best 0.7 -> ERR = 0.032322

09/09, 14:36, N=1000: 
    MONK1 best 0.6 -> ERR = 0.196738
    MONK3 best 0.7 -> ERR = 0.032787

10/09 14:53, N=1000:
    MONK1 best 0.6 -> ERR = 0.196995
    MONK3 best 0.7 -> ERR = 0.032185

10/09 15:32, N=1000:
    MONK1 best 0.6 -> ERR = 0.194208
    MONK3 best 0.7 -> ERR = 0.031597

10/09 15:34, N=1000:
    MONK1 best 0.6 -> ERR = 0.196014
    MONK3 best 0.7 -> ERR = 0.032722

11/09 14:57, N=1000:
    MONK1 best 0.6 -> ERR = 0.196190
    MONK3 best 0.7 -> ERR = 0.032542
'''

    