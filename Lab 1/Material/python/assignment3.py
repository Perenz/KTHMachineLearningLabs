#Import dataset and functions
import monkdata as m
from dtree import averageGain

dataset = [m.monk1, m.monk2, m.monk3]

for k in range(3):
    #For each dataset
    maxIG = 0
    maxInd = -1
    for i in range(6):
        #For each attribute
        gain = averageGain(dataset[k], m.attributes[i])
        print("IG(Dataset %d, Attribute %d): %f" % (k+1,i+1, gain))
        if gain > maxIG:
            #Update max index and max
            maxIG = gain
            maxInd = i

    #Print the max
    print("Dataset %d, Attribute %d MAX IG: %f\n\n" % (k+1, maxInd+1, maxIG))


'''
IG(Dataset 1, Attribute 1): 0.07527
IG(Dataset 1, Attribute 2): 0.00584
IG(Dataset 1, Attribute 3): 0.00471
IG(Dataset 1, Attribute 4): 0.02631
IG(Dataset 1, Attribute 5): 0.28703
IG(Dataset 1, Attribute 6): 0.00076
Dataset 1, Attribute 5 MAX IG: 0.287031 -> ROOT NODE = a5


IG(Dataset 2, Attribute 1): 0.00376
IG(Dataset 2, Attribute 2): 0.00246
IG(Dataset 2, Attribute 3): 0.00106
IG(Dataset 2, Attribute 4): 0.01566
IG(Dataset 2, Attribute 5): 0.01728
IG(Dataset 2, Attribute 6): 0.00625
Dataset 2, Attribute 5 MAX IG: 0.017277 -> ROOT NODE = a5


IG(Dataset 3, Attribute 1): 0.00712
IG(Dataset 3, Attribute 2): 0.29374
IG(Dataset 3, Attribute 3): 0.00083
IG(Dataset 3, Attribute 4): 0.00289
IG(Dataset 3, Attribute 5): 0.25591
IG(Dataset 3, Attribute 6): 0.00708
Dataset 3, Attribute 2 MAX IG: 0.293736 -> ROOT NODE = a2

You should split using the attribute with the highest information gain since you'll obtain datasets that are less uncertain
'''