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
IG(Dataset 1, Attribute 1): 0.075273
IG(Dataset 1, Attribute 2): 0.005838
IG(Dataset 1, Attribute 3): 0.004708
IG(Dataset 1, Attribute 4): 0.026312
IG(Dataset 1, Attribute 5): 0.287031
IG(Dataset 1, Attribute 6): 0.000758
Dataset 1, Attribute 5 MAX IG: 0.287031


IG(Dataset 2, Attribute 1): 0.003756
IG(Dataset 2, Attribute 2): 0.002458
IG(Dataset 2, Attribute 3): 0.001056
IG(Dataset 2, Attribute 4): 0.015664
IG(Dataset 2, Attribute 5): 0.017277
IG(Dataset 2, Attribute 6): 0.006248
Dataset 2, Attribute 5 MAX IG: 0.017277


IG(Dataset 3, Attribute 1): 0.007121
IG(Dataset 3, Attribute 2): 0.293736
IG(Dataset 3, Attribute 3): 0.000831
IG(Dataset 3, Attribute 4): 0.002892
IG(Dataset 3, Attribute 5): 0.255912
IG(Dataset 3, Attribute 6): 0.007077
Dataset 3, Attribute 2 MAX IG: 0.293736
'''