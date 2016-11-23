# -*- coding: utf-8 -*-
## ©Shamir Alavi, Nov 20, 2016

import random as r
import numpy as np
import pandas as pd

from deap import base
from deap import creator
from deap import tools

irisTrain = pd.read_csv('D:\\My Programs\\Python\\PhD\\irisdata-training.csv')
irisTest = pd.read_csv('D:\\My Programs\\Python\\PhD\\irisdata-test.csv')

#stats for iris dataset (training):
#	s-len	s-wid	p-len	p-wid
#setosa
#min	4.3	2.3	1	0.1
#avg	4.9875	3.3725	1.465	0.23
#max	5.8	4.1	1.9	0.6
#versicolor
#min	4.9	2	3	1
#avg	5.865	2.7525	4.21	1.3125
#max	7	3.4	5.1	1.8
#virginica
#min	4.9	2.5	4.5	1.4
#avg	6.6	2.9775	5.5625	2.015
#max	7.9	3.8	6.9	2.5

r.seed(46)

# Following lists are hardcoded using iris data stats
slen_min = np.arange(4.3,5.9,0.1).tolist()
slen_max = np.arange(5.9,8.0,0.1).tolist()
swid_min = np.arange(2.0,3.0,0.1).tolist()
swid_max = np.arange(3.0,4.2,0.1).tolist()
plen_min = np.arange(1.0,4.3,0.1).tolist()
plen_max = np.arange(4.3,7.0,0.1).tolist()
pwid_min = np.arange(0.1,1.4,0.1).tolist()
pwid_max = np.arange(1.4,2.6,0.1).tolist()

attributesCombo = [slen_min,slen_max,swid_min,swid_max,plen_min,plen_max,pwid_min,pwid_max]
average = []
irisClasses = []
irisClasses2 = []
bestIndices = []
bestFitnesses = []
generatedLabelsTrain = np.zeros(len(irisTrain)).tolist()
generatedLabelsTest = np.zeros(len(irisTest)).tolist()

def createRule(listOfAllAttr):
    # Returns a list that contains a random sample of the min and max values of every attribute
    # The entire population contains the ranges of min and max values of every attribute
    # input param: a list containing all of the min-max ranges of each attribute (2D list)
    # output param: a random sample containing one min and one max value for each attribute (1D list)
    # Note: The values are rounded up to one decimal place; the boundary between min and max
    #       values for each attribute is the mean value
    attributeSample = []
    for i in xrange(len(listOfAllAttr)):
        attributeSample.append(r.sample(listOfAllAttr[i],1)[0])
        attributeSample[i] = round(attributeSample[i],1)
    return attributeSample

creator.create("maxAccuracy", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.maxAccuracy)

toolbox = base.Toolbox()
toolbox.register("sample", createRule, attributesCombo)   # random sampling
toolbox.register("individual", tools.initIterate, creator.Individual,
                           toolbox.sample)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def removeIllegals(Pop):
    # min value !>= max value
    # hardcoded
    for i in xrange(len(Pop)):
        if ((   Pop[i][0] >= Pop[i][1]) or (Pop[i][2] >= Pop[i][3])
            or Pop[i][4] >= Pop[i][5]) or (Pop[i][6] >= Pop[i][7]):
            del Pop[i]
    return Pop

# Evaluation heuristic/ Fitness function
def evaluateDataset(data, rule):    # data = irisTrain.values
    # returns the accuracy which is the total number of samples classified
    # input parameters: rule is an association rule (list)
    #                   data is the entire training dataset (pandas dataframe)
    # output: accuracy value
    # limitations: no bound check, hardcoded
    accuracy = 0
    irisClass = []
    for i in xrange(len(data)):
        temp = list(data[i])
        if ((rule[0]<temp[0]<rule[1]) and (rule[2]<temp[1]<rule[3]) and (rule[4]<temp[2]<rule[5]) and (rule[6]<temp[3]<rule[7])):
            accuracy += 1
            irisClass.append(temp[4])
    irisClasses.append(set(irisClass))
    return accuracy

def evaluateTenBest(data, rule):    # data = irisTrain.values
    # returns the class labels if matching rule is found
    # input parameters: rule is an association rule (list)
    #                   data is the entire training dataset (pandas dataframe)
    # output: class labels (set)
    # limitations: no bound check, hardcoded
    irisClass = []
    for i in xrange(len(data)):
        temp = list(data[i])
        if ((rule[0]<temp[0]<rule[1]) and (rule[2]<temp[1]<rule[3]) and (rule[4]<temp[2]<rule[5]) and (rule[6]<temp[3]<rule[7])):
            irisClass.append(temp[4])
    classes = set(irisClass)
    return classes
    
# Evaluation heuristic/ Fitness function
def evaluateAccuracy(data, rule, labelContainer):    # data = irisTrain.values, rule = bestOnes
    # return: None
    # input parameters: rule is an association rule (list)
    #                   data is the entire training dataset (pandas dataframe)
    #                   labelContainer is a list that contains the generated labels (list)
    # output: stores generated labels in a predefined list
    # limitations: no bound check, hardcoded
    ignoreIndices = []
    for j in xrange(len(rule)):
        tempRule = rule[j][1]
        for i in xrange(len(data)):
            if i == 0:
                temp = list(data[i])
                if ((tempRule[0]<temp[0]<tempRule[1]) and (tempRule[2]<temp[1]<tempRule[3]) and (tempRule[4]<temp[2]<tempRule[5]) and (tempRule[6]<temp[3]<tempRule[7])):
                    labelContainer[i] = rule[j][2][0]
                    ignoreIndices.append(i)
            else:
                if  (i in ignoreIndices) == True:
                    continue
                else:
                    temp = list(data[i])
                    if ((tempRule[0]<temp[0]<tempRule[1]) and (tempRule[2]<temp[1]<tempRule[3]) and (tempRule[4]<temp[2]<tempRule[5]) and (tempRule[6]<temp[3]<tempRule[7])):
                        labelContainer[i] = rule[j][2][0]
                        ignoreIndices.append(i)

# Create toolbox functions for our GA
toolbox.register("evaluate", evaluateDataset, irisTrain.values)
toolbox.register("mate", tools.cxUniform, indpb = 0.9)    # crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)    # mutation
toolbox.register("select", tools.selTournament, tournsize = 1 )

def main():
    
    # Generate random rules (our starting population)
    Population = toolbox.population(n=200)
    
    # Remove illegal entries
    legalPop = removeIllegals(Population)
    
    CXPB, MUTPB, NGEN = 0.9, 0.2, 50       # probability of crossover and mutation; number of generations to evolve
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, legalPop)
    
    # store indices of unique(single) class labels only
    goodIndices = []
    for i in range(len(irisClasses)):
        if len(irisClasses[i]) == 1:
            goodIndices.append(i) 
    
    # store rules that correspond to the good indices above
    legalPopGood = []
    for i in range(len(goodIndices)):
        legalPopGood.append(legalPop[goodIndices[i]])
    
    
    ## Start evolution
    k = 5         # selection parameter
    for g in range(NGEN):
        if ((g % 100 == 0) or g == (NGEN - 1)):
            print("-- Generation %i --" % g)
                  
        # Select the next generation individuals
        offspring = toolbox.select(legalPopGood, k)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if r.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    
        # Apply mutation on the offspring
        for mutant in offspring:
            if r.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]
            
        #print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        legalPop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in legalPop]
        bestIndices.append(tools.selBest(legalPop,1)[0])
        bestFitnesses.append(max(fits))
        
        length = len(legalPop)
        mean = sum(fits) / length
        
        average.append(mean)
    
    print("-- End of (successful) evolution --")
    
    # Find the ten best rules based on fitness(accuracy)
    tenBestFitIndices = sorted(range(len(bestFitnesses)), key=lambda i: bestFitnesses[i])[-10:]
    tenBestFitIndices.reverse()
    tenBestRules = []
    tenBestFits = []
    for i in range(len(tenBestFitIndices)):
        tenBestRules.append(bestIndices[tenBestFitIndices[i]])
        tenBestFits.append(bestFitnesses[tenBestFitIndices[i]])
    
    # Find which class(es) the best rules correspond to
    for i in range(len(tenBestRules)):
        irisClasses2.append(evaluateTenBest(irisTrain.values, tenBestRules[i]))
        irisClasses2[i] = list(irisClasses2[i])
    
    # Set class non-unique class labels to virginica and without touching the
    # unique ones (which are all versicolor)
    for i in range(len(irisClasses2)):
        if (irisClasses2[i][0] == 'Iris-virginica'):
            irisClasses2[i] = ['Iris-virginica']

    # Combine all in one list
    bestOnes = zip(tenBestFits, tenBestRules, irisClasses2)
    #print "bestOnes:"
    #print bestOnes

    ## Evaluate accuracy on training set
    #allTrainingLabels = list(irisTrain.ix[:,4])
    #evaluateAccuracy(irisTrain.values, bestOnes, generatedLabelsTrain)    # data = irisTrain.values
    #
    ## if label != virginica or versicolor: then label = setosa
    #for i in range(len(generatedLabelsTrain)):
    #    if generatedLabelsTrain[i] == 0.0:
    #        generatedLabelsTrain[i] = 'Iris-setosa'
    #        
    #match = 0
    #for i in range(len(allTrainingLabels)):
    #    if (allTrainingLabels[i] == generatedLabelsTrain[i]) == True:
    #        match += 1
    #trainingAccuracy = (float(match)/len(allTrainingLabels))*100
           
            
    # Evaluate accuracy on test set          
    allTestLabels = list(irisTest.ix[:,4])    
    evaluateAccuracy(irisTest.values, bestOnes, generatedLabelsTest)    # data = irisTest.values
    
    # if label != virginica or versicolor: then label = setosa
    for i in range(len(generatedLabelsTest)):
        if generatedLabelsTest[i] == 0.0:
            generatedLabelsTest[i] = 'Iris-setosa'
   
    match = 0
    for i in range(len(allTestLabels)):
        if (allTestLabels[i] == generatedLabelsTest[i]) == True:
            match += 1
    testAccuracy = (float(match)/len(allTestLabels))*100
    
    print ""
    print "-------  Statistics  -------"
    print "Generation number: ", g+1
    
    #print "Classification accuracy on Training: ", trainingAccuracy, "%"
    print "Best Classification accuracy: ", round(testAccuracy,1), "%"
    print "Best association rules (in no particular order):"
    print "1.  If (", bestOnes[0][1][0], "<s-length<", bestOnes[0][1][1], ") and (", bestOnes[0][1][2], "<s-width<", bestOnes[0][1][3], ") and (", bestOnes[0][1][4], "<p-length<", bestOnes[0][1][5], ") and (", bestOnes[0][1][6], "<p-width<", bestOnes[0][1][7], "), then class = ", bestOnes[0][2][0]
    print "2.  If (", bestOnes[1][1][0], "<s-length<", bestOnes[1][1][1], ") and (", bestOnes[1][1][2], "<s-width<", bestOnes[1][1][3], ") and (", bestOnes[1][1][4], "<p-length<", bestOnes[1][1][5], ") and (", bestOnes[1][1][6], "<p-width<", bestOnes[1][1][7], "), then class = ", bestOnes[1][2][0]
    print "3.  If (", bestOnes[2][1][0], "<s-length<", bestOnes[2][1][1], ") and (", bestOnes[2][1][2], "<s-width<", bestOnes[2][1][3], ") and (", bestOnes[2][1][4], "<p-length<", bestOnes[2][1][5], ") and (", bestOnes[2][1][6], "<p-width<", bestOnes[2][1][7], "), then class = ", bestOnes[2][2][0]
    print "4.  If (", bestOnes[3][1][0], "<s-length<", bestOnes[3][1][1], ") and (", bestOnes[3][1][2], "<s-width<", bestOnes[3][1][3], ") and (", bestOnes[3][1][4], "<p-length<", bestOnes[3][1][5], ") and (", bestOnes[3][1][6], "<p-width<", bestOnes[3][1][7], "), then class = ", bestOnes[3][2][0]
    print "5.  If (", bestOnes[4][1][0], "<s-length<", bestOnes[4][1][1], ") and (", bestOnes[4][1][2], "<s-width<", bestOnes[4][1][3], ") and (", bestOnes[4][1][4], "<p-length<", bestOnes[4][1][5], ") and (", bestOnes[4][1][6], "<p-width<", bestOnes[4][1][7], "), then class = ", bestOnes[4][2][0]
    print "6.  If (", bestOnes[5][1][0], "<s-length<", bestOnes[5][1][1], ") and (", bestOnes[5][1][2], "<s-width<", bestOnes[5][1][3], ") and (", bestOnes[5][1][4], "<p-length<", bestOnes[5][1][5], ") and (", bestOnes[5][1][6], "<p-width<", bestOnes[5][1][7], "), then class = ", bestOnes[5][2][0]
    print "7.  If (", bestOnes[6][1][0], "<s-length<", bestOnes[6][1][1], ") and (", bestOnes[6][1][2], "<s-width<", bestOnes[6][1][3], ") and (", bestOnes[6][1][4], "<p-length<", bestOnes[6][1][5], ") and (", bestOnes[6][1][6], "<p-width<", bestOnes[6][1][7], "), then class = ", bestOnes[6][2][0]
    print "8.  If (", bestOnes[7][1][0], "<s-length<", bestOnes[7][1][1], ") and (", bestOnes[7][1][2], "<s-width<", bestOnes[7][1][3], ") and (", bestOnes[7][1][4], "<p-length<", bestOnes[7][1][5], ") and (", bestOnes[7][1][6], "<p-width<", bestOnes[7][1][7], "), then class = ", bestOnes[7][2][0]
    print "9.  If (", bestOnes[8][1][0], "<s-length<", bestOnes[8][1][1], ") and (", bestOnes[8][1][2], "<s-width<", bestOnes[8][1][3], ") and (", bestOnes[8][1][4], "<p-length<", bestOnes[8][1][5], ") and (", bestOnes[8][1][6], "<p-width<", bestOnes[8][1][7], "), then class = ", bestOnes[8][2][0]
    print "10. If (", bestOnes[9][1][0], "<s-length<", bestOnes[9][1][1], ") and (", bestOnes[9][1][2], "<s-width<", bestOnes[9][1][3], ") and (", bestOnes[9][1][4], "<p-length<", bestOnes[9][1][5], ") and (", bestOnes[9][1][6], "<p-width<", bestOnes[9][1][7], "), then class = ", bestOnes[9][2][0]
    
    #print "Average classification accuracy: ", np.mean(np.asarray(average))
    #print "_____________________________________"
    #print "Parameters:"
    #print "Fitness heuristic: best accuracy"
    ##print "Population size = 5"
    #print "Crossover probability = ", CXPB, ", ", "type: Tournament Selection (Tournament size = 3, k = ", k, ")"
    #print "Mutation probability = ", MUTPB, ", ", "type: Shuffle Indexes (shuffle probability = 0.02)"
    
    #fig, ax1 = plt.subplots()
    #ax1.plot(range(len(average)), average, "b-", label="Average Tour Length")
    #ax1.set_xlabel("Generation")
    #ax1.set_ylabel("Average Tour Length", color="b")
    #plt.show()
    
if __name__ == "__main__":
    main()