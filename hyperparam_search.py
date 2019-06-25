#The goal of this script is to optimize hyperparameters
#Part 1: Use some random hyperparam values and see which ones perform best
#Part 2: Use neat evolve on the hyperparams to optimize them (later)

#Part 1
#Generate X different config files with random hyperparams

#Run each config file on 16 threads for N iterations
#Alternatively, run until the max_fitness value exceeds some value
#For example, if you know it is "hard" to get 1600, then go until the model
#gets a score of 1600. Then see which model did it the fastest.

#Total the best (or average) fitness scores for the 16 runs

#See which of the X models did best
#Should probably create some fancy graphs
