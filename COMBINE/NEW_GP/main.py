
import operator
import math
import random

import numpy
#import numpy as np
import multiprocessing



from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from sklearn.metrics import mean_absolute_error
# Define new functions
def randomProtectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Define new functions
def randomMul(left, right):
    return (left * right)

# Define new functions
def randomAdd(left, right):
    return (left + right)

# Define new functions
def randomSub(left, right):
    return (left - right)

def protectedLog(value):
    try:
        return math.log(value)
    except:
        return math.log(1+abs(value))

def protectedInverted(value):
    try:
        return 1/float(value)
    except:
        return 1

def protectedSin(value):
    return math.sin(value)

def protectedCos(value):
    return math.cos(value)

def mean_2(left, right):
    return (left + right)/2.

def mean_3(x_1, x_2, x_3):
    return (x_1 + x_2 + x_3)/3.

def mean_4(x_1, x_2, x_3, x_4):
    return (x_1 + x_2 + x_3 + x_4)/4.

def min_2(left, right):
    return min((left, right))

def min_3(x_1, x_2, x_3):
    return min((x_1, x_2, x_3))

def min_4(x_1, x_2, x_3, x_4):
    return min((x_1, x_2, x_3, x_4))

def max_2(left, right):
    return max((left, right))

def max_3(x_1, x_2, x_3):
    return max((x_1, x_2, x_3))

def max_4(x_1, x_2, x_3, x_4):
    return max((x_1, x_2, x_3, x_4))


pset = gp.PrimitiveSet("MAIN", 6) ####### TODO
#pset.terminals =
pset.addPrimitive(randomAdd, 2)
pset.addPrimitive(randomSub, 2)
pset.addPrimitive(randomProtectedDiv, 2)
pset.addPrimitive(randomMul, 2)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(mean_2, 2)
pset.addPrimitive(mean_3, 3)
pset.addPrimitive(mean_4, 4)
pset.addPrimitive(max_2, 2)
pset.addPrimitive(max_3, 3)
pset.addPrimitive(max_4, 4)
pset.addPrimitive(min_2, 2)
pset.addPrimitive(min_3, 3)
pset.addPrimitive(min_4, 4)
pset.addPrimitive(protectedInverted, 1)
pset.addPrimitive(protectedSin, 1)
pset.addPrimitive(protectedCos, 1)
#pset.addPrimitive(operator.neg, 1)
#pset.addPrimitive(math.cos, 1)
#pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.addEphemeralConstant("rand102", lambda: random.uniform(-1,1))
pset.renameArguments(ARG0='x_1')
pset.renameArguments(ARG1='x_2')
pset.renameArguments(ARG2='x_3')
pset.renameArguments(ARG3='x_4')
pset.renameArguments(ARG4='x_5')
pset.renameArguments(ARG5='x_6')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, pset=pset, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#################################################
import pandas as pd
import numpy as np

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

def bound_df_retrain(df):
    assert df.shape[0] == 188318
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE

### Train ###
pred_nn_1_retrain = pd.read_csv('../../NN_1/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_nn_1_retrain)
pred_nn_2_retrain = pd.read_csv('../../NN_2/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_nn_2_retrain)
pred_nn_3_retrain = pd.read_csv('../../NN_3/NN_retrain_3.csv', index_col=0)
bound_df_retrain(pred_nn_3_retrain)
pred_nn_4_retrain = pd.read_csv('../../NN_4/NN_retrain_4.csv', index_col=0)
bound_df_retrain(pred_nn_4_retrain)
pred_nn_5_retrain = pd.read_csv('../../NN_5/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_nn_5_retrain)
pred_nn_6_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
bound_df_retrain(pred_nn_6_retrain)



#################################################

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = (abs(func(x_1,
                         x_2,
                         x_3,
                         x_4,
                         x_5,
                         x_6,
                    ) - y) for (x_1,
                                   x_2,
                                   x_3,
                                   x_4,
                                   x_5,
                                   x_6,
                                   y)
                                    in zip(
                                    pred_nn_1_retrain['loss'].values,
                                    pred_nn_2_retrain['loss'].values,
                                    pred_nn_3_retrain['loss'].values,
                                    pred_nn_4_retrain['loss'].values,
                                    pred_nn_5_retrain['loss'].values,
                                    pred_nn_6_retrain['loss'].values,
                                    train['loss'].values,
                                    ))
    return math.fsum(sqerrors)/len(train['loss'].values),


def predictionSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = [abs(func(x_1,
                         x_2,
                         x_3,
                         x_4,
                         x_5,
                         x_6,
                    )) for x_1,
                                   x_2,
                                   x_3,
                                   x_4,
                                   x_5,
                                   x_6,
                                    in zip(
                                    pred_nn_1_retrain['loss'].values,
                                    pred_nn_2_retrain['loss'].values,
                                    pred_nn_3_retrain['loss'].values,
                                    pred_nn_4_retrain['loss'].values,
                                    pred_nn_5_retrain['loss'].values,
                                    pred_nn_6_retrain['loss'].values
                                    )]
    return sqerrors

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=1000))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=1000))

#pool = multiprocessing.Pool()
#toolbox.register("map", pool.map)

def main():
    #random.seed(318)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(5)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 50, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    tmp, tmp_2, tmp_3 = main()
    tmp[0]
