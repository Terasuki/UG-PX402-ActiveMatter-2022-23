"""
Week 4

Ray Hu & Xietao
"""

import numpy as np
import matplotlib.pyplot as plt
import random

"""
Parameters.
numberOfRods: the number of rods + motors. done
length: the length of the rods, enter a matrix of size equal to the number of rods. done
seed: seed used to generate motor position. done
seqUpdate: 0 is for top to bottom, 1 for bottom to top, 2 is (uniformly) self-avoiding random, 3 for (uniformly) completely random.
diffSampling: 0 is constant, 1 for uniformly distribution, 2 Gaussian.
persSample: 0 is constant, 1 for uniformly distribution, 2 Gaussian.
finalTime: number of timesteps to be considered. done
recordTime: number of timesteps before each recording. done
"""

numberOfRods = 200
length = np.ones(numberOfRods)
seed = 1
seqUpdate = 1
diffSampling = 0
persSample = 2
finalTime = 400
recordTime = 2

motors = np.zeros((numberOfRods, 2))
rods = np.zeros((numberOfRods, 2))
np.random.seed(seed)

def RodsArray(Rposition, Mposition, length, numberOfRods):

    for i in range(1, numberOfRods):
        Rposition[i, 0] += Rposition[i-1, 0] + Mposition[i-1, 0] - Mposition[i-1, 1]

    Rposition[:, 1] = length #Array
    
    return Rposition

def randwalk2(numberofsteps, Rposition, Mposition, numberOfRods, pervelo, diffvelo):
    # randomwalk with velocity v_d and directional movement with v_p 
    v_p = pervelo
    v_d = diffvelo
    for k in range(1, numberofsteps):     
        for i in range(0, -numberOfRods, -1): #for each rod
            dice1 = np.random.uniform(0,1)
            if dice1 < 0.5 :#random walk to the right
                Mposition[i, 0] += v_p + v_d
                if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1] or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d) or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d)):
                    Mposition[i, 0] -= v_p + v_d
            elif dice1 > 0.5 :#random walk to the left
                Mposition[i, 0] += v_p - v_d
                if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1] or Mposition[i, 0] - abs(v_p - v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p - v_d) or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d)):
                    Mposition[i, 0] -= v_p - v_d
            else:
                pass
            
            dice2 = np.random.uniform(0,1)
            if dice2 < 0.5 :#random walk to the right
                Mposition[i, 1] += -v_p + v_d
                if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1] or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d) or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d)):
                    Mposition[i, 1] -= -v_p + v_d
            elif dice2 > 0.5 :#random walk to the left
                Mposition[i, 1] += -v_p - v_d
                if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1] or Mposition[i + 1, 0] - abs(-v_p - v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p - v_d) or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d)):
                    Mposition[i, 1] -= -v_p - v_d
            else:
                pass
            
        Rposition[1:, 0] = (Mposition[:-1, 0] - Mposition[:-1, 1]).cumsum()
    
    return Rposition, Mposition

def randwalk_shuffle(numberofsteps, Rposition, Mposition, numberOfRods, pervelo, diffvelo):
    # randomwalk with velocity v_d and directional movement with v_p 
    v_p = pervelo
    v_d = diffvelo
    for k in range(1, numberofsteps): 
        Ran = list(range(0, -numberOfRods, -1))
        random.shuffle(Ran)
        for i in Ran: #for each rod
            dice1 = np.random.uniform(0,1)
            if dice1 < 0.5 :#random walk to the right
                Mposition[i, 0] += v_p + v_d
                if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1] or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d) or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d)):
                    Mposition[i, 0] -= v_p + v_d
            elif dice1 > 0.5 :#random walk to the left
                Mposition[i, 0] += v_p - v_d
                if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1] or Mposition[i, 0] - abs(v_p - v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p - v_d) or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d)):
                    Mposition[i, 0] -= v_p - v_d
            else:
                pass
            
            dice2 = np.random.uniform(0,1)
            if dice2 < 0.5 :#random walk to the right
                Mposition[i, 1] += -v_p + v_d
                if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1] or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d) or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d)):
                    Mposition[i, 1] -= -v_p + v_d
            elif dice2 > 0.5 :#random walk to the left
                Mposition[i, 1] += -v_p - v_d
                if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1] or Mposition[i + 1, 0] - abs(-v_p - v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p - v_d) or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d)):
                    Mposition[i, 1] -= -v_p - v_d
            else:
                pass
            
        Rposition[1:, 0] = (Mposition[:-1, 0] - Mposition[:-1, 1]).cumsum()
    
    return Rposition, Mposition


def randwalk_shuffle_v_D(numberofsteps, Rposition, Mposition, numberOfRods, pervelo, diffvelo):
    # randomwalk with velocity v_d and directional movement with v_p 
    v_p = pervelo
    v_d = diffvelo
    for k in range(1, numberofsteps): 
        Ran = list(range(0, -numberOfRods, -1))
        random.shuffle(Ran)
        for i in Ran: #for each rod
            dice1 = np.random.uniform(-1,1)
            Mposition[i, 0] += v_p + dice1 * v_d
            if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1] or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d) or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d)):
                Mposition[i, 0] -= v_p + dice1 * v_d
            else:
                pass
            
            dice2 = np.random.uniform(-1,1)
            Mposition[i, 1] += -v_p + dice2 * v_d
            if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1] or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d) or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d)):
                Mposition[i, 1] -= -v_p + dice2 * v_d
            else:
                pass
        Rposition[1:, 0] = (Mposition[:-1, 0] - Mposition[:-1, 1]).cumsum()

    return Rposition, Mposition

def calculateMotorPositions(RodsMatrix, MotorMatrix, numberOfMotors):

    motorPositions = np.zeros(numberOfMotors)
    position = RodsMatrix[0, 0] + MotorMatrix[0, 0]
    motorPositions[0] = position
    for motor in range(1, numberOfMotors):
        position = (RodsMatrix[motor, 0] + MotorMatrix[motor, 0])
        motorPositions[motor] = position
    
    return motorPositions

def systemLength(RodsMatrix, numberOfMotors):

    rightMostPositions = np.zeros(numberOfMotors)
    for rod in range(0, numberOfMotors):
        rightMostPositions[rod] = RodsMatrix[rod, 0] + RodsMatrix[rod, 1]
    length = -np.amin(RodsMatrix[:, 0]) + np.amax(rightMostPositions)
    return length

def multipleSimulations(numberOfSimulations, length, numberOfRods, finalTime, recordTime, plotEvol):
    
    finalLength = np.zeros(numberOfSimulations)
    seeds = [None]*numberOfSimulations
    for simul, seed in enumerate(np.random.choice(1000000, numberOfSimulations, replace=False)):
        seeds[simul] = np.random.default_rng(seed=seed)
    
    for simul in range(0, numberOfSimulations):
        rods = np.zeros((numberOfRods, 2))
        motors = seeds[simul].random((numberOfRods, 2))
        rods = RodsArray(rods, motors, length, numberOfRods)

        for tstep in range(0, int(finalTime/recordTime)):
            rods, motors = randwalk_shuffle(recordTime, rods, motors, numberOfRods, 0.01, 0.01)
            lengthArray[tstep] = systemLength(rods, numberOfRods)
        if plotEvol:
            plt.scatter(np.linspace(0, lengthArray.size, lengthArray.size), lengthArray)
            plt.title('Rod length evolution')
            plt.xlabel('Time step')
            plt.ylabel('Rod length')
        if simul % 100 == 0:
            print(f'Currently {simul}/{numberOfSimulations}')

        finalLength[simul] = lengthArray[-1]

    return finalLength

def plotMotorPositions(rods, motors, numberOfRods):

    plt.clf()
    motorsPos = calculateMotorPositions(rods, motors, numberOfRods)
    plt.hist(motorsPos)
    plt.xlabel('x coordinate')
    plt.ylabel('number of motors')
    plt.pause(0.1)

def plotSystem(RodsMatrix, MotorsMatrix, numberOfRods, gridPoints):

    motorPos = calculateMotorPositions(RodsMatrix, MotorsMatrix, numberOfRods)
    plt.clf()
    for row in range(0, numberOfRods):
        'Rods'
        y_r = np.ones(gridPoints)*row*0.2
        x_r = np.linspace(RodsMatrix[row, 0], RodsMatrix[row, 0] + RodsMatrix[row, 1], gridPoints, endpoint=True)
        
        plt.plot(x_r, y_r)
        'Motors'
        y_m = row*0.2 + 0.1
        x_m = motorPos[row]
        plt.scatter(x_m, y_m)

    plt.xlabel('x coordinate, motors in dots')
    plt.ylabel('Rods, rod 1 is y = 0, next rod is y += 0.2')
    plt.title('1D System')
    plt.pause(0.1)

def plotLengthEvolution(lengths):

    plt.scatter(np.linspace(0, lengths.size, lengths.size), lengths)
    plt.title('Rod length evolution')
    plt.xlabel('Time step')
    plt.ylabel('Rod length')
    #plt.show()

motors = np.random.rand(numberOfRods, 2)
#motors = np.zeros((numberOfRods, 2))
#motors[:,1] = np.random.rand(numberOfRods)
#motors[:,0] = np.zeros(numberOfRods)
rods = RodsArray(rods, motors, length, numberOfRods)
#plotSystem(rods, motors, numberOfRods, 10)
lengthArray = np.zeros(int(finalTime/recordTime))
lengthArray1 = np.zeros(int(finalTime/recordTime))

for tstep in range(0, int(finalTime/recordTime)):

    rods, motors = randwalk_shuffle(recordTime, rods, motors, numberOfRods, 0.01, 0.01)
    #rods1, motors1 = randwalk2(recordTime, rods, motors, numberOfRods, 0.01, 0.01)
    lengthArray[tstep] = systemLength(rods, numberOfRods)
    #lengthArray1[tstep] = systemLength(rods1, numberOfRods) + 0.1
    #plotSystem(rods, motors, numberOfRods, 10)
    #plotMotorPositions(rods, motors, numberOfRods)

#plotLengthEvolution(lengthArray)
#plotLengthEvolution(lengthArray1)
np.savetxt('length.dat', multipleSimulations(2000, length, numberOfRods, finalTime, recordTime, False))
plt.show()

np.savetxt('rods.dat', rods)
np.savetxt('motors.dat', motors)
