"""
Week 10 - EXPERIMENTAL

Ray & Xietao
"""

import numpy as np
import matplotlib.pyplot as plt
import random


"""
Parameters.
numberOfRods: the number of rods, also the number of motors.
length: the length of the rods, enter a matrix of size equal to the number of rods.
velocity_d: diffusion velocity.
velocity_p: persistence velocity.
seed: seed used to generate motor position.
makeAnimation: set True if you want simulation animated, set False to skip.
repeatedSystems: set True for multiple simulations with different seed.
numberOfRepetitions: if repeatedSystems is True, then set here the number of desired simualations.
seqUpdate: 0 for top to bottom, 1 for bottom to top, 
2 is (uniformly) self-avoiding random, 3 for (uniformly) completely random. Default is 0.
diffSampling: 0 is constant, 1 for uniformly distribution, 2 Gaussian.
persSample: 0 is constant, 1 for uniformly distribution, 2 Gaussian.
allowMotorCrossing: set True to allow motor crossing.
finalTime: number of timesteps to be considered.
recordTime: number of timesteps before each recording.
motors: motors initial condition w.r.t the rods, enter the full matrix.
"""

numberOfRods = 21
length = np.ones(numberOfRods)*1000
velocity_d = 10
velocity_p = 1
seed = 1
makeAnimation = True
repeatedSystems = False
numberOfRepetitions = 20
# Don't change these
np.random.seed(seed)
random.seed(seed)
# ----
seqUpdate = 2
diffSampling = 2
persSample = 0
allowMotorCrossing = True
finalTime = 2000
recordTime = 2
motors = np.random.rand(numberOfRods, 2)*1000
rods = np.random.rand(numberOfRods, 2)
m_motor = 1
m_rods = 3
L = 3

def RodsArray(Rposition, Mposition, length, numberOfRods):

    for i in range(1, numberOfRods):
        Rposition[i, 0] += Rposition[i-1, 0] + Mposition[i-1, 0] - Mposition[i-1, 1]

    Rposition[:, 1] = length #Array
    
    return Rposition

def randwalk(numberofsteps, Rposition, Mposition, numberOfRods, pervelo, diffvelo):
    # random walk with velocity v_d and directional movement with v_p 
    v_p = pervelo
    v_d = diffvelo
    velocity_count = np.zeros(numberofsteps)
    Force = np.zeros(numberofsteps - 1)
    for _ in range(0, numberofsteps): 
        order = list(range(0, -numberOfRods, -1))
        if seqUpdate == 2:
            random.shuffle(order)
        elif seqUpdate == 1:
            order.reverse()
        elif seqUpdate == 3:
            order = random.choices(order, k=numberOfRods)
        for i in order: # for each motor
            dice1 = np.random.uniform(0,1)
            v_p0 = pervelo
            v_d = diffvelo
            if diffSampling == 0:
                pass
            elif diffSampling == 1:
                v_d *= random.uniform(0,1)
            elif diffSampling == 2:
                v_d = abs(np.random.normal(0, (v_d)))
            if persSample == 0:
                pass
            elif persSample == 1:
                v_p0 *= random.uniform(0,1)
            elif persSample == 2:
                v_p0 = abs(np.random.normal(0, v_p))
            Mpositiontot = Rposition[:, 0] + Mposition[:, 0]
            Relativeposition = Mpositiontot[i] - Mpositiontot
            #print(Relativeposition)
            Left_number = np.where((Relativeposition < L) & (Relativeposition > 0))[0]
            Right_number = np.where((Relativeposition >-L) & (Relativeposition < 0))[0]
            #print(len(Right_number))
            if len(Left_number) >= len(Right_number):
                v_p = v_p0
            elif len(Left_number) < len(Right_number):
                v_p = - v_p0
            else:
                v_p = 0
            
            if dice1 < 0.5 : #random walk to the right
                Mposition[i, 0] += v_p + v_d
                if allowMotorCrossing:
                    if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1]):
                        Mposition[i, 0] -= v_p + v_d
                else:
                    if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1] or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d) or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d)):
                        Mposition[i, 0] -= v_p + v_d
            elif dice1 > 0.5 :#random walk to the left
                Mposition[i, 0] += v_p - v_d
                if allowMotorCrossing:
                    if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1]):
                        Mposition[i, 0] -= v_p - v_d
                else:
                    if (0 >= Mposition[i, 0] or  Mposition[i, 0] >= Rposition[i, 1] or Mposition[i, 0] - abs(v_p - v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p - v_d) or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d)):
                        Mposition[i, 0] -= v_p - v_d
            else:
                pass
            
            dice2 = np.random.uniform(0,1)
            if dice2 < 0.5 :#random walk to the right
                Mposition[i, 1] += -v_p + v_d
                if allowMotorCrossing:
                    if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1]):
                        Mposition[i, 1] -= -v_p + v_d
                else:
                    if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1] or Mposition[i + 1, 0] - abs(-v_p + v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p + v_d) or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d)):
                        Mposition[i, 1] -= -v_p + v_d
            elif dice2 > 0.5 :#random walk to the left
                Mposition[i, 1] += -v_p - v_d
                if allowMotorCrossing:
                    if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1]):
                        Mposition[i, 1] -= -v_p - v_d
                else:
                    if (0 >= Mposition[i, 1] or  Mposition[i, 1] >= Rposition[i, 1] or Mposition[i + 1, 0] - abs(-v_p - v_d) <= Mposition[i, 1] <= Mposition[i + 1, 0] + abs(-v_p - v_d) or Mposition[i, 0] - abs(v_p + v_d) <= Mposition[i - 1, 1] <= Mposition[i, 0] + abs(v_p + v_d)):
                        Mposition[i, 1] -= -v_p - v_d
            else:
                pass
        center_velocity = np.sum((Mposition[:-1, 0] - Mposition[:-1, 1]).cumsum() - Rposition[1:, 0])
        velocity_count[_] = center_velocity
        Rposition[1:, 0] = (Mposition[:-1, 0] - Mposition[:-1, 1]).cumsum()   
    Force = velocity_count[1:] - velocity_count[0:numberofsteps - 1]
    print(velocity_count)
    return Rposition, Mposition, velocity_count

plt.xlabel("time$/\mu s$")
plt.ylabel("Force$/\mu N$")
timestep = 100
x = np.linspace(0, timestep, timestep - 1)
Forcecount = randwalk(timestep, rods, motors, 21, 0.001, 0.01)[2][1:]

plt.plot(x, Forcecount)
plt.show()

print(np.sum(Forcecount))
print(np.sum(abs(Forcecount)))

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
    
    "Initialise arrays."
    finalLength = np.zeros(numberOfSimulations)
    allLengths = np.zeros((numberOfSimulations, int(finalTime/recordTime)))
    allMotors = np.zeros((numberOfSimulations, numberOfRods))
    seeds = [None]*numberOfSimulations

    "Create seeds for initial conditions."
    for simul, seed in enumerate(np.random.choice(1000000, numberOfSimulations, replace=False)):
        seeds[simul] = np.random.default_rng(seed=seed)
    
    "Execute simulations."
    for simul in range(0, numberOfSimulations):
        rods = np.zeros((numberOfRods, 2))
        motors = seeds[simul].random((numberOfRods, 2))
        rods = RodsArray(rods, motors, length, numberOfRods)

        for tstep in range(0, int(finalTime/recordTime)):
            rods, motors = randwalk(recordTime, rods, motors, numberOfRods, velocity_p, velocity_d)
            lengthArray[tstep] = systemLength(rods, numberOfRods)
        if plotEvol:
            plt.scatter(np.linspace(0, lengthArray.size, lengthArray.size), lengthArray)
            plt.title('Rod length evolution')
            plt.xlabel('Time step')
            plt.ylabel('Rod length')
        if simul % 100 == 0:
            print(f'Currently {simul}/{numberOfSimulations}')

        finalLength[simul] = lengthArray[-1]
        allLengths[simul] = lengthArray
        allMotors[simul] = motors[:, 0]

    return finalLength, allLengths, allMotors

def multipleSimulsSame(type_v, max_v, number, finalTime, recordTime, startFrom, rodsArray, motorsArray):
    results = [None]*number
    rods = rodsArray
    motors = motorsArray
    for simul in range(0, number):
        for _ in range(0, int(finalTime/recordTime)):
            if type_v == 0:
                rods, motors = randwalk(recordTime, rods, motors, numberOfRods, velocity_p, (max_v/number)*simul)
            else:
                rods, motors = randwalk(recordTime, rods, motors, numberOfRods, (max_v/number)*simul, velocity_d)
        
        results[simul] = np.extract(motors[:, 0] >= startFrom, motors).size
        if simul % 5 == 0:
            print(f'Currently {simul}/{number}')
    return results

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

    plt.plot(np.linspace(0, lengths.size, lengths.size), lengths)
    plt.title('Rod length evolution')
    plt.xlabel('Time step')
    plt.ylabel('Rod length')
    #plt.show()

rods = np.zeros((numberOfRods, 2))
rods = RodsArray(rods, motors, length, numberOfRods)
lengthArray = np.zeros(int(finalTime/recordTime))

for tstep in range(0, int(finalTime/recordTime)):

    rods, motors, Force = randwalk(recordTime, rods, motors, numberOfRods, velocity_p, velocity_d)
    lengthArray[tstep] = systemLength(rods, numberOfRods)
    if makeAnimation == True:
        plotSystem(rods, motors, numberOfRods, 10)

plt.show()
plotLengthEvolution(lengthArray)
if repeatedSystems == True:
    lengthFinal, lengths, motors_all = multipleSimulations(numberOfRepetitions, length, numberOfRods, finalTime, recordTime, False)
plt.show()

# Save to files, use graphs jupyter notebook to plot these
np.savetxt('rods.dat', rods)
np.savetxt('motors.dat', motors)
if repeatedSystems == True:
    np.savetxt('length_final.dat', lengthFinal)
    np.savetxt('length_all.dat', lengths)
    np.savetxt('motors_multiple.dat', motors_all)

"""
y = multipleSimulsSame(1, 0.005, 40, finalTime, recordTime, 0.95, rods, motors)
x = np.linspace(0, 0.005, 40)

plt.scatter(x, y)
plt.title('Varying v_p')
plt.xlabel('v_p')
plt.ylabel('amount of motors with x > 0.95')
plt.show()
np.savetxt('v_p_change.dat', y)
"""