import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import json

"""
Term 2 Week 1

Authors: Ray & Xietao
"""
"""
Parameters.
velocity_d: diffusion velocity. (in nm/ns)
velocity_p: persistence velocity. (in nm/ns)
seed: seed used to generate motor position.
makeAnimation: set True if you want simulation animated, set False to skip.
repeatedSystems: set True for multiple simulations with different seed.
numberOfRepetitions: if repeatedSystems is True, then set here the number of desired simualations.
seqUpdate: 0 for top to bottom, 1 for bottom to top, 
2 is (uniformly) self-avoiding random, 3 for (uniformly) completely random. Default is 0.
diffSampling: 0 is constant, 1 for uniformly distribution, 2 Gaussian.
persSample: 0 is constant, 1 for uniformly distribution, 2 Gaussian.
allowMotorCrossing: set True to allow motor crossing.
finalTime: number of timesteps to be considered. (in ns/2)
recordTime: number of timesteps before each recording. (in ns)
motors: motors initial condition w.r.t the rods, enter the full matrix.
"""

n_cols = 3
n_rows = 3
length = 1000
velocity_d = 0
velocity_p = 100
seed = 1
makeAnimation = False
seqUpdate = 2
diffSampling = 1
persSample = 1
allowMotorCrossing = True
finalTime = 4000
recordTime = 2

# Format parameters
parameters = {
    "vp": velocity_p,
    "vd": velocity_d,
    "seed": seed,
    "seqUpdate": seqUpdate,
    "diffSampling": diffSampling,
    "persSampling": persSample,
    "allowCrossing": allowMotorCrossing,
    "finalTime": finalTime,
    "recordTime": recordTime,
}

# Folder to save results. Set to 0 for same workspace.
folder_path = 0

# Obtain current time to automatically save different results
unix_time = time.time()
print(f'Current time: {unix_time}')

# Initialise seeds
np.random.seed(seed)
random.seed(seed)

class Rod:
    def __init__(self, length, polarisation, pluspos, motor1, motor2):
        self.length = length
        self.polarisation = np.array(polarisation,dtype='float64')
        self.pluspos = np.array(pluspos ,dtype='float64')
        self.minuspos = np.array(self.pluspos) + np.array(self.polarisation) * self.length
        self.motor1 = motor1
        self.motor2 = motor2
        self.pos1 = motor1.get_position()
        self.pos2 = motor2.get_position()

    def get_polarisation(self):
        return self.polarisation
    
    def get_pluspos(self):
        return self.pluspos
    
    #The self polarisation is pointing from plus to minus
    def get_minuspos(self):
        return self.minuspos
    
    def get_vector(self):
        return self.polarisation * self.length
    
    def within_bounds(self):
        """Check if a position is within the bounds of the rod"""
        # check the distance between pos and pos1 and pos and pos2 
        return np.linalg.norm(self.pluspos - self.pos1) <= self.length and np.linalg.norm(self.pluspos - self.pos2) <= self.length and np.linalg.norm(self.minuspos - self.pos1) <= self.length and np.linalg.norm(self.minuspos - self.pos2) <= self.length
    
    def in_line_with_rod(self, pos, polarisation):
        """Check if a position is in line with the rod"""
        # check if pos = pos1 + lambda * polarisation
        lambda_value = np.dot((pos - self.pluspos), polarisation) 
        return np.allclose(pos, self.pluspos + lambda_value * self.polarisation)

    def reconnect(self, fixed_motor):
        """Reconnect a moved motor to the rod if it has moved outside of the rod"""
        if fixed_motor == self.motor1:
            moved_motor = self.motor2
            moved_motor_pos = self.motor2.get_position()
            fixed_motor_pos = self.motor1.get_position()
        else:
            moved_motor = self.motor1
            moved_motor_pos = self.motor1.get_position()
            fixed_motor_pos = self.motor2.get_position()
        fixed_motor_distance = np.linalg.norm(fixed_motor.get_position() - self.pluspos)
        moved_motor_distance = np.linalg.norm(moved_motor.get_position() - self.pluspos)
        #print('inline', self.in_line_with_rod(moved_motor_pos, self.polarisation))
        if not self.within_bounds():
            return self
        elif not self.in_line_with_rod(moved_motor_pos, self.polarisation):
            #print('elif', fixed_motor_distance < moved_motor_distance)
            if fixed_motor_distance < moved_motor_distance:
                movable_length = self.length - fixed_motor_distance
                self.polarisation = moved_motor_pos - fixed_motor_pos
                self.polarisation = self.polarisation / np.linalg.norm(self.polarisation)
                #self.length = np.linalg.norm(moved_motor_pos - fixed_motor_pos)
                self.pluspos = fixed_motor_pos - (self.polarisation * fixed_motor_distance)
                self.minuspos = np.array(self.pluspos) + np.array(self.polarisation) * self.length
        
        
            else:
                movable_length = self.length - fixed_motor_distance
                self.polarisation = moved_motor_pos - fixed_motor_pos
                self.polarisation = self.polarisation / np.linalg.norm(self.polarisation)
                #self.length = np.linalg.norm(fixed_motor_pos - moved_motor_pos)
                self.pluspos = fixed_motor_pos - (self.polarisation * movable_length)
                self.minuspos = np.array(self.pluspos) + np.array(self.polarisation) * self.length
        return self


class Motor:
    def __init__(self, pos, v_p, v_D):
        self.position = np.array(pos,dtype='float64')
        self.v_p = v_p
        self.v_d = v_D
        self.connected_rods = []
        
    def add_rod(self, rod):
        self.connected_rods.append(rod)
        
    def move(self):
        # Pick a random rod to move along
        #print(len(self.connected_rods))
        rod = random.choice(self.connected_rods)
        polarisation = rod.get_polarisation()
        movement = (self.v_p + self.v_d * random.uniform(-1,1)) * polarisation
        #print(type(movement))
        self.position += movement
        # Check if motor goes out of the rod
        out_of_bounds = False
        for rod in self.connected_rods:
            if not rod.within_bounds():
                out_of_bounds = True
        if out_of_bounds:
            self.position -= movement
    def get_position(self):
        return self.position
    
    def get_v_p(self):
        return self.v_p
    
    def get_v_d(self):
        return self.v_d

def generate_lattice_network(n, m, rod_length):
    rods = dict()
    motors = dict()
    for i in range(n):
        for j in range(m):
            if i<n+1 and j<m+1:
                motors[(i*m)+j] = Motor(np.multiply((i, j), rod_length/2), velocity_p, velocity_d)
    for i in range(n):
        for j in range(m):
            if  i < n - 1 :
                rods[((i*m)+j, (i*m + m)+j)] = Rod(rod_length, (1,0), np.multiply((i,j), rod_length/2), motors[(i*m)+j], motors[(i*m + m)+j])
            if  j < m - 1:
                rods[((i*m)+j, (i*m)+j + 1)] = Rod(rod_length, (0,1), np.multiply((i,j), rod_length/2), motors[(i*m)+j ], motors[((i*m)+j) + 1])
            
    return rods, motors

def movement(rods,motors, n, m):
    for i in range(n):
        for j in range(m):
            motor = motors[i*m + j]
            if ((i*m)+j) < (n*m) and ((i*m + m)+j) < (n*m) and ((i*m)+j, (i*m + m)+j) in rods:
                rod1 = rods[(i*m)+j,(i*m + m)+j]
                motor.add_rod(rod1)
            if ((i*m - m)+j) >= 0 and ((i*m)+j) < (n*m) and ((i*m - m)+j, (i*m)+j) in rods:
                rod2 = rods[(i*m - m)+j,(i*m)+j]
                motor.add_rod(rod2)
            if ((i*m)+j) < (n*m) and ((i*m)+j + 1) < (n*m) and ((i*m)+j, (i*m)+j + 1) in rods:
                rod3 = rods[(i*m)+j,(i*m)+j + 1]
                motor.add_rod(rod3)
            if ((i*m)+j - 1) >= 0 and ((i*m)+j) < (n*m) and ((i*m)+j - 1, (i*m)+j) in rods:
                rod4 = rods[(i*m)+j - 1,(i*m)+j]
                motor.add_rod(rod4)
            motor.move()
            for rod in motor.connected_rods:
                if rod in rods.values():
                    rod.reconnect(motor)


    return rods, motors       
            
def simulate(rods,motors, n, m, timesteps):
    for _ in range(timesteps):
        movement(rods,motors, n, m)
        for motor in range(n*m):
            motors[motor].connected_rods = []
        
    return rods, motors

# Create folder for saving data.
if folder_path == 0:
    folder_path = os.getcwd()
folder = os.path.join(folder_path, str(unix_time))
os.mkdir(folder)

# Dump parameters
with open(f'{folder}\\parameters.json', 'w') as fp:
    json.dump(parameters, fp)

rods, motors = generate_lattice_network(n_cols, n_rows, length)

for timestep in range(1):

    plt.clf()
    rods, motors = simulate(rods, motors, n_cols, n_rows, 1)

    # Draw the rods
    for rod in rods.values():
        xpos = np.array([rod.get_pluspos()[0], rod.get_minuspos()[0]])
        ypos = np.array([rod.get_pluspos()[1], rod.get_minuspos()[1]])
        plt.plot(xpos, ypos, marker='>')

    # Draw the motors
    for motor in motors.values():
        plt.scatter(motor.get_position()[0], motor.get_position()[1])
    plt.scatter(0, 0, label=timestep)
    plt.legend()
    plt.pause(0.1)
plt.show()
