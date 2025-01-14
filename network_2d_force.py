import numpy as np
import random
import time
import os
import json

"""
Term 2 Week 6

Authors: Ray & Xietao

Parameters.
n_cols: number of columns in the lattice.
n_rows: number of rows in the lattice.
length: length of the rods.
velocity_d: diffusion velocity. (in nm/ns)
velocity_p: persistence velocity. (in nm/ns)
seed: seed used to generate motor position.
finalTime: number of timesteps to be considered. (in ns)
recordTime: number of timesteps before each recording. (in ns)
motor_scale: initial position of the motors in each rod. Any value from [1 to 666.7) (in nm) 
threshold: insert here a number from (0, 1) for reduced connectivity.
ratio: mass of rod/mass of motor.
"""

def main(run_id, folder, parameters):

    n_cols = parameters["n_cols"]
    n_rows = parameters["n_rows"]
    length = parameters["length"]
    velocity_d = parameters["vd"]
    velocity_p = parameters["vp"]
    seed = parameters["seed"]
    finalTime = parameters["finalTime"]
    recordTime = parameters["recordTime"]
    motor_scale = parameters["motor_scale"]
    threshold = parameters["threshold"]
    polarity = parameters["polarity"]
    ratio = parameters["ratio"]
    L = parameters["L"]

    # Initialise seeds
    np.random.seed(seed*(run_id+1))
    random.seed(seed*(run_id+1))

    class Rod:
        def __init__(self, length, polarisation, pluspos, motor1, motor2):
            self.length = length
            self.polarisation = np.array(polarisation, dtype='float64')
            self.pluspos = np.array(pluspos, dtype='float64')
            self.minuspos = np.array(self.pluspos) + np.array(self.polarisation) * self.length
            self.motor1 = motor1
            self.motor2 = motor2
            self.pos1 = motor1.get_position()
            self.pos2 = motor2.get_position()

        def get_polarisation(self):
            return self.polarisation
        
        def get_pluspos(self):
            return self.pluspos
        
        # The self polarisation is pointing from plus to minus
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
            if not self.within_bounds():
                return self
            elif not self.in_line_with_rod(moved_motor_pos, self.polarisation):
                if fixed_motor_distance < moved_motor_distance:
                    movable_length = self.length - fixed_motor_distance
                    self.polarisation = moved_motor_pos - fixed_motor_pos
                    self.polarisation = self.polarisation / np.linalg.norm(self.polarisation)
                    self.pluspos = fixed_motor_pos - (self.polarisation * fixed_motor_distance)
                    self.minuspos = np.array(self.pluspos) + np.array(self.polarisation) * self.length
            
            
                else:
                    movable_length = fixed_motor_distance
                    self.polarisation = -moved_motor_pos + fixed_motor_pos
                    self.polarisation = self.polarisation / np.linalg.norm(self.polarisation)
                    self.pluspos = fixed_motor_pos - (self.polarisation * movable_length)
                    self.minuspos = np.array(self.pluspos) + np.array(self.polarisation) * self.length
            return self

    class Motor:
        def __init__(self, pos, v_p, v_D):
            self.position = np.array(pos, dtype='float64')
            self.v_p = v_p
            self.v_d = v_D
            self.connected_rods = []
            
        def add_rod(self, rod):
            self.connected_rods.append(rod)
            
        def move(self):
            # Pick a random rod to move along
            rod = random.choice(self.connected_rods)
            polarisation = rod.get_polarisation()
            movement = (self.v_p + self.v_d * random.uniform(-1,1)) * polarisation
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
        
        def get_relative_pos(self):
            relative_pos = np.zeros(len(self.connected_rods))
            for i, rod in enumerate(self.connected_rods):
                relative_pos[i] = np.linalg.norm(rod.pluspos - self.position)
            return relative_pos

    def generate_lattice_network(n, m, rod_length, motor_scale, threshold, polar):
        rods = dict()
        motors = dict()
        for i in range(n):
            for j in range(m):
                if i<n+1 and j<m+1:
                    motors[(i*m)+j] = Motor(np.multiply((i, j), motor_scale), velocity_p, velocity_d)
        for i in range(n):
            for j in range(m):
                dice_1 = np.random.random()
                dice_2 = np.random.random()
                if  i < n-1 and np.random.random() > threshold and dice_1 > polar:
                    rods[((i*m)+j, (i*m + m)+j)] = Rod(rod_length, (1,0), np.multiply((i - 0.5,j), motor_scale), motors[(i*m)+j], motors[(i*m + m)+j])
                elif i < n-1   and np.random.random() > threshold and dice_1 < polar:
                    rods[((i*m)+j, (i*m + m)+j)] = Rod(rod_length, (-1,0), np.multiply((i + 1.5,j), motor_scale), motors[(i*m + m)+j], motors[(i*m)+j])
                if  j < m - 1 and np.random.random() > threshold and dice_2 > polar:
                    rods[((i*m)+j, (i*m)+j + 1)] = Rod(rod_length, (0,1), np.multiply((i,j - 0.5), motor_scale), motors[(i*m)+j], motors[((i*m)+j) + 1])
                elif j < m - 1  and np.random.random() > threshold and dice_2 < polar:
                    rods[((i*m)+j, (i*m)+j + 1)] = Rod(rod_length, (0,-1), np.multiply((i,j + 1.5), motor_scale), motors[(i*m)+j+1], motors[((i*m)+j)])
                
        return rods, motors

    def movement(rods,motors, n, m, L):
        Rann = list(range(n))
        random.shuffle(Rann)
        Ranm = list(range(m))
        random.shuffle(Ranm)
        Force = 0
        for i in Rann:
            for j in Ranm:
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
                    #if rod in rods.values():
                    rodsinitp = rod.get_pluspos()[0]
                    rodsinitm = rod.get_minuspos()[0]
                    rod.reconnect(motor)
                    #if rod.get_pluspos()[0] >= L and rod.get_pluspos()[0] >= L:
                        #Force += 0.5 * (rod.get_pluspos()[0] + rod.get_minuspos()[0] - rodsinitp - rodsinitm)
                    if rod.get_pluspos()[0] >= L and rod.get_minuspos()[0] < L:
                        Force += rod.get_pluspos()[0] - rodsinitp
                    elif rod.get_pluspos()[0] < L and rod.get_minuspos()[0] > L:
                        Force += rod.get_minuspos()[0] - rodsinitm
        return rods, motors, Force/len(rods.values())  
                
    def simulate(rods, motors, n, m, L, timesteps):
        Forcecount = np.zeros(timesteps)
        for _ in range(timesteps):
            rods, motors, Force = movement(rods, motors, n, m, L)
            for motor in range(n*m):
                motors[motor].connected_rods = []
            Forcecount[_] = Force
        return rods, motors, Forcecount
    
    rods, motors = generate_lattice_network(n_cols, n_rows, length, motor_scale, threshold, polarity)
    force_over_time = np.zeros(finalTime)

    for timestep in range(finalTime):
        rods, motors, force_over_time[timestep] = simulate(rods, motors, n_cols, n_rows, L, recordTime)

    with open(f'{folder}\\force-all.dat', 'a') as f:
        for row in range(finalTime):
            f.write(f'{force_over_time[row]}\n')

if __name__=='__main__':

    with open('parameters.json') as json_file:
        parameters = json.load(json_file)
    
    n_runs = parameters["n_runs"]
    
    # Obtain current time to automatically save results
    unix_time = round(time.time())
    print(f'Current time: {unix_time}')

    # Create folder for saving data.
    folder_path = os.getcwd()
    folder = os.path.join(folder_path, str(unix_time))
    os.mkdir(folder)

    # Dump parameters
    with open(f'{folder}\\parameters.json', 'w') as fp:
        json.dump(parameters, fp)

    for keys, par in parameters.items():
        print(f'{keys}: {par}')
    
    start = time.time()
    for run in range(n_runs):
        if run % 10 == 0:
            middle = time.time() - start
            print(f'Run: {run}/{n_runs} - took {middle} seconds')
        main(run, folder, parameters)
    end = time.time()
    elapsed = end-start
    print(f'Total time elapsed: {elapsed} seconds.')
