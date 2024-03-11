import math
import pybullet as p
import random
import numpy as np

class SpawnManager:
    def __init__(self, grid_size=10, spawn_objects=True, image_extraction=False):
        # Grid that represents the house
        self.grid = np.zeros((grid_size,grid_size))
        self.max_obj = 5
        self.objs = 0   
        self.goal = random.choice([1, 2, 3, 6, 7, 8])
        self.obstacles_pos = []
        # Determine robot spawn and fill cells
        self.pos, self.angle = self.robot_spawn(image_extraction)
        # Spawn the goal which occupies one cell on the opposite side
        self.spawn_house()
        self.spawn_goal()     
        # In case there are no spots dont look again
        self.valid_types = ['1x1']
        if spawn_objects:
            self.main_loop()


    def main_loop(self):
        # Dont spawn near walls
        for x in [0,1,8,9]:
            for y in range(10):
                self.grid[x,y] = 1
                self.grid[y,x] = 1
        # Look for spots to spawn objects    
        while self.objs < self.max_obj and self.valid_types != []:
            # Randomly select object type / 
            # 0: Cone
            # 1: Barrier
            # 2: Cement Bags
            # 3: Ladder
            # 4: Pallet
            '''
            if len(self.valid_types) == 2:
                obj_type = random.randint(0, 4)
            elif '1x1' in self.valid_types:
                obj_type = 0
            elif '2x2' in self.valid_types:
                obj_type = random.randint(1, 4)
            '''
            # force traffic cone
            obj_type = 0
            pos, roll = self.find_spots(obj_type)
            if pos == None:
                continue
            self.spawn_object(obj_type, pos, roll)

    def robot_spawn(self, image_extraction):

        # When training always spawn in the middle
        if not image_extraction:
            for i in range(7, 10):
                for j in range(4, 6):
                    self.grid[i,j] = 1
            return [0, -4.5], 0

        # 1/3 chance of spawning bottom, left, right
        choice = random.uniform(0, 1)
        random_pos = random.uniform(-4.5, 4.5)
        choice = 0.1
        if choice < 1/3:
            x_pos = random_pos
            y_pos = -4.5
            m = round(x_pos + 4.5)
            for i in range(8, 10):
                for j in range(m-1, m+2):
                    if j < 0 or j > 9:
                        continue
                    self.grid[i,j] = 1
        elif choice < 2/3:
            x_pos = -4.5
            y_pos = random_pos
            n = round(y_pos + 4.5)
            for i in range(n-1, n+2):
                for j in range(0, 2):
                    if i < 0 or i > 9:
                        continue
                    self.grid[i,j] = 1
        else:
            x_pos = 4.5
            y_pos = random_pos
            n = round(y_pos + 4.5)
            for i in range(n-1, n+2):
                for j in range(8, 10):
                    if i < 0 or i > 9:
                        continue
                    self.grid[i,j] = 1
        angle = - math.atan2(-x_pos, -y_pos) + random.uniform(-math.pi/4, math.pi/4)
        return [x_pos, y_pos], angle

    def find_spots(self, obj_type):

        spots = []
        # 1x1 objects
        if obj_type == 0:
            for i in range(10):
                for j in range(10):
                    if self.grid[i,j] == 0:
                        spots.append((i,j))
            if spots == []:
                self.valid_types.remove('1x1')
                return None, None
            spot = random.choice(spots)
            self.fill_cells(obj_type, spot)
            pos = [-4.5 + spot[1], 4.5 - spot[0], 0]
            roll = random.uniform(0, 2*math.pi)
            return pos, roll                  
        # 2x2 objects
        else:
            for i in range(9):
                for j in range(9):
                    if self.grid[i,j] == 0 and self.grid[i+1,j] == 0 and self.grid[i,j+1] == 0 and self.grid[i+1,j+1] == 0:
                        spots.append((i,j))
            if spots == []:
                self.valid_types.remove('2x2')
                return None, None
            spot = random.choice(spots)
            self.fill_cells(obj_type, spot)
            pos = [-4 + spot[1], 4 - spot[0], 0]
            roll = random.uniform(0, 2*math.pi)
            return pos, roll 

    def fill_cells(self, obj_type, spot):

        if obj_type == 0:
            for i in range(max(0, spot[0]-1), min(9, spot[0]+1) + 1):
                for j in range(max(0, spot[1]-1), min(9, spot[1]+1) + 1):
                    self.grid[i,j] = 1
        else:
            for i in range(max(0, spot[0]-1), min(9, spot[0]+2) + 1):
                for j in range(max(0, spot[1]-1), min(9, spot[1]+2) + 1):
                    self.grid[i,j] = 1

    def spawn_house(self):

        file = 'utils/objects/house.obj'

        houseCollisionShape = p.createCollisionShape(p.GEOM_MESH,
                                                    fileName=file)

        houseVisualShape = p.createVisualShape(p.GEOM_MESH,
                                                fileName=file)
        
        houseId = self.multibody(houseCollisionShape, houseVisualShape, 
                                 [0,0,0], [0,1/math.sqrt(2),1/math.sqrt(2),0])

        return houseId
    
    def spawn_goal(self):

        if self.goal not in [3, 6]:
            sway = np.random.uniform(-0.5, 0.5)
        elif self.goal == 3:
            sway = np.random.uniform(-0.5, 0)
        else:
            sway = np.random.uniform(0, 0.5)
        pos = [-4.5 + self.goal + sway, 4.5, 1]

        goalVisualShape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1], rgbaColor=[1, 0, 0, 1])

        for i in range(0, 3):
            for j in range(max(self.goal-2,0), min(self.goal+3,9)):
                self.grid[i,j] = 1

        self.goal = pos[:2]

        p.createMultiBody(baseMass=0, baseVisualShapeIndex=goalVisualShape, basePosition=pos)

    def spawn_object(self, obj_type, pos, roll):

        quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi+roll])

        obj_dict = {0: 'cone2',
                    1: 'barricade2',
                    2: 'cement2',
                    3: 'ladder2',
                    4: 'pallet'}
        
        self.objs += 1
        
        obj_name = obj_dict[obj_type]

        file = f'utils/objects/{obj_name}.obj'

        objCollisionShape = p.createCollisionShape(p.GEOM_MESH,
                                                    fileName=file)
        
        objVisualShape = p.createVisualShape(p.GEOM_MESH,
                                                fileName=file)
        
        pos[0] += random.uniform(-0.25, 0.25)
        pos[1] += random.uniform(-0.25, 0.25)

        objId = self.multibody(objCollisionShape, objVisualShape, 
                               pos, quat)
        
        self.obstacles_pos.append(pos[:2])

        return objId
        
    def multibody(self, collisionId, visualId, position, orientation=[0,1/math.sqrt(2),1/math.sqrt(2),0]):
        return p.createMultiBody(baseCollisionShapeIndex=collisionId,
                                    baseVisualShapeIndex=visualId,
                                    basePosition=position,
                                    baseOrientation=orientation)