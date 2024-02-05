import pybullet as p
import math
import random
import numpy as np

class SpawnManager:
    def __init__(self, grid_size=10):
        # Grid that represents the house
        self.grid = np.zeros((grid_size,grid_size))
        self.max_obj = 5
        self.objs = 0
        self.goal = random.randint(2, 7)
        # Robot always spawns in cells (9,4) & (9,5)
        for i in range(8, 10):
            for j in range(3, 7):
                self.grid[i,j] = 1
        # Spawn the goal which occupies one cell on the opposite side
        self.spawn_house()
        self.spawn_goal()     
        # In case there are no spots dont look again
        self.valid_types = ['1x1', '2x2']   
        # Look for spots to spawn objects    
        while self.objs < self.max_obj and self.valid_types != []:
            # Randomly select object type / 
            # 0: Cone
            # 1: Barrier
            # 2: Cement Bags
            # 3: Ladder
            # 4: Pallet
            obj_type = random.randint(0, 4) if len(self.valid_types) == 2 else 0
            pos, roll = self.find_spots(obj_type)
            if pos == None:
                continue
            self.spawn_object(obj_type, pos, roll)


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

        pos = [-4.5 + self.goal, 4.5, 0]

        radius = 1
        goalVisualShape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])

        for i in range(0, 3):
            for j in range(self.goal-2, self.goal+3):
                self.grid[i,j] = 1

        p.createMultiBody(baseMass=0, baseVisualShapeIndex=goalVisualShape, basePosition=pos)

    def spawn_object(self, obj_type, pos, roll):

        quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi+roll])

        obj_dict = {0: 'cone',
                    1: 'barricade',
                    2: 'cement',
                    3: 'ladder',
                    4: 'pallet'}
        
        self.objs += 1
        
        obj_name = obj_dict[obj_type]

        file = f'utils/objects/{obj_name}.obj'

        objCollisionShape = p.createCollisionShape(p.GEOM_MESH,
                                                    fileName=file)
        
        objVisualShape = p.createVisualShape(p.GEOM_MESH,
                                                fileName=file)
        
        objId = self.multibody(objCollisionShape, objVisualShape, 
                               pos, quat)

        return objId
        
    def multibody(self, collisionId, visualId, position, orientation=[0,1/math.sqrt(2),1/math.sqrt(2),0]):
        return p.createMultiBody(baseCollisionShapeIndex=collisionId,
                                    baseVisualShapeIndex=visualId,
                                    basePosition=position,
                                    baseOrientation=orientation)
    

if __name__ == '__main__':

    p.connect(p.GUI)
    sm = SpawnManager()


    while True:
        pass