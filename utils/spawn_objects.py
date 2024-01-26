import pybullet as p
import math
import random

def spawn_house():

    file = 'objects/house.obj'

    houseCollisionShape = p.createCollisionShape(p.GEOM_MESH,
                                                fileName=file)

    houseVisualShape = p.createVisualShape(p.GEOM_MESH,
                                            fileName=file)
    
    houseId = multibody(houseCollisionShape, houseVisualShape, [0,0,0])

    return houseId

def spawn_object(obj_name, position, orientation):

    file = f'objects/{obj_name}.obj'

    objCollisionShape = p.createCollisionShape(p.GEOM_MESH,
                                                fileName=file)
    
    objVisualShape = p.createVisualShape(p.GEOM_MESH,
                                            fileName=file)
    
    objId = multibody(objCollisionShape, objVisualShape, position, orientation)

    return objId
    
def multibody(collisionId, visualId, position, orientation=[0,1/math.sqrt(2),1/math.sqrt(2),0]):
    return p.createMultiBody(baseCollisionShapeIndex=collisionId,
                                baseVisualShapeIndex=visualId,
                                basePosition=position,
                                baseOrientation=orientation)


if __name__ == '__main__':

    p.connect(p.GUI)
    id = spawn_house()
    print(p.getDynamicsInfo(id, -1))

    while True:
        pass