import pybullet as p
import random
from resources.ramp import Ramp
import pybullet_data

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def spawn_ramp():

    # Randomly generate the ramp's orientation (roll, pitch, yaw)
    roll = random.uniform(0, 0)  # You can adjust the range as needed
    pitch = random.uniform(-90, 90)
    yaw = random.uniform(0, 0)  # Random orientation along the yaw axis

    # Randomly generate the ramp's length
    ramp_length = random.uniform(1.0, 3.0)  # Adjust the range as needed

    # Define the ramp's position
    ramp_position = [0, 0, 0]  # You can adjust the position as needed

    # Ramp material
    friction_coefficient = 0.5  # Adjust as needed
    ramp_material = p.createPhysicsProperties(
        restitution=0.0,  # Elasticity (set to 0 for non-elastic collisions)
        rollingFriction=0.0,  # Rolling friction
        lateralFriction=friction_coefficient  # Lateral friction (set to the desired value)
    )    

    # Create a box-shaped collision shape for the ramp
    ramp_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.1, ramp_length / 2,  0.1],  # Adjust dimensions as needed
        material=ramp_material  # Add the ramp material
    )

    # Create a box-shaped visual shape for the ramp (make it visible)
    ramp_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.1, ramp_length / 2,  0.1],  # Adjust dimensions as needed
        rgbaColor=[0.8, 0.2, 0.2, 1]  # Adjust the color (RGBA format)
    )

    # Create the ramp as a multi-body object with both collision and visual shapes
    ramp_body = p.createMultiBody(
        baseMass=0,  # Set mass to 0 for a static object
        baseCollisionShapeIndex=ramp_collision_shape,
        baseVisualShapeIndex=ramp_visual_shape,  # Add visual shape
        basePosition=ramp_position,
        baseOrientation=p.getQuaternionFromEuler([roll, pitch, yaw])
    )

ramp = Ramp(client)
p.loadURDF("plane.urdf", [0, 0, ramp._lowest])
# Run simulation
while True:
    p.stepSimulation()