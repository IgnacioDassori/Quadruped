import pybullet as p
import time
import pybullet_data
import math
import numpy as np
from resources.ramp import Ramp, Bridge

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

bridge = Ramp(client=client)

p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
#p.setDefaultContactERP(0)
#urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS 
urdfFlags = p.URDF_USE_SELF_COLLISION
quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi])

quadruped = p.loadURDF("laikago/laikago_toes.urdf",
							[0,0,0.45],
							quat, 
							flags = urdfFlags,
							useFixedBase=False,
							physicsClientId=client) 


visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="resources/objects/cone.obj",
    meshScale=[0.2, 0.2, 0.2],  # Adjust the scale if necessary
)

collisionShapeId = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="resources/objects/cone.obj",
    meshScale=[0.2, 0.2, 0.2],  # Adjust the scale if necessary
)

bodyUniqueId = p.createMultiBody(
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[1, 0, -0.5],  # Adjust the position if necessary
	baseOrientation=quat
)


jointIds=[1, 2, 5, 6, 9, 10, 13, 14]
initial_motor_positions = [0, -0.7, 0, -0.7, 0, -0.7, 0, -0.7]

p.loadURDF("plane.urdf", basePosition=[0,0,0], physicsClientId=client)

for i, position in zip(jointIds, initial_motor_positions):
	p.resetJointState(quadruped, jointIndex=i, targetValue=position)

#enable collision between lower legs
for j in range (p.getNumJoints(quadruped)):
	pass
		#print(p.getJointInfo(quadruped,j))

#2,5,8 and 11 are the lower legs
lower_legs = [2,5,8,11]
for l0 in lower_legs:
	for l1 in lower_legs:
		if (l1>l0):
			enableCollision = 1
			#print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
			#p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

jointIds=[]
paramIds=[]
jointOffsets=[]
jointDirections=[-1,1,1,1,1,1,-1,1,1,1,1,1]
jointAngles=[0,0,0,0,0,0,0,0,0,0,0,0]


for i in range (4):
	jointOffsets.append(0)
	jointOffsets.append(0)
	jointOffsets.append(0)

maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        #print(info)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                jointIds.append(j)

		
p.getCameraImage(480,320)
'''
#moving the robot according to data1.txt
p.setRealTimeSimulation(0)

joints=[]

with open(f"{pybullet_data.getDataPath()}/laikago/data1.txt","r") as filestream:
	for line in filestream:
		maxForce = p.readUserDebugParameter(maxForceId)
		currentline = line.split(",")
		frame = currentline[0]
		t = currentline[1]
		joints=currentline[2:14]
		for j in range (12):
			targetPos = float(joints[j])
			p.setJointMotorControl2(quadruped,jointIds[j],p.POSITION_CONTROL,jointDirections[j]*targetPos+jointOffsets[j], force=maxForce)
		p.stepSimulation()
		for lower_leg in lower_legs:
			#print("points for ", quadruped, " link: ", lower_leg)
			pts = p.getContactPoints(quadruped,-1, lower_leg)
			#print("num points=",len(pts))
			#for pt in pts:
			#	print(pt[9])
		time.sleep(1./500.)

'''



#debug parameters for motor positions
index = 0
for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        js = p.getJointState(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,(js[0]-jointOffsets[index])/jointDirections[index]))
                index=index+1

p.setRealTimeSimulation(1)
img_w = 128
img_h = 128

while (1):
	
	for i in range(len(paramIds)):
		c = paramIds[i]
		targetPos = p.readUserDebugParameter(c)
		maxForce = p.readUserDebugParameter(maxForceId)
		p.setJointMotorControl2(quadruped,jointIds[i],p.POSITION_CONTROL,jointDirections[i]*targetPos+jointOffsets[i], force=maxForce)

	# position and orientation of the agent
	agent_pos, agent_orn = p.getBasePositionAndOrientation(quadruped)
	euler = p.getEulerFromQuaternion(agent_orn)
	roll, pitch, yaw = euler
	vel = p.getBaseVelocity(quadruped)
	# rotation matrices
	roll_rot = np.array(([1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]))
	pitch_rot = np.array(([math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]))
	yaw_rot = np.array(([math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]))
	unit_vec = np.array([0, 0, 1])
	camera_up = np.array([0, 1, 0])
	camera_dist = 0.2
	camera_targ = 10000
	rotated = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), unit_vec)
	rotated_point = camera_dist*rotated + agent_pos
	rotated_view = camera_targ*rotated + agent_pos
	rotated_up = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), camera_up)
	# camera view matrix
	view_matrix = p.computeViewMatrix(rotated_point, rotated_view, rotated_up)
	# camera projection matrix
	projection_matrix = p.computeProjectionMatrixFOV(fov=90, aspect=1.0, nearVal=0.1, farVal=100.0)
	# get camera image
	imgs = p.getCameraImage(width=img_w, height=img_h, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
	