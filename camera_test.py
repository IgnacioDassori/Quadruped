import pybullet as p
import pybullet_data

# Start PyBullet simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
laikagoStartPos = [0, 0, 0.2]
laikagoStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
laikagoId = p.loadURDF("laikago/laikago_toes.urdf", laikagoStartPos, laikagoStartOrientation)

# Enable camera sensor
cam_target_pos = [0, 0, 0]
cam_distance = 1.0
cam_yaw = 0
cam_pitch = -30
cam_resolution = [640, 480]
cam_near_plane = 0.01
cam_far_plane = 10.0

cam_id = p.addUserDebugCamera(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_resolution[0], cam_resolution[1])
p.setDebugObjectColor(cam_id, 0, 0, 0)

# Main simulation loop
while True:
    # Get the RGB image from the camera sensor
    img = p.getCameraImage(cam_resolution[0], cam_resolution[1], cam_id, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_img = img[2]

    # Do something with the RGB image
    # For example, display it using OpenCV
    import cv2
    cv2.imshow("RGB Image", rgb_img)
    cv2.waitKey(1)

    # Perform other simulation steps

# End simulation
p.disconnect()