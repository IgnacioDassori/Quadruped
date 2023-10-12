import numpy as np
import matplotlib.pyplot as plt
import cv2
import pybullet as pb
import pybullet_data
from resources.ramp import Ramp
import datetime
import torch

def read_parameters(dbg_params):
    '''Reads values from debug parameters
    
    Parameters
    ----------
    dbg_params : dict
        Dictionary where the keys are names (str) of parameters and the values are
        the itemUniqueId (int) for the corresponing debug item in pybullet
    
    Returns
    -------
    dict 
        Dictionary that maps parameter names (str) to parameter values (float)
    '''
    values = dict()
    for name, param in dbg_params.items():
        values[name] = pb.readUserDebugParameter(param)

    return values

def interactive_camera_placement(pos_scale=10.,
                                 max_dist=10.,
                                 show_plot=True,
                                 verbose=True,
                                 ):
    '''GUI for adjusting camera placement in pybullet. Use the scales to adjust
    intuitive parameters that govern view and projection matrix.  When you are
    satisfied, you can hit the print button and the values needed to recreate
    the camera placement will be logged to console.
    In addition to showing a live feed of the camera, there are also two visual
    aids placed in the simulator to help understand camera placement: the target
    position and the camera. These are both shown as red objects and can be
    viewed using the standard controls provided by the GUI.
    Note
    ----
    There must be a simulator running in GUI mode for this to work
    Parameters
    ----------
    pos_scale : float
        Position scaling that limits the target position of the camera.
    max_dist : float
        Maximum distance the camera can be away from the target position, you
        may need to adjust if you scene is large
    show_plot : bool, default to True
        If True, then a matplotlib window will be used to plot the generated
        image.  This is beneficial if you want different values for image width
        and height (since the built in image visualizer in pybullet is always
        square).
    verbose : bool, default to False
        If True, then additional parameters will be printed when print button
        is pressed.
    '''
    np.set_printoptions(suppress=True, precision=4)

    dbg = dict()
    # for view matrix
    dbg['target_x'] = pb.addUserDebugParameter('target_x', -pos_scale, pos_scale, 0)
    dbg['target_y'] = pb.addUserDebugParameter('target_y', -pos_scale, 20, 0.25)
    dbg['target_z'] = pb.addUserDebugParameter('target_z', -pos_scale, pos_scale, 0.5)
    dbg['distance'] = pb.addUserDebugParameter('distance', 0, max_dist, 0.2)
    dbg['yaw'] =  pb.addUserDebugParameter('yaw', -180, 180, 0)
    dbg['pitch'] =  pb.addUserDebugParameter('pitch', -180, 180, 0)
    dbg['roll'] =  pb.addUserDebugParameter('roll', -180, 180, 0)
    dbg['upAxisIndex'] =  pb.addUserDebugParameter('toggle upAxisIndex', 1, 0, 1)

    # for projection matrix
    dbg['width'] = pb.addUserDebugParameter('width', 0, 200, 128)
    dbg['height'] = pb.addUserDebugParameter('height', 0, 200, 128)
    dbg['fov'] = pb.addUserDebugParameter('fov', 1, 180, 90)
    dbg['near_val'] = pb.addUserDebugParameter('near_val', 1e-6, 1, 0.1)
    dbg['far_val'] = pb.addUserDebugParameter('far_val', 1, 30, 100)

    # visual aids for target and camera pose
    '''
    target_vis_id = pb.createVisualShape(pb.GEOM_SPHERE,
                                         radius=0.01,
                                         rgbaColor=[1,0,0,0.7])
    target_body = pb.createMultiBody(0, -1, target_vis_id)
    '''
    
    camera_vis_id = pb.createVisualShape(pb.GEOM_BOX,
                                         halfExtents=[0.02, 0.05, 0.02],
                                         rgbaColor=[1,0,0,0.7])
    camera_body = pb.createMultiBody(0, -1, camera_vis_id)

    # pyplot window to show feed
    display_width = 320
    display_height = 320
    '''
    if show_plot:
        plt.figure()
        plt_im = plt.imshow(np.zeros((240,320,4)))
        plt.axis('off')
        plt.tight_layout(pad=0)
    '''

    dbg['print'] =  pb.addUserDebugParameter('print params', 1, 0, 1)
    old_print_val = 1
    while 1:
        values = read_parameters(dbg)

        target_pos = np.array([values[f'target_{c}'] for c in 'xyz'])
        upAxisIndex = (int(values['upAxisIndex']) % 2 ) + 1
        view_mtx = pb.computeViewMatrixFromYawPitchRoll(target_pos,
                                                        values['distance'],
                                                        values['yaw'],
                                                        values['pitch'],
                                                        values['roll'],
                                                        upAxisIndex)

        width = int(values['width'])
        height = int(values['height'])
        aspect = width/height
        proj_mtx = pb.computeProjectionMatrixFOV(values['fov'],
                                                 aspect,
                                                 values['near_val'],
                                                 values['far_val'])

        #update visual aids for camera, target pos
        #pb.resetBasePositionAndOrientation(target_body, target_pos, [0,0,0,1])

        view_mtx = np.array(view_mtx).reshape((4,4),order='F')
        cam_pos = np.dot(view_mtx[:3,:3].T, -view_mtx[:3,3])
        cam_euler = np.radians([values['pitch'],values['roll'],values['yaw']])
        cam_quat = pb.getQuaternionFromEuler(cam_euler)
        pb.resetBasePositionAndOrientation(camera_body, cam_pos, cam_quat)

        view_mtx = view_mtx.reshape(-1, order='F')
        '''
        depth_img = (pb.getCameraImage(width, height, view_mtx, proj_mtx)[3]*255).astype(np.uint8)
        img = cv2.resize(depth_img, (display_width, display_height))
        colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        '''
        img = pb.getCameraImage(width, height, view_mtx, proj_mtx)[2]
        colored = cv2.resize(img, (display_width, display_height))
        cv2.imshow('depth', colored)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            dt = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            tensor_rgb = torch.from_numpy(colored)
            torch.save(tensor_rgb, f"images/images_rgb_pt/{dt}.pt")
            cv2.imwrite(f"images/images_rgb/{dt}.png", colored)
        '''
        elif key == ord('s'):
            dt = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            tensor_depth = torch.from_numpy(depth_img)
            torch.save(tensor_depth, f"images/images_pt/{dt}.pt")
            cv2.imwrite(f"images/images_png/{dt}.png", colored)
        '''
        if old_print_val != values['print']:
            old_print_val = values['print']
            print("\n========================================")
            print(f"VIEW MATRIX : \n{np.array_str(view_mtx)}")
            print(f"PROJECTION MATRIX : \n{np.array_str(view_mtx)}")
            if verbose:
                print(f"target position : {np.array_str(target_pos)}")
                print(f"distance : {dbg['distance']:.2f}")
                print(f"yaw : {dbg['yaw']:.2f}")
                print(f"pitch : {dbg['pitch']:.2f}")
                print(f"roll : {dbg['roll']:.2f}")
                print(f"upAxisIndex : {upAxisIndex:d}")
                print(f"width : {width:d}")
                print(f"height : {height:d}")
                print(f"fov : {dbg['fov']:.1f}")
                print(f"aspect : {aspect:.2f}")
                print(f"nearVal : {dbg['near_val']:.2f}")
                print(f"farVal : {dbg['far_val']:.2f}")
            print("========================================\n")

def initialize_simulator():
    '''Creates a pybullet simulator in GUI mode and adds some objects so the
    camera has something to look at
    '''
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF('plane.urdf')

    # add a duck
    pb.loadURDF('duck_vhacd.urdf', baseOrientation=[0.5,0.5,0.5,0.5], globalScaling=2)
    pb.loadURDF('r2d2.urdf', [0.2, 0.2, 0.15],
                pb.getQuaternionFromEuler([0, 0, 1.234]),
                globalScaling=0.3)

if __name__ == "__main__":
    # Substitute this for the initialization of your simulator
    client = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ramp = Ramp(client=client)
    pb.loadURDF("plane.urdf", [0,0,ramp._lowest])

    interactive_camera_placement()