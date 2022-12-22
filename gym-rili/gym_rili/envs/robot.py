import os
import numpy as np
import pybullet as p
import pybullet_data
from gym_rili.envs.assets.objects import YCBObject, InteractiveObj, RBOObject
import gym
from gym import spaces
from gym.utils import seeding


# goal index
goal_idx = [0, 1, 2]

# imaginary plane: height
h = 0.0


class Robot(gym.Env):

    def __init__(self, GUI=False):

        self.action_space = spaces.Box(
            low=-0.1,
            high=+0.1,
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(3,),
            dtype=np.float32
        )

        # create simulation (GUI)
        self.GUI = GUI
        if not self.GUI:
            p.connect(p.DIRECT)
        else:
            self.urdfRootPath = pybullet_data.getDataPath()
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
            p.setGravity(0, 0, -9.81)

            # set up camera
            self._set_camera()

            # load objects
            p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
            p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        # position of goals
        pos_goal1 = np.array([0.6, +0.3, 0.0])
        pos_goal2 = np.array([0.8, 0.0, 0.0])
        pos_goal3 = np.array([0.6, -0.3, 0.0])
        self.pos_goals = [pos_goal1, pos_goal2, pos_goal3]
        self.goal = YCBObject('025_mug')

        # load a panda robot
        self.panda = Panda()

        self.change_partner = 0.99
        self.reset_choice = 0.999
        self.choice = 0
        self.panda.reset()
        self.ego = self.panda.state['ee_position']
        self.other = np.copy(self.pos_goals[self.choice])
        self.partner = 0
        self.timestep = 0

    def visualizeGoal(self, pos_goal):
        self.goal.load()
        p.resetBasePositionAndOrientation(self.goal.body_id, pos_goal, [0, 0, 0, 1])

    def recordVideo(self):
        if not os.path.exists("evals/"+self.run_name):
            os.makedirs("evals/"+self.run_name)
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                            os.path.join("evals", self.run_name,'{}.mp4'.format(self.i_episode)))

    def record_params(self, run_name, i_episode):
        self.run_name = run_name
        self.i_episode = i_episode

    def set_params(self, change_partner):
        self.change_partner = change_partner

    def _get_obs(self):
        return self.panda.state['ee_position']

    def reset(self):
        if self.GUI:
            self.visualizeGoal(self.pos_goals[self.choice])
        return self._get_obs()

    def step(self, action):
        self.timestep += 1
        self.panda.place_ee(self.panda.state['ee_position'] + action, [1, 0, 0, 0])
        reward = -np.linalg.norm(self.other - self.panda.state['ee_position']) * 100
        done = False
        p.stepSimulation()
        if self.timestep == 10:
            self.timestep = 0
            if self.choice == 0 and self.partner < 3:
                reward += 100
            elif self.choice == 1 and self.partner == 3:
                reward += 100
            if np.random.random() > self.reset_choice:
                self.choice = np.random.choice(goal_idx)
            ## choose a new partner from the three options
            if np.random.rand() > self.change_partner:
                self.partner = np.random.choice(range(4))

            if self.partner == 0:
                if self.panda.state['ee_position'][1] < self.other[1]:
                    self.choice = goal_idx[(self.choice+1) % 3]
                else:
                    self.choice = goal_idx[(self.choice-1) % 3]

            elif self.partner == 1:
                if self.panda.state['ee_position'][1] < self.other[1]:
                    pass
                else:
                    self.choice = goal_idx[(self.choice+1) % 3]

            elif self.partner == 2:
                self.choice = goal_idx[(self.choice+1) % 3]

            elif self.partner == 3:
                self.choice = goal_idx[(self.choice-1) % 3]

            self.panda.reset()
            self.other = np.copy(self.pos_goals[self.choice])
        return self._get_obs(), reward, done, {}

    def close(self):
        p.disconnect()

    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                     height=self.camera_height,
                                                                     viewMatrix=self.view_matrix,
                                                                     projectionMatrix=self.proj_matrix)
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-35,
                                     cameraTargetPosition=[0.5, 0, 0.1])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)








#######################################################################
##################              PANDA ROBOT             ###############
#######################################################################
class Panda():

    def __init__(self, basePosition=[0, 0, 0]):
        self.urdfRootPath = pybullet_data.getDataPath()
        self.panda = p.loadURDF(os.path.join(self.urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True, basePosition=basePosition)

    # has two modes: joint space control (0) and ee-space control (1)
    # djoint is a 7-dimensional vector of joint velocities
    # dposition is a 3-dimensional vector of end-effector linear velocities
    # dquaternion is a 4-dimensional vector of end-effector quaternion velocities
    def step(self, mode=1, djoint=[0]*7, dposition=[0]*3, dquaternion=[0]*4, grasp_open=True):

        # velocity control
        self._velocity_control(mode=mode, djoint=djoint, dposition=dposition, dquaternion=dquaternion, grasp_open=grasp_open)

        # update robot state measurement
        self._read_state()

    def place_ee(self, ee_position, ee_quaternion=[1,0,0,0]):
        q = self._inverse_kinematics(ee_position, ee_quaternion)
        self._reset_robot(q)

    def place_joint(self, joint_position, open_gripper=True):
        if open_gripper:
            q = list(joint_position) + [0.0, 0.0, 0.05, 0.05]
        else:
            q = list(joint_position) + [0.0, 0.0, 0.0, 0.0]
        self._reset_robot(q)

    def reset(self, q=[0.0, -np.pi/8, 0.0, -5*np.pi/8, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.05, 0.05]):
        self._reset_robot(q)


    def _read_state(self):
        joint_position = [0]*9
        joint_velocity = [0]*9
        joint_torque = [0]*9
        joint_states = p.getJointStates(self.panda, range(9))
        for idx in range(9):
            joint_position[idx] = joint_states[idx][0]
            joint_velocity[idx] = joint_states[idx][1]
            joint_torque[idx] = joint_states[idx][3]
        ee_states = p.getLinkState(self.panda, 11)
        ee_position = list(ee_states[4])
        ee_quaternion = list(ee_states[5])
        gripper_contact = p.getContactPoints(bodyA=self.panda, linkIndexA=10)
        self.state['joint_position'] = np.asarray(joint_position)
        self.state['joint_velocity'] = np.asarray(joint_velocity)
        self.state['joint_torque'] = np.asarray(joint_torque)
        self.state['ee_position'] = np.asarray(ee_position)
        self.state['ee_quaternion'] = np.asarray(ee_quaternion)
        self.state['ee_euler'] = np.asarray(p.getEulerFromQuaternion(ee_quaternion))
        self.state['gripper_contact'] = len(gripper_contact) > 0

    def _read_jacobian(self):
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.panda, 11, [0, 0, 0], list(self.state['joint_position']), [0]*9, [0]*9)
        linear_jacobian = np.asarray(linear_jacobian)[:, :7]
        angular_jacobian = np.asarray(angular_jacobian)[:, :7]
        full_jacobian = np.zeros((6, 7))
        full_jacobian[0:3, :] = linear_jacobian
        full_jacobian[3:6, :] = angular_jacobian
        self.jacobian['full_jacobian'] = full_jacobian
        self.jacobian['linear_jacobian'] = linear_jacobian
        self.jacobian['angular_jacobian'] = angular_jacobian

    def _reset_robot(self, joint_position):
        self.state = {}
        self.desired = {}
        for idx in range(len(joint_position)):
            p.resetJointState(self.panda, idx, joint_position[idx])
        self._read_state()
        self.desired['joint_position'] = self.state['joint_position']
        self.desired['ee_position'] = self.state['ee_position']
        self.desired['ee_quaternion'] = self.state['ee_quaternion']

    def _inverse_kinematics(self, ee_position, ee_quaternion):
        return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion))

    def _velocity_control(self, mode, djoint, dposition, dquaternion, grasp_open):
        if mode:
            self.desired['ee_position'] += np.asarray(dposition) / 240.0
            if self.desired['ee_position'][2] < 0.05:
                self.desired['ee_position'][2] = 0.05
            self.desired['ee_quaternion'] += np.asarray(dquaternion) / 240.0
            q_dot = self._inverse_kinematics(self.desired['ee_position'], self.desired['ee_quaternion']) - self.state['joint_position']
        else:
            self.desired['joint_position'] += np.asarray(list(djoint)+[0, 0]) / 240.0
            q_dot = self.desired['joint_position'] - self.state['joint_position']
        gripper_position = [0.0, 0.0]
        if grasp_open:
            gripper_position = [0.05, 0.05]
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))
        p.setJointMotorControlArray(self.panda, [9, 10], p.POSITION_CONTROL, targetPositions=gripper_position)