
import socket
import json
import time
from threading import Thread
from bi_flexiv_controller import BiFlexivController
import numpy as np  
import transforms3d as t3d

# unity左手坐标系z东y上转右手坐标系x东z上
def unity2zup_right_frame(pos_quat: np.ndarray):
        pos_quat*=np.array([1,-1,1,1,-1,1,-1])
        rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
        pos_vec = pos_quat[:3]
        T=np.eye(4)
        T[:3,:3]= rot_mat
        T[:3,3]=pos_vec
        fit_mat = t3d.euler.axangle2mat([0,1,0],np.pi/2)
        fit_mat = fit_mat@t3d.euler.axangle2mat([0,0,1],-np.pi/2)
        target_rot_mat=fit_mat@rot_mat
        target_pos_vec=fit_mat@pos_vec
        target = np.array(target_pos_vec.tolist()+t3d.quaternions.mat2quat(target_rot_mat).tolist())
        return target

class Receiver(Thread):
    def __init__(self, 
                 controller:BiFlexivController, 
                 local_ip= "192.168.2.187", # 机械臂那台机子ip
                 port=8082):
        self.address = (local_ip, port)
        self.socket_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_obj.setblocking(0)
        self.controller = controller
        self.isOn=True

    def run(self):
        self.socket_obj.bind(self.address)
        print("start receiving...")
        while self.isOn:
            self.receive()
            time.sleep(0.01)

    def receive(self):
        try:
            data, _ = self.socket_obj.recvfrom(4096)
            if len(data)==0 and not (self.controller.right_robot.homing_state or self.controller.left_robot.homing_state):
                self.controller.mpc_excute(None,None)
                return
            s=json.loads(data)

            if self.controller.left_robot.homing_state or self.controller.right_robot.homing_state:
                return True

            # 夹爪部分
            self.controller.left_robot.gripper.move(0.1-s["leftHand"]["squeeze"]/9, 10, 10)
            self.controller.right_robot.gripper.move(0.1-s["rightHand"]["squeeze"]/9,
                                                     10, # API中显示似乎是速度但修改没有效果
                                                     10)

            if s["leftHand"]["cmd"]==3 or s["rightHand"]["cmd"]==3:
                self.controller.birobot_go_home()
                return True

            r_pos_from_unity = unity2zup_right_frame(np.array(s["rightHand"]["pos"]+s["rightHand"]["quat"]))
            l_pos_from_unity = unity2zup_right_frame(np.array(s["leftHand"]["pos"]+s["leftHand"]["quat"]))


            if self.controller.left_robot.homing_state:
                print("left  still in homing state")
                self.controller.left_robot.tracking_state=False
            else:
                if s["leftHand"]["cmd"]==2:
                    if self.controller.left_robot.tracking_state:
                        print("left robot stop tracking")
                        self.controller.left_robot.tracking_state=False
                    else:
                        print("left robot start tracking")
                        self.controller.left_robot.set_start_tcp(l_pos_from_unity)

            if self.controller.right_robot.homing_state:
                self.controller.right_robot.tracking_state=False
            else:
                if s["rightHand"]["cmd"]==2:
                    if self.controller.right_robot.tracking_state:
                        print("right robot stop tracking")
                        self.controller.right_robot.tracking_state=False
                    else:
                        print("right robot start tracking")
                        self.controller.right_robot.set_start_tcp(r_pos_from_unity)



            if not self.controller.left_robot.homing_state and not self.controller.right_robot.homing_state:
                # 手移动过快或是定位抽搐自动断开跟随状态
                threshold = 0.5
                left_target = self.controller.left_robot.get_relative_target(l_pos_from_unity)
                if np.linalg.norm(left_target[:3]-self.controller.left_robot.get_current_tcp()[:3])>threshold:
                    if self.controller.left_robot.tracking_state:
                        print("left robot lost sync")
                    self.controller.left_robot.tracking_state=False
                if not self.controller.left_robot.tracking_state:
                    left_target =self.controller.left_robot.get_current_tcp()

                right_target = self.controller.right_robot.get_relative_target(r_pos_from_unity)
                if np.linalg.norm(right_target[:3]-self.controller.right_robot.get_current_tcp()[:3])>threshold:
                    if self.controller.right_robot.tracking_state:
                        print("right robot lost sync")
                    self.controller.right_robot.tracking_state=False
                if not self.controller.right_robot.tracking_state:
                    right_target =self.controller.right_robot.get_current_tcp()

                self.controller.mpc_excute(left_target,right_target)
            return True
        except:
            #print("error in udp")
            return False
        
if __name__ == "__main__":
    bi = BiFlexivController()
    r = Receiver(controller=bi)
    r.run()