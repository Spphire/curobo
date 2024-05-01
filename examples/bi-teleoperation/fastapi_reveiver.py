from threading import Thread
import numpy as np  
import transforms3d as t3d
import requests
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List
from curobo.geom.types import WorldConfig, Cuboid
from bi_flexiv_controller import BiFlexivController


class HandMes(BaseModel):
    q: List[float]

    pos: List[float]
    quat: List[float]

    thumbTip: List[float]
    indexTip: List[float]
    middleTip: List[float]
    ringTip: List[float]
    pinkyTip: List[float]

    cmd:int

class UnityMes(BaseModel):
    valid:bool
    leftHand:HandMes
    rightHand:HandMes

def unity2zup_right_frame(pos_quat):
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

bi = BiFlexivController()
from queue import Queue
qq = Queue()
app = FastAPI()
@app.post('/unity')
def unity(mes:UnityMes):
    qq.put(mes)
    return {'status':'ok'}

def MainThread():
    while True:
        try:
            while qq.qsize()>1:
                mes: UnityMes = qq.get()
            if not qq.empty():
                mes: UnityMes = qq.get()
            elif not (bi.right_robot.homing_state or bi.left_robot.homing_state):
                bi.mpc_excute(None,None)
                continue
        except Exception as e:
            print(e)


        if bi.right_robot.homing_state or bi.left_robot.homing_state:
            continue
        
        bi.left_robot.gripper.move(0.1-mes.leftHand.squeeze/9, 10, 20)
        bi.right_robot.gripper.move(0.1-mes.rightHand.squeeze/9, 10, 20) 

        if mes.rightHand.cmd==3 or mes.leftHand.cmd==3:
            bi.birobot_go_home()
            continue

        r_pos_from_unity = unity2zup_right_frame(np.array(mes.rightHand.pos+mes.rightHand.quat))
        l_pos_from_unity = unity2zup_right_frame(np.array(mes.leftHand.pos+mes.leftHand.quat))


        if bi.left_robot.homing_state:
                print("left  still in homing state")
                bi.left_robot.tracking_state=False
        else:
            if mes.leftHand.cmd==2:
                if bi.left_robot.tracking_state:
                    print("left robot stop tracking")
                    bi.left_robot.tracking_state=False
                else:
                    print("left robot start tracking")
                    bi.left_robot.set_start_tcp(l_pos_from_unity)

        if bi.right_robot.homing_state:
            bi.right_robot.tracking_state=False
        else:
            if mes.rightHand.cmd==2:
                if bi.right_robot.tracking_state:
                    print("right robot stop tracking")
                    bi.right_robot.tracking_state=False
                else:
                    print("right robot start tracking")
                    bi.right_robot.set_start_tcp(r_pos_from_unity)

        if not bi.left_robot.homing_state and not bi.right_robot.homing_state:
            # 手移动过快或是定位抽搐自动断开跟随状态
            threshold = 0.5
            left_target = bi.left_robot.get_relative_target(l_pos_from_unity)
            if np.linalg.norm(left_target[:3]-bi.left_robot.get_current_tcp()[:3])>threshold:
                if bi.left_robot.tracking_state:
                    print("left robot lost sync")
                bi.left_robot.tracking_state=False
            if not bi.left_robot.tracking_state:
                left_target =bi.left_robot.get_current_tcp()

            right_target = bi.right_robot.get_relative_target(r_pos_from_unity)
            if np.linalg.norm(right_target[:3]-bi.right_robot.get_current_tcp()[:3])>threshold:
                if bi.right_robot.tracking_state:
                    print("right robot lost sync")
                bi.right_robot.tracking_state=False
            if not bi.right_robot.tracking_state:
                right_target =bi.right_robot.get_current_tcp()

            bi.mpc_excute(left_target,right_target)
        


if __name__ == "__main__":
    import threading
    threading.Thread(target=MainThread,daemon=True).start()
    print("===========================================")
    print("start teleoperation")
    uvicorn.run(app,host="192.168.2.187", port=8082, log_level="critical")