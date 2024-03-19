import socket
import json
import time
from threading import Thread
from ur10e_controller import Ur10eController
import numpy as np  
import transforms3d as t3d
import requests
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List
from curobo.geom.types import WorldConfig, Cuboid

def get_custom_world_model(table_height=0.02):
    table = Cuboid(
        name="table",
        dims=[4.0,4.0,4.0],
        pose=[0.0, 0.0, -2.0+table_height, 1.0, 0, 0, 0],
        color=[0, 1.0, 0, 1.0],
    )
    return WorldConfig(
        cuboid=[table],
    )

class UnityMes(BaseModel):
    q:List[float]
    isTracking:int
    pos:List[float]
    quat:List[float]
    cmd:int

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

uc = Ur10eController(get_custom_world_model())

app = FastAPI()

@app.post('/unity')
def unity(mes:UnityMes):
    if uc.homing_state:
        return True

    if mes.cmd==3:
        uc.robot_go_home()
        return True

    pos_from_unity = unity2zup_right_frame(np.array(mes.pos+mes.quat))
    uc.get_q_from_ros()

    if uc.homing_state:
        print("still in homing state")
        uc.tracking_state=False
    else:
        if mes.cmd==2:
            if not uc.tracking_state:
                print("robot start tracking")
                uc.tracking_state=True
                uc.set_start_tcp(pos_from_unity)
            
            
        if mes.cmd==-2:
            if uc.tracking_state:
                print("robot stop tracking")
                uc.tracking_state=False
                




    if not uc.homing_state:
        target = uc.get_relative_target(pos_from_unity)
        dis = np.linalg.norm(target[:3]-uc.get_current_tcp()[:3])
        #print(dis)
        if dis>0.1:
            if uc.tracking_state:
                print("robot lost sync")
            uc.tracking_state=False
        if not uc.tracking_state:
            target =uc.get_current_tcp()
            #print(target)
        else:
            uc.move_hand(mes.q)

        uc.mpc_excute(target)
    return {'status':'ok'}

if __name__ == "__main__":
    uvicorn.run(app,host="10.53.21.90", port=8000)
