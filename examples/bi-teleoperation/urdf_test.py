import yourdfpy
from curobo.util_file import get_assets_path, join_path, load_yaml, get_robot_configs_path
import json
import numpy as np
import jsbeautifier
import transforms3d as t3d

def read_base_transform(robot_cfg, path="./base_transform.txt"):
    f = open(path,"r")
    _ = json.loads(f.read())
    f.close()

    urdf_path = join_path(get_assets_path(),robot_cfg["kinematics"]["urdf_path"])
    robot = yourdfpy.URDF.load(urdf_path)

    for joint_name in _.keys():
        if joint_name in robot.joint_map.keys():
            joint = robot.joint_map[joint_name]
            if joint.parent == "world" and joint.type=="fixed":
                joint.origin = np.array(_[joint_name])
                print(joint)

    new_urdf_path = urdf_path.replace('.urdf','_temp.urdf')
    yourdfpy.URDF.write_xml_file(robot,new_urdf_path)
    robot_cfg["kinematics"]["urdf_path"] = new_urdf_path.replace(get_assets_path()+'/','')
    return robot_cfg


def save_base_transform():
    YR = t3d.euler.euler2mat(0,np.pi/4,0)

    T = np.eye(4)
    T[:3,:3] =t3d.euler.euler2mat(-np.pi/4,0,0) @ YR 
    T[1,3]=0.114
    T[2,3]=0.15

    _T = np.eye(4)
    _T[:3,:3] = t3d.euler.euler2mat(np.pi/4,0,0) @YR 
    _T[1,3]=-0.114
    _T[2,3]=0.15

    base_transform = {
        "joint0":T.tolist(),
        "joint0_1":_T.tolist(),
    }
    with open("./base_transform.txt","w+") as f:
        f.write(jsbeautifier.beautify(json.dumps(base_transform),opts))


if __name__ == "__main__":
    opts = jsbeautifier.default_options()
    opts.indent_size = 2

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "dual_flexiv.yml"))["robot_cfg"]
    save_base_transform()
    print(read_base_transform(robot_cfg)["kinematics"]["urdf_path"])
    print(np.array([-20,-20,35,38,10,10,-56])*np.pi/180)
    print(np.array([40,-20,-35,38,-20,0,-100])*np.pi/180)
    print((np.array([0,-10,0,90,0,40,0,0,-5,0,90,0,40,0])*np.pi/180).tolist())