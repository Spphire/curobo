import yourdfpy
from curobo.util_file import get_assets_path, join_path, load_yaml, get_robot_configs_path
import json
import numpy as np
import jsbeautifier
opts = jsbeautifier.default_options()
opts.indent_size = 2

robot_cfg = load_yaml(join_path(get_robot_configs_path(), "dual_flexiv.yml"))["robot_cfg"]

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
    base_transform = {
        "joint0":[[1.,0.,0.,0.],
                [0.,1.,0.,0.313],
                [0.,0.,1.,0.],
                [0.,0.,0.,1.]],
        "joint0_1":[[1.,0.,0.,0.],
                [0.,1.,0.,-0.313],
                [0.,0.,1.,0.],
                [0.,0.,0.,1.]]
    }
    with open("./base_transform.txt","w+") as f:
        f.write(jsbeautifier.beautify(json.dumps(base_transform),opts))

save_base_transform()
print(read_base_transform(robot_cfg)["kinematics"]["urdf_path"])