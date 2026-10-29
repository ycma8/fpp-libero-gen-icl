"""Restore recorded fixture and camera poses alongside flattened simulator state.

Fixture body transforms live in MjModel, not qpos/qvel. Keep the installed robot
model and its Python geometry references; replay exports may omit robot geoms.
"""
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def recorded_poses(xml, names, tag='body'):
    if isinstance(xml, bytes):
        xml=xml.decode()
    root=ET.fromstring(xml)
    elements={e.get('name'):e for e in root.iter(tag)}
    poses={}
    for name in names:
        if name not in elements:
            raise ValueError(f'Missing recorded {tag}: {name}')
        e=elements[name]
        pos=np.fromstring(e.get('pos','0 0 0'),sep=' ')
        quat=np.fromstring(e.get('quat','1 0 0 0'),sep=' ')
        if pos.shape!=(3,) or quat.shape!=(4,):
            raise ValueError('Source pose is not a position/quaternion pair')
        poses[name]=(pos,quat,e.get('fovy'))
    return poses


def reset_source(env, hdf5, episode, initial_state, libero_root=None):
    with h5py.File(hdf5,'r') as handle:
        xml=handle['data'][episode].attrs['model_file']
    model=env.sim.model
    fixtures=[obj.root_body for obj in env.env.fixtures_dict.values()]
    for name,(pos,quat,_) in recorded_poses(xml,fixtures).items():
        index=model.body_name2id(name)
        model.body_pos[index]=pos
        model.body_quat[index]=quat
    for name,(pos,quat,fovy) in recorded_poses(xml,['agentview'],tag='camera').items():
        index=model.camera_name2id(name)
        model.cam_pos[index]=pos
        model.cam_quat[index]=quat
        if fovy is not None:
            model.cam_fovy[index]=float(fovy)
    return env.set_init_state(initial_state)
