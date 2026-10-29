import numpy as np
import pytest
from fpp_libero_gen.simulation import recorded_poses


def test_recorded_fixed_body_pose_is_independent_of_flat_state():
    xml='''<mujoco><worldbody><body name="cabinet" pos="0.03 -0.27 0.02" quat="0.2 0 0 0.98"/>
      <body name="identity"/><camera name="agentview" pos="1 2 3" fovy="45"/>
    </worldbody></mujoco>'''
    poses=recorded_poses(xml,['cabinet','identity'])
    np.testing.assert_allclose(poses['cabinet'][0],[.03,-.27,.02])
    np.testing.assert_allclose(poses['cabinet'][1],[.2,0,0,.98])
    np.testing.assert_array_equal(poses['identity'][1],[1,0,0,0])
    assert recorded_poses(xml,['agentview'],tag='camera')['agentview'][2]=='45'
    with pytest.raises(ValueError):
        recorded_poses(xml,['missing'])
