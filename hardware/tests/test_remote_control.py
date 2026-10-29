import numpy as np
import pytest

from vibe_grail_joint.deploy.remote import RemoteIntent,R1,START,SELECT,A,B,Y,UP,DOWN,LEFT,RIGHT
from test_deploy_transport import sdk


def test_sdk_button_bytes_reach_readonly_inspection(sdk):
    from vibe_grail_joint.deploy.unitree import UnitreeBackend
    from vibe_grail_joint.deploy.remote import button_names
    calls,clock,contract,state_message=sdk
    backend=UnitreeBackend('fake0',contract,execute=False)
    try:
        for tick,mask in enumerate((R1|START,A,UP,DOWN,LEFT,RIGHT,B,SELECT,Y),1):
            message=state_message(tick)
            message.wireless_remote=bytes(2)+mask.to_bytes(2,'little')+bytes(36)
            backend._state(message)
            assert backend.latest().remote_buttons==mask
            assert button_names(mask)
        assert backend.fault=='remote_emergency_stop'
        assert calls.publishers==[] and backend.thread is None
    finally:
        backend.close()


def ready():
    intent=RemoteIntent(.25,.1,.2)
    intent.update(0,0.)
    intent.update(R1|START,.1)
    intent.update(R1|START,1.11)
    assert intent.armed
    intent.set_pose_ready()
    intent.update(0,1.2)
    intent.update(A,1.3)
    intent.update(A,2.31)
    assert intent.policy_enabled
    intent.update(0,2.4)
    return intent


def test_initialization_requires_neutral_then_deliberate_hold():
    intent=RemoteIntent()
    intent.update(R1|START,0.)
    intent.update(R1|START,5.)
    assert not intent.armed
    intent.update(0,6.)
    intent.update(R1|START|UP,7.)
    intent.update(R1|START|UP,9.)
    assert not intent.armed
    intent.update(R1|START,10.)
    intent.update(R1|START,10.5)
    assert not intent.armed
    intent.update(R1|START,11.1)
    assert intent.armed and intent.update(UP,11.2)==0.
    intent.update(0,11.3)
    assert intent.update(UP,11.4)==0.
    assert not intent.policy_enabled


def test_policy_requires_completed_pose_fresh_neutral_and_exclusive_a_hold():
    intent=RemoteIntent()
    intent.update(0,0.)
    intent.update(R1|START,.1)
    intent.update(R1|START,1.2)
    intent.update(0,1.3)
    intent.update(A,2.)
    intent.update(A,4.)
    assert not intent.policy_enabled
    intent.set_pose_ready()
    intent.update(A,5.)
    intent.update(A,7.)
    assert not intent.policy_enabled  # held during ramp cannot queue start
    intent.update(0,8.)
    intent.update(A|UP,8.1)
    intent.update(A|UP,10.)
    assert not intent.policy_enabled
    intent.update(A,11.)
    intent.update(A,11.9)
    assert not intent.policy_enabled
    intent.update(A,12.1)
    assert intent.policy_enabled
    assert intent.update(UP,12.2)==0.  # release A/all directions first
    intent.update(0,12.3)
    assert intent.update(UP,12.4)==.15


def test_directions_turns_release_and_conflicts():
    intent=ready()
    assert intent.update(UP,2.)==.25
    assert intent.update(DOWN,2.1)==-.1
    assert intent.update(LEFT,2.2)==0. and intent.yaw_rate==.2
    assert intent.update(RIGHT,2.3)==0. and intent.yaw_rate==-.2
    assert intent.update(UP|DOWN|LEFT|RIGHT,2.4)==0. and intent.yaw_rate==0.
    assert intent.update(0,2.5)==0. and intent.yaw_rate==0.


def test_normal_stop_requires_direction_release_before_resuming():
    intent=ready()
    assert intent.update(UP,2.)==.25
    assert intent.update(B|UP,3.)==0.
    assert intent.update(UP,4.)==0.
    intent.update(0,5.)
    assert intent.update(UP,6.)==.25


def test_emergency_latches_and_exit_requires_hold_after_stop():
    intent=ready()
    intent.update(Y,2.)
    intent.update(Y,5.)
    assert not intent.quit
    assert intent.update(SELECT|UP,6.)==0. and not intent.armed and not intent.policy_enabled
    for buttons in (0,R1|START,A,UP,DOWN):
        assert intent.update(buttons,7.)==0.
    intent.update(Y,8.)
    intent.update(Y,10.1)
    assert intent.quit and intent.fault=='remote_emergency_stop'


def test_deployed_observations_match_simulation_for_backward_and_turns():
    import json
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from vibe_grail_joint.deploy.observations import ObservationBuilder,RobotState,Reference
    import pytest
    SimBuilder=pytest.importorskip('sim2sim.vendor.observations').ObservationBuilder
    contract=json.loads((Path(__file__).resolve().parents[2]/'models/velocity3999/bundle.json').read_text())['robot']
    state=RobotState(np.array(contract['default_joint_pos']),np.zeros(29),np.array([1.,0,0,0]),np.zeros(3),0.)
    quat=Rotation.from_euler('z',.3).as_quat()[[3,0,1,2]]
    reference=Reference(np.tile(state.q,(10,1)),np.zeros((10,29)),np.tile(quat,(10,1)),np.zeros(3),np.zeros(3))
    deployed,sim=ObservationBuilder(contract),SimBuilder(contract)
    for speed,heading,rate in ((-.1,0.,0.),(0.,.3,.2),(.25,-.4,-.2)):
        arguments=(state,reference,np.zeros((21,192)),np.zeros(192),speed)
        a=deployed.build(*arguments,heading=heading,yaw_rate=rate)
        b=sim.build(*arguments,heading=heading,yaw_rate=rate)
        for name in a:
            np.testing.assert_array_equal(a[name],b[name])


def test_signed_planner_speed_is_magnitude_and_keeps_facing():
    from types import SimpleNamespace
    from vibe_grail_joint.deploy.reference import OnlineReference
    class Session:
        def get_outputs(self):
            return [SimpleNamespace(name=n) for n in ('num_pred_frames','mujoco_qpos')]
        def run(self,_,inputs):
            self.inputs=inputs
            poses=np.zeros((1,44,36),np.float32)
            poses[:,:,3]=1.
            return [np.array([44]),poses]
    session=Session()
    contract=dict(root_height=.78,default_joint_pos=[0.]*29)
    for speed,heading in ((-.1,.4),(.25,-.4),(0.,.7)):
        ref=OnlineReference('unused',contract,session_override=session)
        ref.replan(0.,speed,heading)
        data=session.inputs
        facing=np.array([[np.cos(heading),np.sin(heading),0.]],np.float32)
        np.testing.assert_array_equal(data['facing_direction'],facing)
        np.testing.assert_array_equal(data['movement_direction'],facing*(-1 if speed<0 else 1))
        np.testing.assert_allclose(data['target_vel'],[abs(speed) if speed else -1.])
        assert data['mode'][0]==(2 if speed else 0)


@pytest.mark.parametrize('start_policy',[False,True])
def test_complete_remote_control_and_stop_use_only_fake_sdk(sdk,monkeypatch,tmp_path,start_policy):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import pytest
    from vibe_grail_joint.deploy import run,unitree,inference_process,remote
    calls,clock,_,state_message=sdk
    root=Path(__file__).resolve().parents[2]
    contract=json.loads((root/'models/velocity3999/bundle.json').read_text())['robot']
    default=np.asarray(contract['default_joint_pos'],np.float32)
    def buttons():
        t=clock[0]-100.
        for until,keys in ((.2,0),(1.4,R1|START),(4.7,A),(5.,UP),(5.3,0),
                           (6.5,A if start_policy else 0),(6.7,0),(7.,UP),(7.3,B),
                           (7.5,0),(7.9,DOWN),(8.3,LEFT),(8.7,RIGHT),(8.8,SELECT)):
            if t<until:
                return keys
        return Y
    class Backend(unitree.UnitreeBackend):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.refresh()
        def refresh(self):
            msg=state_message(1 if self.last_tick is None else self.last_tick+1)
            for m,q in zip(msg.motor_state,default): m.q=float(q)
            msg.wireless_remote=bytes(2)+buttons().to_bytes(2,'little')+bytes(36)
            self._state(msg)
        def latest(self):
            self.refresh()
            if self.armed:
                self._write_once()
            return super().latest()
    computed=[]
    class Inference:
        def __init__(self,config):
            assert config['calibration']==(None if start_policy else 'fake.json')
            self.metadata=dict(motor_transport=False)
        def step(self,state,speed,**kwargs):
            result=dict(image_at=clock[0],frame_id=1,source_state_at=state.received_at,
                        planner_s=0.,worker_s=0.)
            if kwargs.get('compute',True):
                computed.append((clock[0],speed,kwargs))
                result.update(action=np.zeros(29,np.float32),target=default.copy(),policy_s=0.)
            return result
        def close(self): pass
    def sleep(dt): clock[0]+=max(dt,.02)
    fake_time=SimpleNamespace(monotonic=lambda:clock[0],sleep=sleep)
    monkeypatch.setattr(run,'time',fake_time)
    monkeypatch.setattr(remote,'time',fake_time)
    monkeypatch.setattr(unitree,'UnitreeBackend',Backend)
    monkeypatch.setattr(inference_process,'InferenceProcess',Inference)
    log=tmp_path/'remote.jsonl'
    mount_option=['--allow-unmeasured-camera'] if start_policy else ['--calibration','fake.json']
    args=run.parse_args(['--bundle',str(root/'models/velocity3999'),'--interface','fake0',
        '--mode','control','--execute','--input','remote',
        '--seconds','12','--log',str(log)]+mount_option)
    monkeypatch.setattr(run,'parse_args',lambda:args)
    with pytest.raises(RuntimeError,match='remote_emergency_stop'):
        run.main()
    rows=[json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['camera_mount_unmeasured_override'] is start_policy
    assert rows[0]['camera_calibration_file']==(None if start_policy else 'fake.json')
    ticks=[r for r in rows if r['event']=='control_tick' and r['lowcmd_enabled']]
    if start_policy:
        assert max(r['speed_command'] for r in ticks)>0
        assert min(r['speed_command'] for r in ticks)<0
        assert max(r['yaw_rate_command'] for r in ticks)>0
        assert min(r['yaw_rate_command'] for r in ticks)<0
        assert computed[0][0]>=106.3 and computed[0][1]==0.
        assert computed[0][2]['reset'] is True
        assert sum(r['event']=='policy_started' for r in rows)==1
    else:
        assert not computed and not ticks
        assert all(r['event']!='policy_started' for r in rows)
    holds=[r for r in rows if r['event']=='nominal_pose_hold']
    assert len(holds)>90  # no automatic policy after three-second ramp
    for row in holds:
        np.testing.assert_allclose(row['q_target'],default)
        assert row['policy_enabled'] is False
    assert {'nominal_pose_ready','damping_hold','damping_hold_exit'}<={r['event'] for r in rows}
    assert len(calls.publishers)==1 and calls.releases==0
    assert all(m.kp==0 for m in calls.publishers[0].messages[-1].motor_cmd[:29])
