"""G1 wireless button intents, using SDK utils/joystick.py bit assignments."""
import time

R1,L1,START,SELECT = 1,2,4,8
A,B,Y,UP,RIGHT,DOWN,LEFT = 1<<8,1<<9,1<<11,1<<12,1<<13,1<<14,1<<15
DIRECTIONS = UP|RIGHT|DOWN|LEFT
BUTTON_NAMES = ('R1','L1','START','SELECT','R2','L2','F1','F2',
                'A','B','X','Y','UP','RIGHT','DOWN','LEFT')


def button_names(mask):
    return [name for bit,name in enumerate(BUTTON_NAMES) if mask&(1<<bit)]


class RemoteIntent:
    def __init__(self,speed=.15,backward_speed=.1,turn_rate=.2):
        if not 0<speed<=.7 or not 0<backward_speed<=.3 or not 0<turn_rate<=.5:
            raise ValueError('Invalid remote speed limits')
        self.speed,self.backward_speed,self.turn_rate = speed,backward_speed,turn_rate
        self.armed = False
        self.pose_ready = False
        self.policy_enabled = False
        self.policy_neutral_seen = False
        self.policy_since = None
        self.fault = None
        self.yaw_rate = 0.
        self.quit = False
        self.neutral_seen = False
        self.block_motion = True
        self.arm_since = self.exit_since = None

    def set_pose_ready(self):
        # No queued start: require a fresh release and A hold after transition.
        if not self.pose_ready:
            self.pose_ready = True
            self.policy_neutral_seen = False
            self.policy_since = None

    def update(self,buttons,now):
        self.yaw_rate = 0.
        if buttons&SELECT:
            self.fault = self.fault or 'remote_emergency_stop'
        if self.fault:
            self.armed = False
            self.policy_enabled = False
            self.exit_since = (self.exit_since if self.exit_since is not None else now) if buttons&Y else None
            self.quit = self.exit_since is not None and now-self.exit_since>=2.
            return 0.
        if buttons==0:
            self.neutral_seen = True
            self.block_motion = False
            if self.pose_ready:
                self.policy_neutral_seen = True
        arm_combo = buttons==(R1|START)
        if not self.armed and self.neutral_seen and arm_combo:
            self.arm_since = self.arm_since if self.arm_since is not None else now
            if now-self.arm_since>=1.:
                self.armed = True
                self.block_motion = True  # release initialization buttons first
        else:
            self.arm_since = None
        if (self.armed and self.pose_ready and self.policy_neutral_seen
                and not self.policy_enabled and buttons==A):
            self.policy_since = self.policy_since if self.policy_since is not None else now
            if now-self.policy_since>=1.:
                self.policy_enabled = True
                self.block_motion = True
        else:
            self.policy_since = None
        if buttons&B:
            self.block_motion = True  # direction must be released before restart
        if not self.policy_enabled or self.block_motion or buttons&(R1|L1|START|A|B|Y):
            return 0.
        forward = bool(buttons&UP)-bool(buttons&DOWN)
        turn = bool(buttons&LEFT)-bool(buttons&RIGHT)
        self.yaw_rate = turn*self.turn_rate
        return self.speed if forward>0 else -self.backward_speed if forward<0 else 0.


class RemoteController:
    """Headless input; SDK callback independently latches SELECT."""
    def __init__(self,backend,speed,backward_speed,turn_rate):
        self.backend = backend
        self.intent = RemoteIntent(speed,backward_speed,turn_rate)
        self.quit = False

    def poll(self,status):
        if self.backend.fault:
            self.intent.fault = self.intent.fault or self.backend.fault
        speed = self.intent.update(self.backend.latest().remote_buttons,time.monotonic())
        if self.intent.fault:
            self.backend.trip(self.intent.fault)
        self.quit = self.intent.quit
        return speed

    def close(self):
        pass
