"""Fixed-view diagnostic frames from the actual portable ONNX control loop."""
from pathlib import Path


class PortableVideo:
    def __init__(self, env, output, bundle, family, level):
        import numpy as np
        import imageio.v2 as iio
        from PIL import ImageFont
        self.env=env;self.path=Path(output)/'diagnostic_single.mp4'
        self.writer=iio.get_writer(self.path,fps=25,codec='libx264',quality=8,macro_block_size=1,
                                  ffmpeg_params=['-threads','1','-pix_fmt','yuv420p'])
        self.font=ImageFont.load_default(size=21);self.small=ImageFont.load_default(size=17)
        self.title=Path(bundle).name.replace('stones_dual_','').replace('_20261025','')
        self.family=family;self.level=level;self.frames=0;self.drift=0.;self.closed=False
        camera=env.scene['review_camera'];self.camera=camera;self.ci=camera.camera_idx
        if int(env.sim.mj_model.cam_bodyid[self.ci])!=0:raise ValueError('Observer must be world-fixed')
        self.pose=(env.sim.data.cam_xpos[0,self.ci].clone(),env.sim.data.cam_xmat[0,self.ci].clone())
        robot=env.scene['robot']
        self.initial={k:getattr(robot.data,k)[0].cpu().tolist() for k in
            ('root_link_pos_w','root_link_quat_w','root_link_lin_vel_w','root_link_ang_vel_w','joint_pos','joint_vel')}

    def frame(self, rgb, elapsed, command, status='RUNNING'):
        import numpy as np
        from PIL import Image,ImageDraw
        self.drift=max(self.drift,float((self.env.sim.data.cam_xpos[0,self.ci]-self.pose[0]).abs().max()),
                       float((self.env.sim.data.cam_xmat[0,self.ci]-self.pose[1]).abs().max()))
        if self.drift>1e-7:raise RuntimeError('Recording observer moved')
        picture=Image.new('RGB',(960,816),(18,23,32))
        picture.paste(Image.fromarray(self.camera.data.rgb[0].cpu().numpy()).resize((960,540)),(0,40))
        picture.paste(Image.fromarray(rgb).resize((336,189)),(12,612))
        draw=ImageDraw.Draw(picture);draw.text((12,9),self.title,font=self.font,fill='white')
        draw.text((12,588),'Actual RGB last consumed by policy',font=self.small,fill='white')
        draw.text((368,612),status[:48],font=self.font,fill=(92,228,140) if status=='PASS + STOP' else (255,160,120) if status!='RUNNING' else 'white')
        draw.text((368,649),f'W command: {command:.2f} m/s | {elapsed:.2f} s',font=self.small,fill='white')
        draw.text((368,681),f'{self.family} | level {self.level}',font=self.small,fill='white')
        draw.text((368,713),'Raw RGB -> ONNX | stationary start',font=self.small,fill='white')
        draw.text((368,745),'30 Hz camera | existing actuator guards',font=self.small,fill='white')
        draw.text((368,777),'Simulation demonstration; hardware unvalidated',font=self.small,fill=(180,190,200))
        return np.asarray(picture)

    def append(self, rgb, elapsed, command):
        self.writer.append_data(self.frame(rgb,elapsed,command));self.frames+=1

    def finish(self, report, rgb, command):
        from .outcome import terminal_status
        status=terminal_status(report)
        final=self.frame(rgb,report['simulated_seconds'],command,status)
        for _ in range(25):self.writer.append_data(final);self.frames+=1
        self.close()
        return dict(path=str(self.path.resolve()),fps=25,frames=self.frames,
            actual_rgb_panel=True,camera_parent='world',camera_position=self.pose[0].cpu().tolist(),
            camera_rotation=self.pose[1].cpu().tolist(),camera_max_pose_drift=self.drift,
            initial_state=self.initial,temporary_single_for_combination=True,terminal_status=status)

    def close(self):
        if not self.closed:self.writer.close();self.closed=True
