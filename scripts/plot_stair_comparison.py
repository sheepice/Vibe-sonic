"""Create static scientific plots with the system Python matplotlib installation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'records/g1_sim_compare_20261029'
real=np.load(OUT/'real_telemetry.npz')
summary=json.loads((OUT/'analysis.json').read_text())
datasets=[('Real robot (recorded)',real,real,summary['positive_command_start_s'])]
for name,title in [('sim_15cm_30cm_04','Simulation: continue to top'),('sim_stop_first_steps','Simulation: stop on stairs')]:
    datasets.append((title,np.load(OUT/name/'telemetry.npz'),np.load(OUT/name/'analysis.npz'),2.))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(4,3,figsize=(15,11),sharex='col',constrained_layout=True)
for col,(title,data,analysis,origin) in enumerate(datasets):
    t=data['time']-origin;command=analysis['command'];q=data['q'];feet=analysis['feet']
    axs[0,col].set_title(title)
    axs[0,col].plot(t,command,color='#146989');axs[0,col].set_ylim(-.02,.46)
    axs[1,col].plot(t,q[:,3],label='Left knee',color='#1578a2')
    axs[1,col].plot(t,q[:,9],label='Right knee',color='#d06127');axs[1,col].set_ylim(-.1,2.5)
    axs[2,col].plot(t,100*(feet[:,1,2]-feet[:,0,2]),color='#4d8194')
    axs[2,col].axhline(15,color='#d06127',ls='--',lw=1,label='One riser: 15 cm')
    axs[2,col].axhline(0,color='grey',lw=.6);axs[2,col].set_ylim(-38,38)
    rpy=np.rad2deg(analysis['rpy'])
    axs[3,col].plot(t,rpy[:,1],color='#6d528a',label='Pelvis pitch')
    axs[3,col].plot(t,rpy[:,2]-rpy[0,2],color='#2b8c65',label='Relative yaw')
    axs[3,col].set_ylim(-22,22);axs[3,col].set_xlabel('Time from forward command onset (s)')
    for row in range(4):
        ax=axs[row,col];ax.axvline(0,color='#555',lw=.7,ls=':');ax.grid(alpha=.2)
        positive=t[command>1e-6]
        ax.axvspan(positive[0],positive[-1],color='#409bb0',alpha=.08)
        ax.set_xlim(max(-3,t[0]),t[-1])
    axs[1,col].legend(fontsize=8);axs[2,col].legend(fontsize=8);axs[3,col].legend(fontsize=8)
for ax,label in zip(axs[:,0],['Forward command (m/s)','Knee angle (rad)','Right - left ankle height (cm)','Pelvis orientation (deg)']):ax.set_ylabel(label)
fig.suptitle('Same policy, 0.4 m/s; simulated riser 15 cm, tread 30 cm\nReal ankle heights reconstructed from q + IMU; no contact/absolute-height measurement',fontsize=13)
fig.savefig(OUT/'motion_comparison.png',dpi=160);fig.savefig(OUT/'motion_comparison.pdf');plt.close(fig)

fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
t=real['time'];error=np.abs(real['target']-real['q'])
ax[0].plot(t,real['image_age']*1000,lw=.6,color='#1578a2');ax[0].set_ylabel('Image age (ms)');ax[0].set_ylim(0,60)
ax[1].plot(t,error[:,4],lw=.6,label='Left ankle pitch',color='#1578a2')
ax[1].plot(t,error[:,10],lw=.6,label='Right ankle pitch',color='#d06127');ax[1].legend()
ax[1].set_ylabel('|target - q| (rad)');ax[1].set_xlabel('Time from real policy start (s)')
for a in ax:
    a.axvspan(summary['positive_command_start_s'],summary['positive_command_end_s'],color='#409bb0',alpha=.1)
    a.grid(alpha=.2)
fig.suptitle('Real trial: images remained fresh; PD target error is not a motor-fault measurement')
fig.savefig(OUT/'real_health_and_tracking.png',dpi=160);plt.close(fig)
print(OUT/'motion_comparison.png')
