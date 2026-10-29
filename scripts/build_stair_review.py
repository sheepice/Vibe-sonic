"""Assemble offline HTML, attention keyframes and a compact real-input movie."""
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'records/g1_sim_compare_20261029'
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
SMALL=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
summary=json.loads((OUT/'analysis.json').read_text())
real_dir=ROOT/'records/g1_recordings/trial_20261028_235905/preview_checked_20261029'

def load(folder,real=False):
    d=json.loads((folder/'preview.json').read_text());frames=[]
    for f in d['frames']:
        if f.get('attention') is None:continue
        f=dict(f)
        f['t']=f['step']['inference_at']-summary['policy_started_monotonic'] if real else f['time']
        f['forward_t']=f['t']-(summary['positive_command_start_s'] if real else 2.)
        for key in ['policy','raw','edges']:
            if f.get(key):f[key]=str((folder/f[key]).relative_to(OUT.parent.parent))
        frames.append(f)
    return frames

real=load(real_dir,True)
normal=load(OUT/'sim_15cm_30cm_04/preview')
stop=load(OUT/'sim_stop_first_steps/preview')

def panel(f,width=280):
    im=Image.open(ROOT/f['policy']).convert('RGB').resize((width,width*64//112),Image.Resampling.NEAREST)
    return im

def overlay(f,query,width=280):
    # Display-only bilinear resampling of the original 4x7 weights.
    weights=np.array(f['attention'][query],np.float32).reshape(4,7)
    peak=max(float(weights.max()),1e-12)
    values=np.asarray(Image.fromarray(weights/peak).resize((224,128),Image.Resampling.BILINEAR))
    stops=np.array([[22,26,90],[22,95,220],[22,211,192],[255,231,60],[240,45,35]])
    heat=np.stack([np.interp(values,np.linspace(0,1,5),stops[:,j]) for j in range(3)],axis=-1)
    base=np.asarray(Image.open(ROOT/f['policy']).convert('RGB').resize((224,128),Image.Resampling.BILINEAR))
    rgb=np.round(base*.45+heat*.55).astype(np.uint8)
    return Image.fromarray(rgb).resize((width,width*64//112),Image.Resampling.BILINEAR)

def movie_frame(f):
    canvas=Image.new('RGB',(1120,230),(20,27,34));d=ImageDraw.Draw(canvas)
    d.text((10,6),f"REAL | policy t={f['t']:.2f}s | forward-relative t={f['forward_t']:.2f}s | command={f['step']['speed']:.2f}m/s",font=FONT,fill='white')
    for i,title in enumerate(['Exact policy image','q_task_cmd attention','q_proprio attention','q_cls attention']):
        canvas.paste(panel(f) if i==0 else overlay(f,i-1),(i*280,48))
        d.text((i*280+6,29),title,font=SMALL,fill='white')
    d.text((10,211),'Smooth display of recorded 4x7 weights; relative color scale. NOT causal pixel saliency.',font=SMALL,fill='#bbccd8')
    return canvas

if not (OUT/'real_policy_attention_smooth.mp4').exists():
    movie=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1120x230','-r','5',
        '-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',str(OUT/'real_policy_attention_smooth.mp4')],stdin=subprocess.PIPE)
    for t in np.arange(0,real[-1]['t'],.2):
        f=min(real,key=lambda f:abs(f['t']-t));movie.stdin.write(movie_frame(f).tobytes())
    movie.stdin.close();assert movie.wait()==0

keyframes=[('Real: approach',min(real,key=lambda f:abs(f['t']-19.8))),
    ('Real: stairs',min(real,key=lambda f:abs(f['t']-22.5))),
    ('Real: stopped',min(real,key=lambda f:abs(f['t']-35.))),
    ('Sim: approach',min(normal,key=lambda f:abs(f['t']-4.))),
    ('Sim: stairs',min(normal,key=lambda f:abs(f['t']-6.4))),
    ('Sim: stopped on stairs',min(stop,key=lambda f:abs(f['t']-18.)))]
canvas=Image.new('RGB',(1120,220*len(keyframes)),(20,27,34));d=ImageDraw.Draw(canvas)
for row,(label,f) in enumerate(keyframes):
    y=row*220;d.text((8,y+4),f"{label} | t={f['t']:.2f}s | v={f['step']['speed']:.2f}m/s",font=FONT,fill='white')
    for col,title in enumerate(['Exact policy input','q_task_cmd','q_proprio','q_cls']):
        d.text((col*280+8,y+26),title,font=SMALL,fill='white')
        canvas.paste(panel(f) if col==0 else overlay(f,col-1),(col*280,y+47))
canvas.save(OUT/'attention_comparison_smooth.jpg',quality=92)

data=[dict(title='真机：已有记录',frames=real,origin=summary['positive_command_start_s'],preview='../g1_recordings/trial_20261028_235905/preview_checked_20261029/index.html'),
      dict(title='仿真：持续前进至楼梯顶端',frames=normal,origin=2,preview='sim_15cm_30cm_04/preview/index.html'),
      dict(title='仿真：在楼梯中途松开前进',frames=stop,origin=2,preview='sim_stop_first_steps/preview/index.html')]
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 真机与上楼仿真对照 · 2026-10-29</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#111921;color:#e6edf4;margin:24px auto;max-width:1600px;padding:0 20px}h1{font-size:27px}a{color:#7bd0ed}p{max-width:1100px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{background:#1b2732;border:1px solid #344651;padding:16px;border-radius:10px}h2{font-size:20px;margin:0 0 12px}canvas{width:100%;height:auto;image-rendering:auto;background:#26333d}input[type=range]{width:100%}button,select{padding:8px;background:#314857;color:white;border:1px solid #607580;border-radius:6px}.meta{min-height:76px;font-size:14px}.small{font-size:13px;color:#b6c7d2}.external{image-rendering:auto;max-height:260px;object-fit:contain;max-width:100%}.links{display:flex;gap:20px;flex-wrap:wrap}.toolbar{margin:24px 0;background:#233440;padding:12px}#opacity{width:180px}.plot{width:100%;background:white}.legend{height:9px;border-radius:4px;margin-top:10px;background:linear-gradient(90deg,rgb(22,26,90),rgb(22,95,220),rgb(22,211,192),rgb(255,231,60),rgb(240,45,35))}.legendLabels{display:flex;justify-content:space-between;font-size:12px;color:#b6c7d2}.heatRange{margin:5px 0 12px;min-height:40px}@media(max-width:1000px){.cards{grid-template-columns:1fr}}</style>
<h1>同一策略：真机记录与 15 cm × 30 cm 上楼仿真</h1>
<p>仿真以 0.4 m/s 指令完成三级上楼。真机停步后双脚踝的重建高度差约 14.6 cm；“楼梯中途停止”仿真也出现约 15 cm 高度差。当前证据更符合停在相差一级高度的支撑面上的姿态，但不能确定停步之前是否先发生了绊脚。真机这一段图像持续更新，最终由遥控急停结束。</p>
<div class="links"><a href="REPORT_ZH.md">完整分析报告</a><a href="sim_15cm_30cm_04/external_side.mp4">完整上楼视频</a><a href="sim_stop_first_steps/external_side.mp4">中途停步视频</a><a href="real_policy_attention_smooth.mp4">真机输入与三路注意力视频</a><a href="attention_comparison_smooth.jpg">注意力关键帧</a><a href="sim_occlusion.jpg">仿真遮挡敏感性</a></div>
<p class="small">全部本地分析，没有连接 G1。各段时间、起步距离、环境外观不同；滑块独立，不表示同一空间位置。真机只有头部视角，仿真附外部视角。热力图是经过空间混合后的 4×7 槽位权重，不能等同于台阶分割或因果贡献。</p>
<div class="toolbar">数据：<select id="query"><option value="mean" selected>三组注意力均值</option><option value="0">任务查询 q_task_cmd</option><option value="1">本体查询 q_proprio</option><option value="2">图像查询 q_cls</option><option value="feature">特征 L2 幅值</option><option value="none">原始图像</option></select>　显示 <select id="heatStyle"><option value="smooth">平滑热区</option><option value="blocks">原始色块</option></select>　色标 <select id="heatScale"><option value="relative">本帧范围</option><option value="absolute">注意力固定 0～1</option></select>　透明度 <input id="opacity" type="range" min="0" max="1" step=".05" value=".55">　<label><input type="checkbox" id="grid">显示网格</label><div class="small">平滑仅改变显示，不增加原始 4×7 数据的分辨率。蓝色低、红色高；特征 L2 按本帧最小/最大值着色。</div></div>
<div class="cards" id="cards"></div>
<h2 style="margin-top:30px">动作与相对足部高度</h2><p class="small">横轴从各自首次前进指令开始计时，蓝色阴影为前进指令非零的区间；足部高度差由关节角、IMU 和同一 G1 模型正运动学推算，不是足底接触测量。</p><a href="motion_comparison.png"><img class="plot" src="motion_comparison.png"></a>
<script>__HEATMAP_JS__</script>
<script>const datasets=__DATA__;const query=document.querySelector('#query'),opacity=document.querySelector('#opacity'),grid=document.querySelector('#grid'),heatStyle=document.querySelector('#heatStyle'),heatScale=document.querySelector('#heatScale');
const states=datasets.map((d,i)=>{const div=document.createElement('div');div.className='card';div.innerHTML=`<h2>${d.title}</h2><canvas width="560" height="320"></canvas><div class="legend"></div><div class="legendLabels"><span>低</span><span>高</span></div><div class="heatRange small"></div><div class="meta"></div><input type="range" min="0" max="${d.frames.length-1}" value="0" step="1"><button>播放</button> <a href="${d.preview}">详细回放</a><p class="small rawlabel"></p><img class="external">`;document.querySelector('#cards').append(div);const s={d,div,slider:div.querySelector('input'),canvas:div.querySelector('canvas'),button:div.querySelector('button'),playing:false,generation:0};s.slider.value=d.frames.findIndex(f=>f.forward_t>=0);s.slider.oninput=()=>{s.playing=false;s.button.textContent='播放';render(s)};s.button.onclick=()=>{s.playing=!s.playing;s.button.textContent=s.playing?'暂停':'播放';if(s.playing)advance(s)};return s});
function render(s){
 const f=s.d.frames[+s.slider.value],generation=++s.generation,im=new Image();
 im.onload=()=>{if(generation!==s.generation)return;const result=VibeHeatmap.draw(s.canvas,im,f,{choice:query.value,style:heatStyle.value,scale:heatScale.value,alpha:+opacity.value,grid:grid.checked});s.div.querySelector('.heatRange').textContent=VibeHeatmap.describe(result);s.canvas.dataset.step=f.step.step_id;};
 im.src='../../'+f.policy;
 s.div.querySelector('.meta').textContent=`策略 t=${f.t.toFixed(2)} s · 前进后 ${f.forward_t.toFixed(2)} s · 指令 ${f.step.speed.toFixed(2)} m/s · 帧 ${f.step.image_frame_id} · ${+s.slider.value+1}/${s.d.frames.length}`;
 const raw=s.div.querySelector('.external');raw.hidden=!f.raw;if(f.raw)raw.src='../../'+f.raw;s.div.querySelector('.rawlabel').textContent=f.raw?(s===states[0]?'同源原始彩色图的压缩预览（非另一相机）':'同一步仿真外部视角'):'此抽样帧没有同源原始彩色预览；策略输入如上。';
}
function advance(s){if(!s.playing)return;render(s);const i=+s.slider.value;if(i>=s.d.frames.length-1){s.playing=false;s.button.textContent='播放';return}const dt=(s.d.frames[i+1].t-s.d.frames[i].t)*1000;setTimeout(()=>{if(!s.playing)return;s.slider.value=i+1;advance(s)},Math.max(20,dt))}for(const e of [query,opacity,grid,heatStyle,heatScale])e.oninput=()=>states.forEach(render);states.forEach(render);
</script></html>'''
(OUT/'index.html').write_text(html.replace('__HEATMAP_JS__',(ROOT/'hardware/src/vibe_grail_joint/deploy/heatmap.js').read_text()).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
print(OUT/'index.html')
