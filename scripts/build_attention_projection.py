"""Render recorded pre-V softmax attention without changing model or source data."""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'records/attention_projection_20261029'
PREVIEW = ROOT / 'records/g1_sim_compare_20261029/sim_15cm_30cm_04/preview'


def main():
    OUT.mkdir(exist_ok=True)
    source = json.loads((PREVIEW / 'preview.json').read_text())
    frames = []
    for f in source['frames']:
        if not f.get('policy') or not f.get('attention') or not f['step']['compute']:
            continue
        attention = np.asarray(f['attention'])
        assert attention.shape == (3, 28) and np.isfinite(attention).all()
        assert (attention >= 0).all() and (attention <= 1).all()
        assert np.allclose(attention.sum(-1), 1, atol=1e-6)
        frames.append(dict(source_step=f['step']['step_id'], time=f['step']['inference_at'],
                           width=f['width'], height=f['height'], attention=f['attention'],
                           image='data:image/png;base64,'+base64.b64encode((PREVIEW/f['policy']).read_bytes()).decode('ascii')))
    labels = ['Policy RGB', 'task_cmd attention', 'proprio attention', 'CLS attention']
    palette = np.asarray([[22,26,90],[22,95,220],[22,211,192],[255,231,60],[240,45,35]])
    selected = [next(f for f in frames if f['source_step'] == step) for step in (280,320,360)]
    canvas = Image.new('RGB', (1120, 690), (20,27,34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for row, f in enumerate(selected):
        rgb = Image.open(io.BytesIO(base64.b64decode(f['image'].split(',')[1]))).convert('RGB')
        base = np.asarray(rgb.resize((280,160), Image.Resampling.BILINEAR))
        y = row * 230
        draw.text((5,y+3), f"SIM {f['time']:.1f} s | Pre-V softmax attention | Shared color range 0..1 | Aligned 4 x 7 patches", font=font, fill='white')
        for col, label in enumerate(labels):
            draw.text((col*280+5,y+26), label, font=font, fill='white')
            if col == 0:
                im = Image.fromarray(base)
            else:
                weights = np.asarray(f['attention'][col-1], np.float32).reshape(4,7)
                values = np.asarray(Image.fromarray(weights).resize((280,160), Image.Resampling.BILINEAR))
                heat = np.stack([np.interp(values, np.linspace(0,1,5), palette[:,j]) for j in range(3)], axis=-1)
                im = Image.fromarray(np.round(base*.45+heat*.55).astype(np.uint8))
                draw.text((col*280+5,y+209), f'max={weights.max():.4f}  sum={weights.sum():.4f}', font=font, fill='#bbccd8')
            canvas.paste(im, (col*280,y+47))
    canvas.save(OUT / 'attention_projection.png')
    helper = (ROOT / 'hardware/src/vibe_grail_joint/deploy/heatmap.js').read_text()
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>交叉注意力投影 · 已保存楼梯回放</title><style>
body{background:#111820;color:#e8eef2;font:15px system-ui;margin:26px auto;padding:0 20px;max-width:1400px}h1{font-size:24px}p{line-height:1.7;color:#b8c7d2}a{color:#8be1b6}.views{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card{background:#18222d;border:1px solid #34424e;padding:12px;border-radius:10px}.card h2{font-size:15px;font-weight:500;margin:0 0 12px}.card canvas{width:100%;height:auto;display:block}.card p{font-size:12px;margin:10px 0 0}input[type=range]{width:100%;accent-color:#8be1b6}button,select{background:#253443;color:white;border:1px solid #415263;border-radius:6px;padding:9px;margin-right:12px}.controls{margin:18px 0;display:flex;flex-wrap:wrap;align-items:center;gap:8px}.legend{height:8px;background:linear-gradient(90deg,#161a5a,#165fdc,#16d3c0,#ffe73c,#f02d23);margin:14px 0 5px}.scale{display:flex;justify-content:space-between;font-size:12px}@media(max-width:850px){.views{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.views{grid-template-columns:1fr}}</style>
<h1>交叉注意力投影 · 乘 V 之前</h1>
<p>已保存的仿真：上楼 3 级 · 每级高 15 cm · 踏面 30 cm · 目标速度 0.4 m/s。这里回放历史记录，实时网页的台阶数量由当前场景设置决定。</p>
<div class="views">__CARDS__</div><div class="legend"></div><div class="scale"><span>0</span><span id="range">1 · 注意力权重</span></div>
<p id="stamp"></p><input id="frame" aria-label="选择帧" type="range" min="0" max="__MAX__" value="__START__" step="1">
<div class="controls"><button id="previous">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><select id="scale" aria-label="色标"><option value="absolute">统一 0～1</option><option value="relative">每图按自身最大值着色</option></select><select id="style" aria-label="插值"><option value="smooth">平滑投影</option><option value="blocks">原始色块</option></select><label><input id="grid" type="checkbox">16×16 网格</label></div>
<p>直接读取该帧记录的 A = softmax(QKᵀ / 8)，三种 query 分别显示；图像与注意力按原记录配对。当前模型为 4×7 个 patch，投影覆盖完整的 112×64 图像。平滑不增加空间分辨率，热区不等于精确的边缘轮廓。<a href="https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/cross_attention.py#L127-L148" target="_blank">注意力取点源码</a></p>
<script>__HELPER__</script><script>
const frames=__FRAMES__,$=id=>document.getElementById(id);let revision=0,timer=null,playing=false;
function show(){const f=frames[Number($('frame').value)],version=++revision,image=new Image();image.onload=()=>{if(version!==revision)return;for(let i=0;i<4;i++){const r=VibeHeatmap.draw($('view'+i),image,f,{choice:i===0?'none':String(i-1),scale:$('scale').value,style:$('style').value,alpha:.55,grid:$('grid').checked});$('info'+i).textContent=i===0?'该次推理使用的原始 RGB':`峰值 ${r.peak.toPrecision(3)} · 第 ${Math.floor(r.peakIndex/r.cols)+1} 行、第 ${r.peakIndex%r.cols+1} 列 · 色标 0～${r.high.toPrecision(3)}`;}$('stamp').textContent=`仿真 ${f.time.toFixed(2)} s · 第 ${f.source_step} 步 · ${Number($('frame').value)+1}/${frames.length} 帧`;$('range').textContent=$('scale').value==='absolute'?'1 · 三组 query 统一色标':'各图最大值 · 以图下数值为准';};image.src=f.image;}
function stop(){playing=false;clearTimeout(timer);timer=null;$('play').textContent='播放';}
function advance(){if(!playing)return;const i=Number($('frame').value);if(i>=frames.length-1){stop();return;}timer=setTimeout(()=>{$('frame').value=i+1;show();advance();},Math.max(20,(frames[i+1].time-frames[i].time)*1000));}
$('play').onclick=()=>{if(playing)stop();else{if(Number($('frame').value)===frames.length-1)$('frame').value=0;playing=true;$('play').textContent='暂停';show();advance();}};
$('previous').onclick=()=>{stop();$('frame').value=Math.max(0,Number($('frame').value)-1);show();};$('next').onclick=()=>{stop();$('frame').value=Math.min(frames.length-1,Number($('frame').value)+1);show();};$('frame').oninput=()=>{stop();show();};['scale','style','grid'].forEach(id=>$(id).oninput=show);show();
</script></html>'''
    cards = ''.join(f'<div class="card"><h2>{title}</h2><canvas id="view{i}" width="448" height="256"></canvas><p id="info{i}"></p></div>' for i,title in enumerate(['策略 RGB','task_cmd · 任务指令','proprio · 本体状态','CLS · 视觉摘要']))
    html = html.replace('__CARDS__', cards).replace('__MAX__', str(len(frames)-1)).replace('__START__', str(next(i for i,f in enumerate(frames) if f['source_step']==280)))
    html = html.replace('__HELPER__',helper).replace('__FRAMES__',json.dumps(frames,separators=(',',':')))
    (OUT / 'attention_replay.html').write_text(html)
    metadata = dict(source_preview=str(PREVIEW.relative_to(ROOT)/'preview.json'), frames=len(frames),
                    start_time=frames[0]['time'],end_time=frames[-1]['time'], attention_shape=[3,28],
                    queries=['task_cmd','proprio','CLS'], color_scale=[0,1], patch_grid=[4,7],
                    source='Recorded pre-V softmax output of the deployed policy',
                    interpolation='Display only, bilinear scalar interpolation aligned to 16x16 patch centers',
                    original_results_retained=True)
    (OUT/'replay_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
