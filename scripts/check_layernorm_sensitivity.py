"""Compare the live diagnostic with the saved third column and build its view."""
import base64
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rosbags.highlevel import AnyReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.sensitivity import LayerNormProbe


def main():
    output = ROOT / 'records/feature_layers_20261029'
    originals = json.loads((output / 'late_output_sensitivity.json').read_text())['probes']
    expected = {p['step_id']: p for p in originals}
    probe = LayerNormProbe(ROOT / 'models/vggt_hierarchical_rgb64')
    bag = ROOT / 'records/g1_sim_compare_20261029/sim_15cm_30cm_04/bag'
    packets, checks = [], []
    with AnyReader([bag]) as reader:
        stamps = {}
        for c, stamp, raw in reader.messages(connections=[c for c in reader.connections if c.topic == '/vibe/policy/step']):
            step = json.loads(reader.deserialize(raw, c.msgtype).data)
            if step['step_id'] in expected:
                stamps[step['step_id']] = stamp
        assert set(stamps) == set(expected)
        for step_id, stamp in sorted(stamps.items()):
            feed = {}
            for c, _, raw in reader.messages(start=stamp, stop=stamp+1):
                m = reader.deserialize(raw, c.msgtype)
                if c.topic.startswith('/vibe/policy/inputs/'):
                    feed[c.topic.split('/')[-1]] = np.array(m.data, np.float32).reshape([d.size for d in m.layout.dim])
                elif c.topic == '/vibe/camera/policy/image':
                    bgr = np.asarray(m.data).reshape(64, 112, 3)
            start = time.monotonic()
            scores = probe.compute(bgr, feed)
            elapsed = time.monotonic() - start
            error = float(np.max(np.abs(np.array(scores) - expected[step_id]['after_layernorm_change_rmse'])))
            assert error < 1e-6, (step_id, error)
            buffer = io.BytesIO()
            Image.fromarray(bgr[..., ::-1]).save(buffer, format='PNG')
            packets.append(dict(source_step=step_id, time=step_id*.02, width=112, height=64,
                                layernorm_sensitivity=scores,
                                image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')))
            checks.append(dict(step=step_id, max_error_vs_original_third_column=error, compute_s=elapsed))
    # Keep the original four-column figure intact. Build only column 3 here.
    palette = np.array([[22,26,90],[22,95,220],[22,211,192],[255,231,60],[240,45,35]])
    canvas = Image.new('RGB', (560, 420*len(packets)), (20,27,34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    for row, p in enumerate(packets):
        rgb = np.array(Image.open(io.BytesIO(base64.b64decode(p['image'].split(',')[1]))))
        values = np.array(p['layernorm_sensitivity'], np.float32).reshape(4,7)
        high = float(values.max())
        display = np.asarray(Image.fromarray(values/max(high,1e-12)).resize((560,320), Image.Resampling.BILINEAR))
        heat = np.stack([np.interp(display, np.linspace(0,1,5), palette[:,j]) for j in range(3)], axis=-1)
        base = np.asarray(Image.fromarray(rgb).resize((560,320), Image.Resampling.BILINEAR))
        overlay = Image.fromarray(np.round(base*.45+heat*.55).astype(np.uint8))
        y = row*420
        draw.text((12,y+7), f"Post-LayerNorm sensitivity | SIM {p['time']:.1f} s", font=font, fill='white')
        canvas.paste(overlay, (0,y+40))
        draw.text((12,y+370), f'RMSE: 0 .. {high:.5f} | 4 x 7 patches', font=font, fill='#bbccd8')
    canvas.save(output / 'layernorm_sensitivity.png')
    (output / 'layernorm_only.json').write_text(json.dumps(packets, indent=2)+'\n')
    helper = (ROOT / 'hardware/src/vibe_grail_joint/deploy/heatmap.js').read_text()
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第三列 · LayerNorm 输出敏感性</title><style>
body{background:#111820;color:#e8eef2;font:16px system-ui;margin:32px auto;padding:0 20px;max-width:800px}h1{font-size:23px}canvas{width:100%;height:auto;background:#18222d;border-radius:10px}p{line-height:1.7;color:#b8c7d2}input[type=range]{width:100%;accent-color:#8be1b6}button,select{background:#253443;color:white;border:1px solid #415263;border-radius:6px;padding:9px;margin-right:12px}.legend{height:8px;background:linear-gradient(90deg,#161a5a,#165fdc,#16d3c0,#ffe73c,#f02d23);margin:10px 0}.scale{display:flex;justify-content:space-between;font-size:13px}.controls{margin:18px 0}</style>
<h1>第三列 · LayerNorm 输出敏感性</h1>
<p>0.4 m/s · 台阶高 15 cm · 踏面 30 cm。只显示逐块遮挡引起的 LayerNorm 输出变化，叠加在同一帧策略图像上。</p>
<canvas id="heat" width="672" height="384"></canvas><div class="legend"></div><div class="scale"><span>0</span><span id="maximum"></span></div>
<p id="info"></p><input id="frame" aria-label="选择取样帧" type="range" min="0" max="2" step="1" value="0">
<div class="controls"><button id="play">播放三帧</button><select id="style" aria-label="热图显示方式"><option value="smooth">平滑热区</option><option value="blocks">原始色块</option></select><label><input id="grid" type="checkbox">显示网格</label></div>
<p>原始分辨率为 4×7，色标按每帧最大值归一化；跨帧比较强弱时请看 RMSE 数值。全部非视觉输入保持该帧原值，并重新计算遮挡后的 patch 与 CLS。这是输出敏感性，不是台阶边缘标注。</p>
<script>__HELPER__</script><script>
const packets=__PACKETS__;const $=id=>document.getElementById(id);let version=0,timer=null;
function show(){const p=packets[Number($('frame').value)],revision=++version,image=new Image();image.onload=()=>{if(revision!==version)return;const r=VibeHeatmap.draw($('heat'),image,p,{choice:'layernorm',style:$('style').value,scale:'relative',alpha:.55,grid:$('grid').checked});$('maximum').textContent=r.high.toFixed(5)+' RMSE';$('info').textContent=`仿真 ${p.time.toFixed(1)} s · 第 ${p.source_step} 步 · ${VibeHeatmap.describe(r)}`;};image.src=p.image;}
['frame','style','grid'].forEach(id=>$(id).oninput=show);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放三帧';}else{timer=setInterval(()=>{$('frame').value=(Number($('frame').value)+1)%packets.length;show();},1500);$('play').textContent='暂停';}};show();
</script></html>'''
    (output / 'layernorm_only.html').write_text(html.replace('__HELPER__', helper).replace('__PACKETS__', json.dumps(packets)))
    (ROOT / 'records/layernorm_view_20261029/numeric_check.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
