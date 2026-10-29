"""Browser/data checks for display-only heatmaps and exact live frame pairing."""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from rosbags.highlevel import AnyReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'records/heatmap_style_20261029'
report=dict(browser_errors=[],replays=[],synthetic={},live={})
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1600,'height':1100})
    page.on('pageerror',lambda e:report['browser_errors'].append(str(e)))
    compare=ROOT/'records/g1_sim_compare_20261029/index.html'
    page.goto(compare.as_uri())
    page.wait_for_function("Array.from(document.querySelectorAll('.heatRange')).every(e=>e.textContent.includes('4×7'))")
    page.screenshot(path=str(OUT/'comparison_smooth.png'),full_page=True)
    for choice in ['mean','0','1','2','feature','none']:
        page.select_option('#query',choice);page.wait_for_timeout(60)
    page.select_option('#query','mean');page.select_option('#heatStyle','blocks');page.check('#grid')
    page.wait_for_timeout(100);page.screenshot(path=str(OUT/'comparison_blocks.png'),full_page=True)
    report['synthetic']=page.evaluate("""()=>{
      const image=document.createElement('canvas');image.width=112;image.height=64;
      image.getContext('2d').fillStyle='black';image.getContext('2d').fillRect(0,0,112,64);
      const canvas=document.createElement('canvas');canvas.width=224;canvas.height=128;
      const a=Array(28).fill(0);a[10]=1;const frame={width:112,height:64,attention:[a,a,a]};
      const before=JSON.stringify(frame),pixel=(x,y)=>Array.from(canvas.getContext('2d').getImageData(x,y,1,1).data).slice(0,3);
      VibeHeatmap.draw(canvas,image,frame,{alpha:1,style:'smooth'});
      const cold=pixel(0,0),peak=pixel(111,47),s1=pixel(97,47),s2=pixel(103,47);
      VibeHeatmap.draw(canvas,image,frame,{alpha:1,style:'blocks'});
      const b1=pixel(97,47),b2=pixel(103,47);
      VibeHeatmap.draw(canvas,image,frame,{alpha:0});const transparent=pixel(111,47);
      const legacy={width:112,height:63,attention:[Array(21).fill(1),Array(21).fill(1),Array(21).fill(1)]};
      VibeHeatmap.draw(canvas,image,legacy,{alpha:1});const legacyBottom=pixel(100,125);
      return {cold,peak,smooth_varies:JSON.stringify(s1)!==JSON.stringify(s2),blocks_constant:JSON.stringify(b1)===JSON.stringify(b2),transparent,legacyBottom,source_unchanged:before===JSON.stringify(frame)};
    }""")
    x=report['synthetic'];assert x['cold']==[22,26,90] and x['peak'][0]>200,x
    assert x['smooth_varies'] and x['blocks_constant'] and x['source_unchanged'],x
    assert x['transparent']==[0,0,0] and x['legacyBottom']==[0,0,0],x
    dirs=[ROOT/'records/g1_recordings/trial_20261028_235905/preview_checked_20261029']
    dirs+=list((ROOT/'records/g1_sim_compare_20261029').glob('sim_*/preview'))
    for directory in dirs:
        page.goto((directory/'index.html').as_uri())
        page.wait_for_function("document.querySelector('#range').textContent.includes('4×7')")
        for choice in ['0','feature','mean']:page.select_option('#map',choice)
        page.select_option('#style','blocks');page.select_option('#scale','absolute');page.check('#grid')
        page.wait_for_timeout(60)
        report['replays'].append(str(directory))
    # All motor-free local simulation. Check a mid-stair live frame against the
    # already retained exact-input simulation bag at the same policy step.
    url='http://127.0.0.1:8091';page.goto(url);page.wait_for_function('state.ready')
    page.click('#reset');page.wait_for_function('state.t===0 && !state.running')
    page.click('#auto');page.wait_for_function('state.t>=6.',timeout=25000)
    page.click('#pause');page.wait_for_function('!state.running');page.wait_for_timeout(250)
    packet=page.request.get(url+'/api/vision').json()
    state=page.request.get(url+'/api/state').json()
    assert len(packet['attention'])==3 and all(len(a)==28 for a in packet['attention'])
    assert len(packet['feature_norm'])==28
    assert abs(state['t']-(packet['source_step']+1)*.02)<1e-9
    saved=np.load(ROOT/'records/g1_sim_compare_20261029/sim_15cm_30cm_04/telemetry.npz')
    error=float(np.max(abs(np.array(packet['attention'])-saved['attention'][packet['source_step']])))
    assert error==0,error
    with AnyReader([ROOT/'records/g1_sim_compare_20261029/sim_15cm_30cm_04/bag']) as reader:
        image_stamp=None
        for c,stamp,raw in reader.messages(connections=[c for c in reader.connections if c.topic=='/vibe/policy/step']):
            step=json.loads(reader.deserialize(raw,c.msgtype).data)
            if step['step_id']==packet['source_step']:image_stamp=step['image_stamp_ns'];break
        c,stamp,raw=next(reader.messages(connections=[c for c in reader.connections if c.topic=='/vibe/camera/policy/image'],start=image_stamp,stop=image_stamp+1))
        m=reader.deserialize(raw,c.msgtype);rgb=np.array(m.data).reshape(64,112,3)[:,:,::-1].copy()
        stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='JPEG',quality=95)
        assert stream.getvalue()==base64.b64decode(packet['image'].split(',')[1])
    page.wait_for_function("document.querySelector('#liveHeat').dataset.sourceStep===String(visionFrame.source_step)")
    page.screenshot(path=str(OUT/'live_smooth.png'),full_page=True)
    page.locator('.heat-settings summary').click()
    page.select_option('#heatQuery','feature');page.select_option('#heatStyle','blocks');page.check('#heatGrid')
    page.wait_for_timeout(100);page.screenshot(path=str(OUT/'live_feature_blocks.png'),full_page=True)
    report['live']=dict(source_step=packet['source_step'],source_time=packet['time'],state_time=state['t'],
                       max_attention_error=error,exact_same_source_image=True)
    (OUT/'live_packet.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    assert not report['browser_errors'],report['browser_errors']
    browser.close()
report['passed']=True
(OUT/'browser_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
