"""Check the four requested height presets against the live local simulator."""
import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8091')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result=dict(browser_errors=[],presets=[],controls={},interface_passed=False)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.on('pageerror',lambda e:result['browser_errors'].append(str(e)))
        page.goto(args.url)
        page.wait_for_function('state.ready',timeout=60000)
        options=page.locator('#stepHeight option').evaluate_all('(o)=>o.map(x=>Number(x.value))')
        assert options==[.05,.10,.15,.20],options
        assert page.locator('#terrain').input_value()=='up'
        assert float(page.locator('#speed').input_value())==.4
        initial=page.request.get(args.url+'/api/state').json()
        assert initial['terrain']['height']==.15 and initial['terrain']['n']==3 and initial['terrain']['tread']==.30,initial['terrain']
        for h in options:
            page.select_option('#stepHeight',f'{h:.2f}')
            page.click('#reset')
            page.wait_for_function('(h)=>state.t===0 && state.terrain.height===h && !state.running',arg=h,timeout=20000)
            start=page.request.get(args.url+'/api/state').json()
            assert start['terrain']['tread']==.30 and start['terrain']['n']==3
            page.click('#auto')
            page.wait_for_function('state.validation!==null && !state.running',timeout=90000)
            state=page.request.get(args.url+'/api/state').json()
            case={k:state[k] for k in ['terrain','fall','reached','validation','report_file','t','position','feet']}
            result['presets'].append(case)
            page.screenshot(path=str(args.output/f'height_{round(h*100):02d}cm.png'),full_page=True)
            (args.output/'browser_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(dict(height_cm=round(h*100),passed=state['validation']['passed'],fall=state['fall'],reached=state['reached'],time=state['t'],report=state['report_file']),ensure_ascii=False),flush=True)
        # Exercise the existing controls after changing presets, before reaching stairs.
        for key,field,sign in [('w','forward',1),('s','forward',-1),('q','turn',1),('e','turn',-1)]:
            page.select_option('#stepHeight','0.05');page.click('#reset')
            page.wait_for_function('state.t===0 && !state.running',timeout=20000)
            page.locator('#scene').click()
            page.keyboard.down(key)
            page.wait_for_function('state.t>=.6 && state.walking',timeout=10000)
            held=page.request.get(args.url+'/api/state').json()
            assert held[field]==sign and not held['fall']
            page.keyboard.up(key)
            page.wait_for_function('!state.walking && state.speed_command===0 && state.yaw_rate_command===0',timeout=10000)
            result['controls'][key]=dict(held_direction=held[field],released=True)
        page.select_option('#stepHeight','0.15');page.click('#reset')
        page.wait_for_function('state.t===0 && state.terrain.height===.15 && !state.running',timeout=20000)
        page.screenshot(path=str(args.output/'ready_15cm.png'),full_page=True)
        result['interface_passed']=not result['browser_errors']
        browser.close()
    (args.output/'browser_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Browser height presets and W/S/Q/E passed; physical results retained separately.',flush=True)


if __name__=='__main__':main()
