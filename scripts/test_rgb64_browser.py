"""Test the imported model through real browser keys, using simulation time."""
import argparse
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False, cases={}, browser_errors=[])
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width':1440, 'height':1100})
        page.on('pageerror', lambda e:result['browser_errors'].append(str(e)))
        page.goto(a.url)
        try:
            page.wait_for_function('state.ready', timeout=90000)
            def state():
                return page.request.get(a.url+'/api/state').json()
            def reset(kind='flat'):
                page.select_option('#terrain',kind)
                page.click('#reset')
                page.wait_for_function('state.t===0 && !state.running',timeout=15000)
                page.locator('#scene').click()
            def advance(seconds):
                start = state()['t']
                until = time.monotonic()+90.
                rows = []
                while time.monotonic()<until:
                    s = state()
                    rows.append(s)
                    if s['fall']:
                        raise AssertionError(f"Robot fell at t={s['t']}")
                    if s['t']>=start+seconds:
                        return rows
                    page.wait_for_timeout(100)
                raise AssertionError('Simulation progress timeout')
            model = state()['model']
            assert model['image_shape']==[64,112,3] and model['patch_shape']==[28,192],model
            assert model['checkpoint']=='model_2400.pt',model
            result['model'] = model
            for key in ['w','s','q','e']:
                reset()
                case = result['cases'][key] = {}
                page.keyboard.down(key)
                case['held_trace'] = advance(4.)
                case['held'] = state()
                page.keyboard.up(key)
                case['stop_trace'] = advance(3.)
                end = case['stopped'] = state()
                assert end['speed_command']==0 and end['yaw_rate_command']==0 and not end['walking'],end
                if key=='w': assert end['position'][0]>.4,end
                if key=='s': assert end['position'][0]<-.4,end
                if key=='q': assert end['yaw']>.5,end
                if key=='e': assert end['yaw']<-.5,end
                assert abs(end['actual_forward_speed'])<.15,end
                case['passed'] = True
                print(json.dumps(dict(key=key,position=end['position'],yaw=end['yaw'],passed=True)),flush=True)
            page.click('#pause')
            page.wait_for_function('!state.running')
            paused = state()['t']
            page.wait_for_timeout(500)
            assert state()['t']==paused
            result['pause_verified'] = True
            assert not result['browser_errors'],result['browser_errors']
            reset('up_down')
            page.wait_for_timeout(300)
            page.screenshot(path=str(a.output.with_suffix('.png')),full_page=True)
            result['passed'] = True
        except BaseException as exc:
            result['failure'] = repr(exc)
            page.screenshot(path=str(a.output.with_suffix('.failure.png')),full_page=True)
        finally:
            page.request.post(a.url+'/api/control',data=dict(action='pause'))
            a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            browser.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','model')},ensure_ascii=False),flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
