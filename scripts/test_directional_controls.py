"""Verify browser keys produce physical backward motion/turns and stop on release."""
import json
from pathlib import Path
import time

import numpy as np
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"records/directional_controls"
URL="http://127.0.0.1:8080"


def main():
    errors=[]
    cases={}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={"width":1440,"height":1150})
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.on("response",lambda r:errors.append(f"HTTP {r.status} {r.url}") if r.status>=400 else None)
        page.goto(URL)
        page.wait_for_function("!document.querySelector('#backward').disabled",timeout=60000)
        def state(): return page.request.get(URL+"/api/state").json()
        def reset():
            page.select_option("#terrain","flat")
            page.click("#reset")
            page.wait_for_function("state.t===0&&!state.running")
            page.locator("#scene").click()
        def sample(seconds):
            start=time.monotonic()
            rows=[]
            while time.monotonic()-start<seconds:
                s=state();rows.append(s)
                assert not s["fall"],s
                page.wait_for_timeout(100)
            return rows
        for key,sign in [("s",-1),("q",1),("e",-1)]:
            reset()
            page.keyboard.down(key)
            rows=sample(4)
            held=state()
            page.keyboard.up(key)
            rows+=sample(3)
            stopped=state()
            assert stopped["speed_command"]==0 and stopped["yaw_rate_command"]==0,stopped
            assert not stopped["walking"],stopped
            if key=="s": assert stopped["position"][0]<-.5,stopped
            else:
                assert sign*stopped["yaw"]>.65,stopped
                assert np.linalg.norm(stopped["position"][:2])<.4,stopped
            assert abs(stopped["actual_forward_speed"])<.12,stopped
            cases[key]=dict(held=held,stopped=stopped,trace=rows)
            print(key,stopped["position"],stopped["yaw"],flush=True)

        # Simultaneous keys and releasing one key must preserve the other.
        reset()
        page.keyboard.down("w");page.keyboard.down("q")
        sample(3)
        combined=state()
        assert combined["speed_command"]>0 and combined["yaw_rate_command"]>0,combined
        page.keyboard.up("q")
        sample(1)
        remaining=state()
        assert remaining["speed_command"]>0 and remaining["yaw_rate_command"]==0,remaining
        assert remaining["position"][1]>.05,remaining
        page.keyboard.up("w");sample(2)
        cases["w_q"]=dict(combined=combined,remaining=remaining,stopped=state())

        # A pointer button controls real turn commands; focus loss stops all keys.
        reset()
        page.locator("#left").hover();page.mouse.down();sample(.6)
        pointer=state();assert pointer["turn"]==1,pointer
        page.mouse.up();sample(.8)
        assert state()["turn"]==0
        page.locator("#scene").click();page.keyboard.down("s");sample(.6)
        page.evaluate("window.dispatchEvent(new Event('blur'))")
        sample(1)
        lost_focus=state()
        assert lost_focus["speed_command"]==0 and lost_focus["yaw_rate_command"]==0,lost_focus
        page.keyboard.up("s")
        cases["pointer_and_blur"]=dict(pointer=pointer,stopped=lost_focus)

        # No client heartbeat: the backend must cancel both motion axes.
        page.request.post(URL+"/api/control",data=dict(action="drive",speed=.25,forward=-1,turn=1))
        sample(1.5)
        disconnected=state()
        assert disconnected["speed_command"]==0 and disconnected["yaw_rate_command"]==0,disconnected
        cases["heartbeat_timeout"]=disconnected
        assert not errors,errors
        page.select_option("#terrain","up_down");page.click("#reset")
        page.wait_for_function("state.t===0&&!state.running")
        page.screenshot(path=str(OUT/"controls.png"),full_page=True)
        result=dict(passed=True,browser_errors=errors,cases=cases)
        (OUT/"browser.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        print("Directional browser tests passed",flush=True)
        browser.close()


if __name__=="__main__": main()
