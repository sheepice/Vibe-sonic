"""Exercise the actual page: W down/up, pause, reset, automatic stair traversal."""
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
url="http://127.0.0.1:8080"


def main():
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={"width":1440,"height":1000})
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.on("response",lambda r:errors.append(f"HTTP {r.status} {r.url}") if r.status>=400 else None)
        page.goto(url)
        page.wait_for_function("!document.querySelector('#auto').disabled",timeout=60000)
        def state(): return page.request.get(url+"/api/state").json()
        page.click("#reset")
        page.wait_for_timeout(500)
        page.locator("#scene").click()
        page.keyboard.down("w")
        page.wait_for_timeout(5000)
        walking=state()
        page.keyboard.up("w")
        page.wait_for_timeout(4500)
        stopped=state()
        assert walking["position"][0]>.5 and not walking["fall"],walking
        assert stopped["speed_command"]==0 and abs(stopped["actual_vx"])<.10 and not stopped["fall"],stopped
        page.click("#pause")
        page.wait_for_timeout(300)
        paused=state()
        page.wait_for_timeout(500)
        assert state()["t"]==paused["t"]
        page.select_option("#terrain","up_down")
        page.click("#reset")
        page.wait_for_timeout(600)
        assert state()["t"]==0
        page.click("#auto")
        start=time.monotonic()
        screenshots=set()
        while time.monotonic()-start<75:
            s=state()
            if s.get("fall"): raise AssertionError(s)
            if 2.8<s["position"][0]<3.8 and "top" not in screenshots:
                page.screenshot(path=str(ROOT/"records/browser_top.png"),full_page=True)
                screenshots.add("top")
            if 4.6<s["position"][0]<5.2 and "down" not in screenshots:
                page.screenshot(path=str(ROOT/"records/browser_down.png"),full_page=True)
                screenshots.add("down")
            if s.get("validation"):
                assert s["validation"]["passed"],s
                break
            page.wait_for_timeout(200)
        else: raise AssertionError("Browser traversal did not finish")
        page.wait_for_timeout(300)
        page.screenshot(path=str(ROOT/"records/browser_passed.png"),full_page=True)
        assert not errors,errors
        result=dict(passed=True,browser_errors=errors,walking=walking,stopped=stopped,final=s,
                    screenshots=sorted(screenshots),wall_seconds=time.monotonic()-start)
        (ROOT/"records/browser_test.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({"passed":True,"report":s["report_file"],"checks":s["validation"]},ensure_ascii=False),flush=True)
        page.click("#reset")
        page.wait_for_timeout(500)
        browser.close()


if __name__=="__main__": main()
