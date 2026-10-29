"""Portable version of the training G1, actuators and simple box stairs."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def terrain(kind="up_down", height=.10, count=4, tread=.30, start=1.5, plateau=1.2):
    return dict(kind=kind, height=height, n=count, tread=tread, start=start, plateau=plateau,
                goal=start + count*tread + (plateau+count*tread if kind == "up_down" else 0) + .8)


def ground(x, tile):
    n, h, start, tread = [tile[k] for k in ("n", "height", "start", "tread")]
    up = np.clip(np.floor((np.asarray(x)-start)/tread)+1, 0, n)
    if tile["kind"] == "flat":
        return np.zeros_like(up)
    if tile["kind"] == "down":
        return (n-up)*h
    if tile["kind"] == "up_down":
        down = np.clip(np.floor((np.asarray(x)-(start+n*tread+tile["plateau"]))/tread)+1, 0, n)
        return (up-down)*h
    return up*h


def make_model(contract, camera, tile):
    root = ET.parse(ROOT / "assets/xmls/g1.xml").getroot()
    root.find("compiler").set("meshdir", str(ROOT / "assets/xmls/assets"))
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast", iterations="10",
                  ls_iterations="20", cone="pyramidal", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="540")
    ET.SubElement(visual, "map", znear="0.01", zfar="100")
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.3", diffuse="0.7 0.7 0.7", specular="0 0 0")
    ET.SubElement(root, "statistic", extent="5", center="2 0 0.5")
    world = root.find("worldbody")
    ET.SubElement(world, "light", name="sun", directional="true", pos="0 0 10", dir="0 0 -1", castshadow="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .1", rgba=".35 .36 .38 1")
    breaks = [-4., tile["start"]] + [tile["start"]+j*tile["tread"] for j in range(1, tile["n"])]
    if tile["kind"] == "up_down":
        ds = tile["start"] + tile["n"]*tile["tread"] + tile["plateau"]
        breaks += [ds+j*tile["tread"] for j in range(tile["n"])]
    breaks += [16.]
    for i, (left, right) in enumerate(zip(sorted(set(breaks))[:-1], sorted(set(breaks))[1:])):
        h = float(ground((left+right)/2, tile))
        if h > 0:
            ET.SubElement(world, "geom", name=f"stair_{i}", type="box",
                          size=f"{(right-left)/2} 2 {h/2}", pos=f"{(left+right)/2} 0 {h/2}",
                          rgba=".50 .36 .22 1", friction=".8 .005 .0001")
    torso = root.find(".//body[@name='torso_link']")
    ET.SubElement(torso, "camera", name="head_rgb", pos=" ".join(map(str,camera["pos"])),
                  quat=" ".join(map(str,camera["quat"])), fovy=str(camera["fovy"]))
    # FULL_COLLISION from mjlab's G1 config: feet have priority 1, friction .6.
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.endswith("_collision"):
            foot = "_foot" in name
            geom.set("condim", "3" if foot else "1")
            if foot:
                geom.set("priority", "1")
                geom.set("friction", ".6 .005 .0001")
    actuators = ET.SubElement(root, "actuator")
    for i, name in enumerate(contract["joint_names"]):
        kp, kd, force = [contract[k][i] for k in ("stiffness", "damping", "effort_limit")]
        joint = root.find(f".//joint[@name='{name}']")
        joint.set("armature", str(kp/(20*np.pi)**2))
        ET.SubElement(actuators, "position", name=name, joint=name, kp=str(kp), kv=str(kd),
                      forcerange=f"{-force} {force}", ctrllimited="false")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
