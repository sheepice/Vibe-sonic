"""Export a local, interactive HTML replay of actual bag images and attention.

Join by the recorded source image timestamp, never a nearby substitute frame.
Feature magnitude and cross attention are diagnostics, not stair segmentation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader


def array(message):
    return np.asarray(message.data).reshape([d.size for d in message.layout.dim])


def rgb(message):
    if message.encoding not in ('bgr8','rgb8') or message.is_bigendian:
        raise ValueError('Unsupported image encoding')
    data = np.asarray(message.data).reshape(message.height,message.step)
    value = data[:,:message.width*3].reshape(message.height,message.width,3)
    return value[:,:,::-1].copy() if message.encoding=='bgr8' else value.copy()


def viewer_html(report):
    """Self-contained viewer so local replays need no server or external assets."""
    template = Path(__file__).with_name('bag_viewer.html').read_text()
    if 'overview_hz' in report['metadata']:
        template = template.replace('原始彩色图像（预览缩小）','仿真外部视角（同一步）').replace('原图采样：','外部视角：')
        template = template.replace('DATA.metadata.raw_hz','DATA.metadata.overview_hz')
        template = template.replace('bag 中的 image_raw 保留原始像素，左侧 JPEG 仅用于快速预览。',
                                    '本 bag 保存实际策略图像；左侧为仿真外部视角，不是 D435i 原图。')
    javascript = Path(__file__).with_name('heatmap.js').read_text()
    payload = json.dumps(report,ensure_ascii=False).replace('<','\\u003c')
    return template.replace('__VIBE_HEATMAP_JS__',javascript).replace('__VIBE_DATA__',payload)


def export(bag, output, *, fps=5., start=0., seconds=None, max_frames=1500):
    bag, output = Path(bag), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new preview directory')
    if not 0<fps<=50 or start<0 or max_frames<=0 or (seconds is not None and seconds<=0):
        raise ValueError('Invalid preview interval')
    assets = output/'frames'
    assets.mkdir(parents=True)
    steps, attention, actions, metadata, events = [], {}, {}, {}, []
    topics = {'/vibe/policy/step','/vibe/policy/attention','/vibe/policy/actions',
              '/vibe/recording/metadata','/vibe/recording/status','/vibe/control/events'}
    recording_status = None
    with AnyReader([bag]) as reader:
        for connection, stamp, raw in reader.messages(connections=[c for c in reader.connections if c.topic in topics]):
            value = reader.deserialize(raw, connection.msgtype)
            topic = connection.topic
            if topic=='/vibe/policy/step':
                steps.append((stamp,json.loads(value.data)))
            elif topic=='/vibe/policy/attention':
                attention[stamp] = array(value).tolist()
            elif topic=='/vibe/policy/actions':
                actions[stamp] = array(value).tolist()
            elif topic=='/vibe/recording/metadata':
                metadata = json.loads(value.data)
            elif topic=='/vibe/recording/status':
                recording_status = json.loads(value.data)
            else:
                event = json.loads(value.data)
                if event.get('event') not in ('control_tick','nominal_pose_hold','recording_status'):
                    events.append(dict(stamp_ns=stamp,**event))
        if not steps:
            raise ValueError('No inference steps recorded in this bag')
        origin = steps[0][0]
        selected = []
        next_time = start
        for stamp, step in steps:
            at = (stamp-origin)/1e9
            if at < next_time or (seconds is not None and at > start+seconds):
                continue
            selected.append((stamp,step))
            next_time = at+1./fps-1e-4
        truncated = len(selected)>max_frames
        selected = selected[:max_frames]
        image_topics = {'/vibe/camera/policy/image','/vibe/camera/color/image_raw','/vibe/vision/patches'}
        conns = [c for c in reader.connections if c.topic in image_topics]
        image_data = {}
        for image_stamp in sorted({s['image_stamp_ns'] for _,s in selected}):
            entry = dict(policy=None,raw=None,edges=None,feature_norm=None)
            for connection, _, raw in reader.messages(connections=conns,start=image_stamp,stop=image_stamp+1):
                msg = reader.deserialize(raw, connection.msgtype)
                if connection.topic=='/vibe/vision/patches':
                    entry['feature_norm'] = np.linalg.norm(array(msg),axis=-1).tolist()
                    continue
                frame = rgb(msg)
                if connection.topic=='/vibe/camera/policy/image':
                    name = f'frames/{image_stamp}_policy.png'
                    Image.fromarray(frame).save(output/name)
                    entry.update(policy=name,width=frame.shape[1],height=frame.shape[0])
                    gray = frame.astype(np.float32).mean(-1)
                    gy,gx = np.gradient(gray)
                    magnitude = np.hypot(gx,gy)
                    edge = magnitude>max(8.,float(np.percentile(magnitude,88)))
                    rgba = np.full((*edge.shape,4),255,np.uint8)
                    rgba[:,:,3] = edge.astype(np.uint8)*220
                    name = f'frames/{image_stamp}_edges.png'
                    Image.fromarray(rgba).save(output/name)
                    entry['edges'] = name
                else:
                    # Only the HTML preview is reduced/JPEG-compressed. Bag
                    # sensor_msgs/Image remains the exact captured BGR bytes.
                    name = f'frames/{image_stamp}_raw.jpg'
                    image = Image.fromarray(frame)
                    image.thumbnail((640,360))
                    image.save(output/name,quality=85)
                    entry['raw'] = name
            image_data[image_stamp] = entry
        frames = [dict(time=(stamp-origin)/1e9,stamp_ns=stamp,step=step,
                       attention=attention.get(stamp),action=actions.get(stamp),
                       **image_data[step['image_stamp_ns']]) for stamp,step in selected]
    if not frames:
        raise ValueError('No frames in requested interval')
    status_path = bag/'recording_status.json'
    if status_path.exists():
        recording_status = json.loads(status_path.read_text())
    report = dict(bag=str(bag.resolve()),frames=frames,metadata=metadata,events=events,
        recording_status=recording_status,truncated=truncated,
        missing_policy_images=sum(f['policy'] is None for f in frames),
        missing_attention_on_computed_steps=sum(f['step']['compute'] and f['attention'] is None for f in frames),
        note='True exported attention after spatial mixing. No causal/edge-recognition claim.')
    (output/'preview.json').write_text(json.dumps(report,ensure_ascii=False)+'\n')
    (output/'index.html').write_text(viewer_html(report))
    return dict(frames=len(frames),missing_policy_images=report['missing_policy_images'],
                missing_attention_on_computed_steps=report['missing_attention_on_computed_steps'],
                truncated=truncated,index=str((output/'index.html').resolve()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fps',type=float,default=5.)
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--seconds',type=float)
    p.add_argument('--max-frames',type=int,default=1500)
    a = p.parse_args()
    print(json.dumps(export(a.bag,a.output,fps=a.fps,start=a.start,seconds=a.seconds,
                            max_frames=a.max_frames),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
