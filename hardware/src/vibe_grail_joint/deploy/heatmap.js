/* Shared display renderer. Interpolation never changes the recorded patch values. */
(function (root) {
  'use strict';
  const stops = [[22,26,90],[22,95,220],[22,211,192],[255,231,60],[240,45,35]];
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  function color(value) {
    const x=clamp(value,0,1)*4, i=Math.min(3,Math.floor(x)), t=x-i;
    return stops[i].map((v,j)=>Math.round(v*(1-t)+stops[i+1][j]*t));
  }
  function values(frame, choice) {
    if (choice==='none') return null;
    if (choice==='feature') return frame.feature_norm || null;
    if (choice==='layernorm') return frame.layernorm_sensitivity || null;
    if (!frame.attention?.length) return null;
    if (choice==='mean') return frame.attention[0].map((_,j)=>frame.attention.reduce((s,a)=>s+a[j],0)/frame.attention.length);
    return frame.attention[Number(choice)] || null;
  }
  function draw(canvas, image, frame, options={}) {
    const choice=options.choice??'mean', style=options.style??'smooth';
    const ctx=canvas.getContext('2d');
    const width=frame.width||image.naturalWidth||image.width, height=frame.height||image.naturalHeight||image.height;
    ctx.globalAlpha=1;ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.imageSmoothingEnabled=style!=='blocks';ctx.drawImage(image,0,0,canvas.width,canvas.height);
    const data=values(frame,choice), cols=Math.floor(width/16), rows=data?data.length/cols:Math.floor(height/16);
    const result={available:false,rows,cols,style,choice};
    if (data && Number.isInteger(rows) && rows>0 && data.every(Number.isFinite)) {
      const low=choice==='feature'?Math.min(...data):0;
      const high=choice==='feature'||options.scale!=='absolute'?Math.max(...data):1;
      const normalized=data.map(v=>high-low>1e-12?clamp((v-low)/(high-low),0,1):0);
      // Sample scalar weights first, then apply the color map. Clamped bilinear
      // interpolation aligns cell centers and cannot overshoot source extrema.
      const size=32, heat=document.createElement('canvas');heat.width=cols*size;heat.height=rows*size;
      const hc=heat.getContext('2d'), rgba=hc.createImageData(heat.width,heat.height);
      const at=(x,y)=>normalized[clamp(y,0,rows-1)*cols+clamp(x,0,cols-1)];
      for (let y=0;y<heat.height;y++) for(let x=0;x<heat.width;x++) {
        let value;
        if(style==='blocks') value=at(Math.floor(x/size),Math.floor(y/size));
        else {
          const sx=(x+.5)/size-.5,sy=(y+.5)/size-.5,x0=Math.floor(sx),y0=Math.floor(sy),dx=sx-x0,dy=sy-y0;
          value=(1-dy)*((1-dx)*at(x0,y0)+dx*at(x0+1,y0))+dy*((1-dx)*at(x0,y0+1)+dx*at(x0+1,y0+1));
        }
        const k=(y*heat.width+x)*4,c=color(value);rgba.data[k]=c[0];rgba.data[k+1]=c[1];rgba.data[k+2]=c[2];rgba.data[k+3]=255;
      }
      hc.putImageData(rgba,0,0);ctx.globalAlpha=clamp(options.alpha??.55,0,1);
      ctx.imageSmoothingEnabled=style!=='blocks';
      ctx.drawImage(heat,0,0,canvas.width*cols*16/width,canvas.height*rows*16/height);ctx.globalAlpha=1;
      Object.assign(result,{available:true,low,high,peak:Math.max(...data),peakIndex:data.indexOf(Math.max(...data))});
    }
    if (options.grid && Number.isInteger(rows)) {
      ctx.strokeStyle='rgba(255,255,255,.45)';ctx.lineWidth=1;
      for(let x=0;x<=cols;x++){ctx.beginPath();ctx.moveTo(x*16/width*canvas.width,0);ctx.lineTo(x*16/width*canvas.width,rows*16/height*canvas.height);ctx.stroke();}
      for(let y=0;y<=rows;y++){ctx.beginPath();ctx.moveTo(0,y*16/height*canvas.height);ctx.lineTo(cols*16/width*canvas.width,y*16/height*canvas.height);ctx.stroke();}
    }
    return result;
  }
  function describe(result) {
    if (!result.available) return result.choice==='none'?'原始策略图像':'该帧没有所选热力图数据';
    return `原始 ${result.rows}×${result.cols} · 显示 ${result.low.toPrecision(3)}～${result.high.toPrecision(3)} · ${result.style==='blocks'?'原始网格':'双线性平滑'} · 峰值 patch ${result.peakIndex}`;
  }
  root.VibeHeatmap={draw,describe,values,color};
})(globalThis);
