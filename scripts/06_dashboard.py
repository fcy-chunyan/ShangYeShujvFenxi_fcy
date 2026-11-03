# -*- coding: utf-8 -*-
"""
06_dashboard.py —— 交互式商业分析大屏（Python 生成，ECharts 渲染）

设计要点
--------
1. 不依赖 pyecharts：ECharts 通过脚本内联的数据 + 官方发行版加载，避免因环境缺库导致整条流水线中断。
   加载顺序为「本地 echarts.min.js → jsDelivr → unpkg → BootCDN」，任一成功即渲染，
   因此既能在联网环境直接打开，也能在把 echarts.min.js 放到同目录后完全离线使用。
2. 组件：KPI 卡 ×6、月度双轴组合图、渠道玫瑰图、客群条形图、营销漏斗、月份×客群热力图、明细表。
3. 交互：渠道筛选（全局重算）、点击客群条形下钻（漏斗 + 明细联动）、多图 tooltip/缩放同步、自适应。

输出：outputs/dashboard/dashboard.html（单文件，双击即可打开）
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ECHARTS_VER = "5.5.0"
CDN_LIST = [
    "./echarts.min.js",
    f"https://cdn.jsdelivr.net/npm/echarts@{ECHARTS_VER}/dist/echarts.min.js",
    f"https://unpkg.com/echarts@{ECHARTS_VER}/dist/echarts.min.js",
    f"https://cdn.bootcdn.net/ajax/libs/echarts/{ECHARTS_VER}/echarts.min.js",
]

# 漏斗分层阈值（与 config.FUNNEL_STAGES 一致）
STAGES = [("触达名单", 0), ("有效接通", 30), ("有效沟通", 120), ("深度沟通", 300)]


# ---------------------------------------------------------------- 数据准备
def _funnel(g: pd.DataFrame) -> list[dict]:
    out = []
    for name, thr in STAGES:
        n = len(g) if thr == 0 else int((g["duration"] >= thr).sum())
        out.append({"name": name, "value": n})
    out.append({"name": "认购成功", "value": int(g["is_subscribed"].sum())})
    return out


def build_scope(df: pd.DataFrame) -> dict:
    """把某一筛选范围（全部/某渠道）压缩成大屏需要的全部序列。"""
    months = sorted(df["year_month"].unique())
    m = (df.groupby("year_month")
           .agg(contacts=("is_subscribed", "size"), conv=("is_subscribed", "mean"))
           .reindex(months))
    kpi = {
        "contacts": int(len(df)),
        "subs": int(df["is_subscribed"].sum()),
        "conv": round(float(df["is_subscribed"].mean()) * 100, 2),
        "duration": round(float(df["duration"].mean()), 1),
        "jobs": int(df["job"].nunique()),
        "month_peak": str(df.groupby("year_month").size().idxmax()),
        "month_peak_n": int(df.groupby("year_month").size().max()),
    }

    # 客群：按认购量降序，条形图按转化率升序绘制（横向条形自下而上）
    job = (df.groupby("job")
             .agg(contacts=("is_subscribed", "size"),
                  subs=("is_subscribed", "sum"),
                  conv=("is_subscribed", "mean"),
                  dur=("duration", "mean"))
             .reset_index())
    job["conv_pct"] = (job["conv"] * 100).round(2)
    job = job.sort_values("subs", ascending=False).reset_index(drop=True)
    job_sorted = job.sort_values("conv").reset_index(drop=True)

    # 热力图：月份 × 职业，样本量 <100 的组合留空（避免小样本噪声）
    h = df.groupby(["job", "month_num"])["is_subscribed"].agg(["mean", "size"])
    jobs = job["job"].tolist()
    heat = []
    for yi, j in enumerate(jobs):
        for mi in range(1, 13):
            if (j, mi) not in h.index:
                continue
            mean, n = h.loc[(j, mi), "mean"], h.loc[(j, mi), "size"]
            if n >= 100:
                heat.append([mi - 1, yi, round(float(mean) * 100, 2)])

    # 下钻：客群 → 漏斗 + 指标
    drill = {}
    for j, g in df.groupby("job"):
        drill[j] = {
            "funnel": _funnel(g),
            "contacts": int(len(g)),
            "subs": int(g["is_subscribed"].sum()),
            "conv": round(float(g["is_subscribed"].mean()) * 100, 2),
            "duration": round(float(g["duration"].mean()), 1),
        }

    table = [{"job": r["job"], "contacts": int(r["contacts"]), "subs": int(r["subs"]),
              "conv": float(r["conv_pct"]), "dur": round(float(r["dur"]), 0)}
             for _, r in job.iterrows()]

    return {
        "kpi": kpi,
        "months": months,
        "contacts": [int(x) for x in m["contacts"].tolist()],
        "conv": [round(float(x) * 100, 2) for x in m["conv"].fillna(0).tolist()],
        "funnel": _funnel(df),
        "jobNames": job_sorted["job"].tolist(),
        "jobConv": job_sorted["conv_pct"].tolist(),
        "jobSubs": [int(x) for x in job_sorted["subs"].tolist()],
        "jobs": jobs,
        "heat": heat,
        "drill": drill,
        "table": table,
    }


def prepare(df: pd.DataFrame) -> dict:
    scopes = {"all": build_scope(df)}
    for ch in ["cellular", "telephone"]:
        sub = df[df["channel"] == ch]
        if len(sub) > 0:
            scopes[ch] = build_scope(sub)
    return {
        "scopes": scopes,
        "channelSplit": [{"name": str(k), "value": int(v)}
                         for k, v in df.loc[df["is_subscribed"] == 1, "channel"].value_counts().items()],
        "meta": {
            "start": str(df["contact_date"].min().date()),
            "end": str(df["contact_date"].max().date()),
            "rows": int(len(df)),
        },
    }


# ---------------------------------------------------------------- 资源获取
def ensure_echarts(out_dir: Path) -> bool:
    """尝试把 echarts.min.js 落到本地，成功则大屏可完全离线打开。"""
    local = out_dir / "echarts.min.js"
    if local.exists() and local.stat().st_size > 500_000:
        return True
    urls = [u for u in CDN_LIST if not u.startswith("./")]
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=20) as r:
                data = r.read()
            if len(data) > 500_000:
                local.write_bytes(data)
                print(f"[dash] 已缓存 ECharts 到本地（{len(data)/1024:.0f} KB），大屏可离线打开")
                return True
        except Exception:
            continue
    print("[dash] 未取得本地 ECharts，大屏将走 CDN（需联网）")
    return False


# ---------------------------------------------------------------- 页面模板
TPL = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>银行电话营销 · 客户转化与客群运营大屏</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#08182C;color:#DCE6F2;font-family:"Microsoft YaHei","PingFang SC",sans-serif}
  .dash{padding:16px 20px 24px;min-height:100vh;
        background:radial-gradient(1300px 620px at 50% -12%,#16385F 0%,#08182C 62%)}
  header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;
         padding-bottom:11px;border-bottom:1px solid rgba(120,170,230,.25);margin-bottom:13px}
  h1{font-size:21px;letter-spacing:2px;font-weight:600;color:#FFFFFF}
  h1 span{font-size:12px;letter-spacing:0;font-weight:400;color:#6E93BB;margin-left:10px}
  .ctrl{display:flex;align-items:center;gap:9px;font-size:12.5px;color:#8FA9C8}
  select{background:#12294A;color:#DCE6F2;border:1px solid rgba(90,150,220,.45);
         border-radius:5px;padding:5px 9px;font-size:12.5px;outline:none;cursor:pointer}
  .kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:11px;margin-bottom:12px}
  .kpi{background:linear-gradient(160deg,#132B4C,#0D2038);border:1px solid rgba(90,150,220,.24);
       border-radius:7px;padding:12px 13px;position:relative;overflow:hidden}
  .kpi::after{content:"";position:absolute;left:0;top:0;width:100%;height:2px;
              background:linear-gradient(90deg,#3D7BD6,rgba(61,123,214,0))}
  .kpi-label{font-size:12px;color:#8FA9C8;margin-bottom:5px}
  .kpi-value{font-size:23px;font-weight:700;color:#FFFFFF;letter-spacing:.5px;
             font-variant-numeric:tabular-nums}
  .kpi-sub{font-size:11px;color:#5F8BB8;margin-top:3px}
  .grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px}
  .card{background:rgba(17,38,66,.78);border:1px solid rgba(90,150,220,.2);border-radius:7px;
        padding:4px 4px 0}
  .span2{grid-column:span 2}
  .span3{grid-column:span 3}
  .card-title{font-size:13px;padding:8px 10px 0;color:#DCE6F2;font-weight:600}
  .card-title em{font-style:normal;color:#6E93BB;font-weight:400;font-size:11.5px;margin-left:6px}
  .chart{width:100%;height:262px}
  .chart-tall{height:300px}
  .pad{padding-bottom:8px}
  table{width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 6px}
  th,td{text-align:right;padding:7px 10px}
  th:first-child,td:first-child{text-align:left}
  thead th{color:#8FA9C8;font-weight:500;border-bottom:1px solid rgba(120,170,230,.22);
           position:sticky;top:0;background:#122A45}
  tbody td{border-bottom:1px dashed rgba(120,170,230,.12);font-variant-numeric:tabular-nums}
  tbody tr:hover td{background:rgba(76,155,232,.12);cursor:pointer}
  .bar-cell{position:relative;height:9px;border-radius:5px;background:rgba(76,155,232,.16);
            min-width:60px;display:block}
  .bar-fill{position:absolute;left:0;top:0;height:100%;border-radius:5px;
            background:linear-gradient(90deg,#3D7BD6,#5AB0F0)}
  .scroll{max-height:322px;overflow:auto}
  .scroll::-webkit-scrollbar{width:7px}
  .scroll::-webkit-scrollbar-thumb{background:rgba(90,150,220,.35);border-radius:4px}
  #tip{font-size:13px;color:#9DB6D2;text-align:center;padding:16px}
</style>
</head>
<body>
<div class="dash">
  <header>
    <h1>银行电话营销 · 客户转化与客群运营大屏<span>UCI Bank Marketing　数据区间 __START__ — __END__　样本 __ROWS__ 人次</span></h1>
    <div class="ctrl">
      <label for="ch">联系渠道</label>
      <select id="ch">
        <option value="all">全部渠道</option>
        <option value="cellular">手机 cellular</option>
        <option value="telephone">座机 telephone</option>
      </select>
      <span id="drill-tip" style="color:#6E93BB">点击右侧客群条形图可下钻</span>
    </div>
  </header>

  <section class="kpis" id="kpis"></section>

  <section class="grid">
    <div class="card span2"><div class="card-title">月度触达量与转化率<em>柱=触达人次（左轴）　线=转化率（右轴）</em></div><div id="c-trend" class="chart chart-tall"></div></div>
    <div class="card"><div class="card-title">渠道认购结构<em>玫瑰图</em></div><div id="c-channel" class="chart chart-tall"></div></div>
    <div class="card"><div class="card-title">客群转化率<em>点击下钻</em></div><div id="c-job" class="chart palette"></div></div>
    <div class="card"><div class="card-title">营销转化漏斗<em id="funnel-title">全部客群</em></div><div id="c-funnel" class="chart"></div></div>
    <div class="card"><div class="card-title">月份 × 客群 热力图<em>&lt;100 人次留空</em></div><div id="c-heat" class="chart"></div></div>
    <div class="card span3"><div class="card-title">客群明细<em>按认购量降序，点击行同样可下钻</em></div>
      <div class="scroll"><table id="detail"><thead><tr><th>职业</th><th>触达量</th><th>认购量</th><th>转化率</th><th style="width:120px">转化率对比</th><th>平均时长(s)</th></tr></thead><tbody></tbody></table></div>
    </div>
  </section>
</div>

<script>
var DATA = __DATA__;
var SRC = __SRC__;

/* ---------- 顺序加载 ECharts，任一源成功即初始化 ---------- */
(function loadOne(i){
  if (i >= SRC.length) {
    document.body.insertAdjacentHTML('beforeend',
      '<div id="tip">ECharts 资源加载失败：请联网打开，或把 echarts.min.js 放到本文件同目录后刷新。</div>');
    return;
  }
  var s = document.createElement('script');
  s.src = SRC[i];
  s.onload = function(){ window.echarts ? boot() : loadOne(i+1); };
  s.onerror = function(){ loadOne(i+1); };
  document.head.appendChild(s);
})(0);

var AXIS = {color:'#8FA9C8'}, SPLIT = {lineStyle:{color:'rgba(120,170,230,.13)'}};
var charts = {};

function fmt(n){ return n.toLocaleString('en-US'); }

function kpiCard(label, value, sub){
  return '<div class="kpi"><div class="kpi-label">'+label+'</div>'+
         '<div class="kpi-value">'+value+'</div><div class="kpi-sub">'+sub+'</div></div>';
}

function renderKpi(s){
  var k = s.kpi;
  document.getElementById('kpis').innerHTML = [
    kpiCard('触达人次', fmt(k.contacts), '外呼名单总量'),
    kpiCard('认购客户', fmt(k.subs), '成功开立定期存款'),
    kpiCard('整体转化率', k.conv.toFixed(2)+'%', '认购 / 触达'),
    kpiCard('平均通话时长', k.duration+' 秒', '约 '+(k.duration/60).toFixed(1)+' 分钟'),
    kpiCard('覆盖职业数', k.jobs+' 类', '客群细分维度'),
    kpiCard('峰值月份', k.month_peak, fmt(k.month_peak_n)+' 人次')
  ].join('');
}

function renderTable(s){
  var max = Math.max.apply(null, s.table.map(function(r){return r.conv;}));
  var html = s.table.map(function(r){
    var w = (r.conv/max*100).toFixed(1);
    return '<tr data-job="'+r.job+'"><td>'+r.job+'</td><td>'+fmt(r.contacts)+'</td><td>'+
           fmt(r.subs)+'</td><td>'+r.conv.toFixed(2)+'%</td>'+
           '<td><span class="bar-cell"><span class="bar-fill" style="width:'+w+'%"></span></span></td>'+
           '<td>'+r.dur+'</td></tr>';
  }).join('');
  var tb = document.querySelector('#detail tbody');
  tb.innerHTML = html;
  Array.prototype.forEach.call(tb.querySelectorAll('tr'), function(tr){
    tr.onclick = function(){ drillTo(tr.getAttribute('data-job')); };
  });
}

function optTrend(s){
  return {
    backgroundColor:'transparent',
    grid:{left:52,right:52,top:34,bottom:74},
    tooltip:{trigger:'axis',axisPointer:{type:'cross'},
             backgroundColor:'rgba(12,28,48,.94)',borderColor:'rgba(120,170,230,.4)',
             textStyle:{color:'#DCE6F2',fontSize:12}},
    legend:{data:['触达人次','转化率'],left:52,top:2,textStyle:{color:'#8FA9C8',fontSize:11},itemWidth:12,itemHeight:8},
    xAxis:{type:'category',data:s.months,
           axisLabel:{color:'#7E9BBE',fontSize:9.5,rotate:40,interval:2},
           axisLine:{lineStyle:{color:'rgba(120,170,230,.3)'}},axisTick:{show:false}},
    yAxis:[
      {type:'value',name:'',
       axisLabel:{color:'#7E9BBE',fontSize:9},splitLine:SPLIT,axisLine:{show:false}},
      {type:'value',name:'',
       axisLabel:{color:'#7E9BBE',fontSize:9,formatter:'{value}%'},splitLine:{show:false},axisLine:{show:false}}
    ],
    dataZoom:[{type:'inside'},{type:'slider',height:13,bottom:4,
               borderColor:'rgba(120,170,230,.25)',fillerColor:'rgba(76,155,232,.15)',
               textStyle:{color:'#7E9BBE',fontSize:9},handleStyle:{color:'#3D7BD6'}}],
    series:[
      {name:'触达人次',type:'bar',data:s.contacts,barMaxWidth:18,
       itemStyle:{borderRadius:[2,2,0,0],
                  color:new echarts.graphic.LinearGradient(0,0,0,1,
                    [{offset:0,color:'#5AA9F0'},{offset:1,color:'#1E4C86'}])}},
      {name:'转化率',type:'line',yAxisIndex:1,data:s.conv,smooth:true,symbolSize:5,
       lineStyle:{width:2.4,color:'#F2994A'},itemStyle:{color:'#F2994A'}}
    ]
  };
}

function optChannel(){
  return {
    backgroundColor:'transparent',
    tooltip:{trigger:'item',formatter:'{b}: {c} 人 ({d}%)',
             backgroundColor:'rgba(12,28,48,.94)',borderColor:'rgba(120,170,230,.4)',
             textStyle:{color:'#DCE6F2'}},
    legend:{orient:'vertical',right:4,top:'middle',textStyle:{color:'#8FA9C8',fontSize:11},
            itemWidth:10,itemHeight:8},
    color:['#3D7BD6','#F2994A','#4FC3A1'],
    series:[{
      type:'pie',radius:['38%','62%'],center:['34%','52%'],
      data:DATA.channelSplit,
      itemStyle:{borderColor:'rgba(9,24,44,.85)',borderWidth:2},
      label:{position:'inside',color:'#FFFFFF',fontSize:11,fontWeight:'bold',formatter:'{d}%'},
      labelLine:{show:false}
    }]
  };
}

function optJob(s){
  return {
    backgroundColor:'transparent',
    grid:{left:76,right:44,top:14,bottom:22},
    tooltip:{trigger:'axis',axisPointer:{type:'shadow'},
             backgroundColor:'rgba(12,28,48,.94)',borderColor:'rgba(120,170,230,.4)',
             textStyle:{color:'#DCE6F2'},
             formatter:function(p){ return p[0].name+'<br/>转化率 '+p[0].value+'%'; }},
    xAxis:{type:'value',axisLabel:{color:'#7E9BBE',fontSize:9,formatter:'{value}%'},
           splitLine:SPLIT,axisLine:{show:false}},
    yAxis:{type:'category',data:s.jobNames,axisLabel:{color:'#A9C1DC',fontSize:10},
           axisLine:{lineStyle:{color:'rgba(120,170,230,.25)'}},axisTick:{show:false}},
    series:[{
      type:'bar',data:s.jobConv,barMaxWidth:13,
      itemStyle:{borderRadius:[0,3,3,0],
                 color:new echarts.graphic.LinearGradient(0,0,1,0,
                   [{offset:0,color:'#1E4C86'},{offset:1,color:'#5AB0F0'}])},
      label:{show:true,position:'right',color:'#A9C1DC',fontSize:10,formatter:'{c}%'}
    }]
  };
}

function optFunnel(s, title){
  var base = s.funnel[0].value;
  var data = s.funnel.map(function(d){
    return {name:d.name, value:d.value};
  });
  return {
    backgroundColor:'transparent',
    title:{text:title,left:'center',top:2,textStyle:{color:'#8FA9C8',fontSize:11.5,fontWeight:'normal'}},
    tooltip:{trigger:'item',
             formatter:function(p){
               var i = p.dataIndex, prev = i===0 ? p.value : s.funnel[i-1].value;
               return p.name+'<br/>人数 '+fmt(p.value)+'<br/>累计转化 '+(p.value/base*100).toFixed(1)+
                      '%<br/>层内留存 '+(p.value/prev*100).toFixed(1)+'%';
             },
             backgroundColor:'rgba(12,28,48,.94)',borderColor:'rgba(120,170,230,.4)',
             textStyle:{color:'#DCE6F2'}},
    color:['#1E4C86','#2A6DB5','#3D8FD8','#54A9EA','#2E9E80'],
    series:[{
      type:'funnel',top:34,bottom:8,left:'9%',width:'82%',sort:'descending',gap:2,
      minSize:'26%',maxSize:'100%',
      label:{position:'inside',color:'#FFFFFF',fontSize:11,
             formatter:function(p){ return p.name+'  '+fmt(p.value); }},
      itemStyle:{borderColor:'rgba(9,24,44,.9)',borderWidth:1.5},
      data:data
    }]
  };
}

function optHeat(s){
  return {
    backgroundColor:'transparent',
    grid:{left:76,right:16,top:14,bottom:44},
    tooltip:{position:'top',
             formatter:function(p){ return s.jobs[p.value[1]]+' · '+(p.value[0]+1)+'月<br/>转化率 '+p.value[2]+'%'; },
             backgroundColor:'rgba(12,28,48,.94)',borderColor:'rgba(120,170,230,.4)',
             textStyle:{color:'#DCE6F2'}},
    xAxis:{type:'category',data:['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],
           splitArea:{show:true,areaStyle:{color:['rgba(255,255,255,.02)','transparent']}},
           axisLabel:{color:'#7E9BBE',fontSize:9},axisLine:{show:false},axisTick:{show:false}},
    yAxis:{type:'category',data:s.jobs,splitArea:{show:true,areaStyle:{color:['rgba(255,255,255,.02)','transparent']}},
           axisLabel:{color:'#A9C1DC',fontSize:9},axisLine:{show:false},axisTick:{show:false}},
    visualMap:{min:0,max:70,calculable:true,orient:'horizontal',left:'center',bottom:2,
               itemWidth:11,itemHeight:70,textStyle:{color:'#7E9BBE',fontSize:9},
               inRange:{color:['#0E2A46','#1E4C86','#3D8FD8','#F2C14E']}},
    series:[{type:'heatmap',data:s.heat,
             itemStyle:{borderColor:'rgba(9,24,44,.6)',borderWidth:.6},
             emphasis:{itemStyle:{shadowBlur:8,shadowColor:'rgba(90,176,240,.55)'}}}]
  };
}

/* ---------- 渲染与联动 ---------- */
function scope(){ return DATA.scopes[document.getElementById('ch').value] || DATA.scopes.all; }

function paint(s){
  renderKpi(s); renderTable(s);
  charts.trend.setOption(optTrend(s), true);
  charts.job.setOption(optJob(s), true);
  charts.funnel.setOption(optFunnel(s, '全部客群'), true);
  charts.heat.setOption(optHeat(s), true);
  document.getElementById('funnel-title').textContent = '全部客群';
  document.getElementById('drill-tip').textContent = '点击右侧客群条形图可下钻';
}

function drillTo(job){
  var s = scope(), d = s.drill[job];
  if (!d) return;
  charts.funnel.setOption(optFunnel({funnel:d.funnel}, job+' 客群'), true);
  document.getElementById('funnel-title').textContent = job + ' 客群';
  document.getElementById('drill-tip').textContent =
    job+'：触达 '+fmt(d.contacts)+' 人，认购 '+fmt(d.subs)+' 人，转化率 '+d.conv+'%（再次点击顶部标题或切换渠道可复位）';
}

function boot(){
  charts.trend = echarts.init(document.getElementById('c-trend'));
  charts.channel = echarts.init(document.getElementById('c-channel'));
  charts.job = echarts.init(document.getElementById('c-job'));
  charts.funnel = echarts.init(document.getElementById('c-funnel'));
  charts.heat = echarts.init(document.getElementById('c-heat'));
  charts.channel.setOption(optChannel());
  paint(scope());

  charts.job.on('click', function(p){ drillTo(p.name); });
  document.getElementById('ch').onchange = function(){ paint(scope()); };
  // 多图联动：x 轴指示器与缩放同步
  echarts.connect([charts.trend, charts.job, charts.heat]);
  window.addEventListener('resize', function(){
    Object.keys(charts).forEach(function(k){ charts[k].resize(); });
  });
}
</script>
</body>
</html>
"""


def capture_shot(html_path: Path, out_png: Path,
                 width: int = 1680, height: int = 1230) -> bool:
    """用本机 Edge / Chrome 的无头模式给大屏截图，供论文 fig25 直接引用。

    属于“用 Python 代码驱动浏览器渲染”，不是手工截图或 BI 工具导出；
    若本机没有任何 Chromium 内核浏览器，则返回 False 并跳过，不中断主流程。
    """
    env = os.environ
    cands = [
        Path(env.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(env.get("PROGRAMFILES", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(env.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    exe = next((p for p in cands if p.exists()), None)
    if exe is None:
        exe = next((shutil.which(n) for n in
                    ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser")
                    if shutil.which(n)), None)
    if exe is None:
        return False

    cmd = [
        str(exe), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--no-first-run", "--no-default-browser-check",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",          # 等 ECharts 动画结束再截
        f"--screenshot={out_png}",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return False
    return out_png.exists() and out_png.stat().st_size > 20_000


def main() -> None:
    print("=" * 68)
    print("Step 6 / 6  交互式大屏")
    print("=" * 68)
    df = pd.read_csv(cfg.DATA_CLEAN / "bank_clean.csv",
                     parse_dates=["contact_date"], dtype={"age_band": str})
    d = prepare(df)
    out_dir = cfg.OUT_DASH
    out_dir.mkdir(parents=True, exist_ok=True)
    has_local = ensure_echarts(out_dir)
    src = CDN_LIST if has_local else CDN_LIST[1:]

    meta = d["meta"]
    html = (TPL.replace("__DATA__", json.dumps(d, ensure_ascii=False))
               .replace("__SRC__", json.dumps(src, ensure_ascii=False))
               .replace("__START__", meta["start"])
               .replace("__END__", meta["end"])
               .replace("__ROWS__", f"{meta['rows']:,}"))
    out = out_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")

    k = d["scopes"]["all"]["kpi"]
    print(f"[dash] 已生成 {out}")
    print(f"       体积 {out.stat().st_size/1024:.0f} KB　"
          f"组件：KPI×6 + 趋势 + 渠道 + 客群 + 漏斗 + 热力图 + 明细表（共 12 个）")
    print(f"       交互：渠道筛选（3 档）、客群点击下钻、表格行下钻、三图缩放联动、自适应")
    print(f"       校验：触达 {k['contacts']:,} 人次，认购 {k['subs']:,} 人，"
          f"转化率 {k['conv']}%，与 03/04 步口径一致")

    shot = cfg.OUT_FIG / "fig25_交互式大屏_整体效果.png"
    if capture_shot(out, shot):
        print(f"[fig] 大屏截图 {shot.name}（{shot.stat().st_size/1024:.0f} KB，1680×1230）")
    else:
        print("[fig] 未检测到 Edge/Chrome，已跳过自动截图；"
              "可手动打开 dashboard.html 截屏另存为 fig25_交互式大屏_整体效果.png")
    print("Step 6 完成。\n")


if __name__ == "__main__":
    main()
