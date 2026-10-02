document.addEventListener('DOMContentLoaded',()=>{
  const note=document.querySelector('.dataset-note'), button=document.querySelector('#langToggle');
  const en='<strong>Open data release.</strong> We release the processed evolving-network datasets <strong>PEMSD4(L)</strong> and <strong>PEMSD8(M)</strong>, with documentation, through the Dataset link.';
  const cn='<strong>开放数据发布。</strong> 我们通过 Dataset 链接公开发布了处理后的演化网络数据集 <strong>PEMSD4(L)</strong> 和 <strong>PEMSD8(M)</strong>，并提供相应的数据说明文档。';
  const network=document.querySelector('.network-copy p');
  const networkEn='Interactive schematic anchored to <strong>PEMSD3(S), 2011–2017</strong>. Its yearly periods and selected-update counts follow the paper; node positions and links are illustrative rather than the released raw sensor topology.';
  const networkCn='交互式示意基于 <strong>PEMSD3(S), 2011–2017</strong>。年度时期与所选更新节点数遵循论文；节点位置与连线用于机制说明，并非发布的原始传感器拓扑。';
  const abstractParagraphs=document.querySelectorAll('.abstract-card > div:first-of-type > p:not(.dataset-note)');
  const abstractEn=[
    'Traffic networks evolve as sensors are added, removed, and shift in distribution. CoMemNet selectively updates changed nodes while carrying compact temporal memory forward. Forecasting is adjacency-free; topology only optionally expands local updates.',
    'Across three evolving PeMS benchmarks, CoMemNet delivers accurate 15-, 30-, and 60-minute forecasts with bounded updates.'
  ];
  const abstractCn=[
    '交通网络会因传感器新增、移除和分布变化而不断演化。CoMemNet 选择性更新变化节点，并持续复用紧凑的时间记忆。预测不依赖邻接矩阵；拓扑信息仅可选地扩展局部更新。',
    '在三个演化 PeMS 基准上，CoMemNet 以有界更新实现准确的 15、30 和 60 分钟预测。'
  ];
  const rho=document.querySelector('.rho-insight');
  const rhoEn='<strong>How to read this table</strong><p>Each cell is selected nodes / 12-step MAE. A larger ρ generally selects more shared nodes and can improve current-period accuracy, but increases update cost. The highlighted ρ=0.05 is the CoMemNet operating point used in the paper.</p><div><span>Lower budget</span><b>efficiency ↔ accuracy</b><span>Higher budget</span></div>';
  const rhoCn='<strong>如何阅读此表</strong><p>每个单元格为“所选节点数 / 12 步 MAE”。更大的 ρ 通常选择更多共享节点，并可能提升当前时期精度，但会增加更新成本。高亮的 ρ=0.05 是论文中 CoMemNet 的工作点。</p><div><span>较低预算</span><b>效率 ↔ 精度</b><span>较高预算</span></div>';
  const extendedEn=['Controlled evidence,<br><em>not just one table.</em>','Additional revision experiments quantify the update budget, temporal-memory aggregation, distribution shifts, and controlled update-policy choices.','ρ budget sensitivity · PEMSD3(S)','Selected-node count and 12-step MAE across evolving periods. The operating point used by CoMemNet is highlighted.','K<sub>mem</sub> sensitivity','Temporal-memory aggregation remains stable around K<sub>mem</sub>=12.','Annual-average 12-step MAE over tested temporal-memory aggregation sizes.','Sampler and update-policy controls · PEMSD3(S)','The controlled study compares whether a bounded subset alone is sufficient (Random), whether recent error or recency explains selection, alternative feature distances (L2 / KL / JS / MMD), target branches and momenta, and selected-only versus 1/2/3-hop update neighborhoods.'];
  const extendedCn=['控制性证据，<br><em>不止一张表。</em>','额外修订实验量化了更新预算、时间记忆聚合、分布变化与更新策略控制。','ρ 预算敏感性 · PEMSD3(S)','展示演化时期中所选节点数与 12 步 MAE；高亮 CoMemNet 使用的工作点。','K<sub>mem</sub> 敏感性','时间记忆聚合在 K<sub>mem</sub>=12 附近保持稳定。','不同时间记忆聚合大小下的年度平均 12 步 MAE。','采样器与更新策略控制 · PEMSD3(S)','控制实验比较有界子集是否足够（Random）、近期误差或时效性是否能解释选择、替代特征距离（L2 / KL / JS / MMD）、目标分支与动量，以及仅选中节点与 1/2/3-hop 更新邻域。'];
  const extendedTargets=[document.querySelector('.extended .heading h2'),document.querySelector('.extended .heading p'),document.querySelector('.experiment-card:first-child h3'),document.querySelector('.experiment-card:first-child>p'),document.querySelector('.experiment-card:nth-child(2) h3'),document.querySelector('.experiment-card:nth-child(2)>p'),document.querySelector('.experiment-card figcaption'),document.querySelector('.control-summary h3'),document.querySelector('.control-summary p')];
  const captions=['PEMSD3 · 2012','PEMSD3 · 2014','PEMSD3 · 2017','PEMSD4 · 2010','PEMSD4 · 2012','PEMSD4 · 2013','PEMSD8 · 2013','PEMSD8 · 2016','PEMSD8 · 2018'];
  const captionsCn=['PEMSD3 · 2012 年','PEMSD3 · 2014 年','PEMSD3 · 2017 年','PEMSD4 · 2010 年','PEMSD4 · 2012 年','PEMSD4 · 2013 年','PEMSD8 · 2013 年','PEMSD8 · 2016 年','PEMSD8 · 2018 年'];
  button.addEventListener('click',()=>{setTimeout(()=>{const chinese=button.textContent==='EN';note.innerHTML=chinese?cn:en;network.innerHTML=chinese?networkCn:networkEn;abstractParagraphs.forEach((p,i)=>p.textContent=(chinese?abstractCn:abstractEn)[i]);rho.innerHTML=chinese?rhoCn:rhoEn;extendedTargets.forEach((el,i)=>el.innerHTML=(chinese?extendedCn:extendedEn)[i]);document.querySelectorAll('.distribution-grid figcaption').forEach((el,i)=>el.textContent=(chinese?captionsCn:captions)[i])},0)});
});
