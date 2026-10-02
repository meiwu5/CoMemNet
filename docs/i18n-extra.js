document.addEventListener('DOMContentLoaded',()=>{
  const note=document.querySelector('.dataset-note'), button=document.querySelector('#langToggle');
  const en='<strong>Open data release.</strong> We release the processed evolving-network datasets <strong>PEMSD4(L)</strong> and <strong>PEMSD8(M)</strong>, with documentation, through the Dataset link.';
  const cn='<strong>开放数据发布。</strong> 我们通过 Dataset 链接公开发布了处理后的演化网络数据集 <strong>PEMSD4(L)</strong> 和 <strong>PEMSD8(M)</strong>，并提供相应的数据说明文档。';
  const network=document.querySelector('.network-copy p');
  const networkEn='Interactive schematic anchored to <strong>PEMSD3(S), 2011–2017</strong>. Its yearly periods and selected-update counts follow the paper; node positions and links are illustrative rather than the released raw sensor topology.';
  const networkCn='交互式示意基于 <strong>PEMSD3(S), 2011–2017</strong>。年度时期与所选更新节点数遵循论文；节点位置与连线用于机制说明，并非发布的原始传感器拓扑。';
  button.addEventListener('click',()=>{setTimeout(()=>{const chinese=button.textContent==='EN';note.innerHTML=chinese?cn:en;network.innerHTML=chinese?networkCn:networkEn},0)});
});
