'use strict';
(() => {
  const content=window.SCDM;
  let language='en', methodIndex=0, datasetIndex=2, galleryIndex=0, metric='auprc';
  const $=selector=>document.querySelector(selector);
  const $$=selector=>Array.from(document.querySelectorAll(selector));
  const text=value=>value[language];
  const htmlEscape=value=>String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const originalText=new Map($$('[data-i18n]').map(el=>[el,el.innerHTML]));
  const format=value=>value.toFixed(3);
  $('#authors').innerHTML=content.authors.map(author=>{const name=`${htmlEscape(author.name)}${author.corresponding?'<sup title="Corresponding author">*</sup>':''}`;return author.url?`<a href="${author.url}" target="_blank" rel="noopener">${name}</a>`:`<span class="author-name">${name}</span>`;}).join('');
  $('#bibtex').textContent=content.bibtex;

  function updateDimension(){
    const d=Number($('#dimension').value);
    $('#edge-count').textContent=(d*(d-1)).toLocaleString('en-US');
    $('#variable-count').textContent=d.toLocaleString('en-US');
    $$('.presets button').forEach(button=>{const selected=Number(button.dataset.dimension)===d;button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));});
  }
  $('#dimension').addEventListener('input',updateDimension);
  $$('.presets button').forEach(button=>button.addEventListener('click',()=>{$('#dimension').value=button.dataset.dimension;updateDimension();}));

  function renderMethod(){
    $('#method-tabs').innerHTML=content.methods.map((step,index)=>`<button id="method-tab-${index}" role="tab" aria-selected="${index===methodIndex}" aria-controls="step-panel" tabindex="${index===methodIndex?0:-1}" data-method="${index}"><span>0${index+1}</span><span><strong>${text(step.title)}</strong><small>${text(step.subtitle)}</small></span></button>`).join('');
    const step=content.methods[methodIndex];
    $('#step-panel').setAttribute('aria-labelledby',`method-tab-${methodIndex}`);
    $('#step-tag').textContent=language==='en'?`STEP 0${methodIndex+1} / 04`:`步骤 0${methodIndex+1} / 04`;
    $('#step-title').textContent=text(step.heading);
    $('#step-copy').textContent=text(step.copy);
    $('#step-equation').innerHTML=step.equation;
    $('#step-diagram').innerHTML=text(step.blocks).map((label,index)=>`${index?'<span class="arrow">→</span>':''}<span class="mini-block ${index===1?'emphasis':''}">${label}</span>`).join('');
    $$('.step-pagination i').forEach((dot,index)=>dot.classList.toggle('current',index===methodIndex));
  }
  function renderResults(){
    const dataset=content.datasets[datasetIndex];
    $('#dataset-tabs').innerHTML=content.datasets.map((entry,index)=>`<button id="dataset-tab-${index}" role="tab" aria-controls="result-panel" aria-selected="${index===datasetIndex}" tabindex="${index===datasetIndex?0:-1}" data-dataset="${index}">${text(entry.name)}</button>`).join('');
    $('#result-panel').setAttribute('aria-labelledby',`dataset-tab-${datasetIndex}`);
    $('#chart-title').textContent=`${text(dataset.name)} / ${metric.toUpperCase()}`;
    $('#result-chart').innerHTML=content.results.map(row=>{
      const [mean,ci]=row[dataset.key][metric];
      const left=Math.max(0,mean-ci)*100, right=Math.min(1,mean+ci)*100;
      return `<div class="bar-row ${row.name==='SCDM'?'ours':''}"><span class="bar-label">${row.name}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${mean*100}%"></div><div class="ci-line" style="left:${left}%;width:${right-left}%"></div></div><span class="bar-value"><strong>${format(mean)}</strong> ± ${format(ci)}</span></div>`;
    }).join('');
    $('#result-takeaway').textContent=text(dataset.takeaway);
    const header=`<tr><th rowspan="2" scope="col">${language==='en'?'Method':'方法'}</th>${content.datasets.map(entry=>`<th colspan="2" scope="colgroup">${text(entry.name)}</th>`).join('')}</tr><tr>${content.datasets.map(()=>'<th scope="col">AUROC ↑</th><th scope="col">AUPRC ↑</th>').join('')}</tr>`;
    $('#results-table thead').innerHTML=header;
    $('#results-table tbody').innerHTML=content.results.map(row=>`<tr class="${row.name==='SCDM'?'ours':''}"><th scope="row">${row.name}</th>${content.datasets.map(entry=>['auroc','auprc'].map(key=>{
      const [mean,ci]=row[entry.key][key],best=Math.max(...content.results.map(r=>r[entry.key][key][0]));
      return `<td class="${mean===best?'best':''}">${format(mean)} ± ${format(ci)}</td>`;
    }).join('')).join('')}</tr>`).join('');
  }
  function renderGallery(){
    const item=content.gallery[galleryIndex];
    $('#gallery-tabs').innerHTML=content.gallery.map((entry,index)=>`<button id="gallery-tab-${index}" role="tab" aria-controls="gallery-panel" aria-selected="${index===galleryIndex}" tabindex="${index===galleryIndex?0:-1}" data-gallery="${index}">${text(entry.tab)}</button>`).join('');
    $('#gallery-panel').setAttribute('aria-labelledby',`gallery-tab-${galleryIndex}`);
    $('#gallery-type').textContent=text(item.type);
    $('#gallery-title').textContent=text(item.title);
    $('#gallery-caption').textContent=text(item.caption);
    if(galleryIndex===2){const a=document.createElement('a');a.href='https://www.openstreetmap.org/copyright';a.target='_blank';a.rel='noopener';a.textContent=language==='en'?' Map attribution ↗':' 地图版权说明 ↗';$('#gallery-caption').append(a);}
    $('#gallery-boundary').textContent=text(item.boundary);
    const image=$('#gallery-image');image.src=item.file;image.alt=text(item.caption);image.width=item.width;image.height=item.height;
    for(const button of [$('#gallery-zoom'),$('#gallery-image-button')]){button.dataset.image=item.file;button.dataset.alt=text(item.title);button.setAttribute('aria-label',(language==='en'?'Expand: ':'放大：')+text(item.title));}
  }
  function renderLanguage(){
    document.documentElement.lang=language==='en'?'en':'zh-CN';
    $$('[data-i18n]').forEach(el=>{el.innerHTML=language==='en'?originalText.get(el):content.zh[el.dataset.i18n]||originalText.get(el);});
    $('h1').innerHTML=text(content.title);
    $('#abstract-text').textContent=text(content.abstract);
    $('#language').innerHTML=language==='en'?'中文 <span aria-hidden="true">↗</span>':'EN <span aria-hidden="true">↗</span>';
    $('#language').setAttribute('aria-label',language==='en'?'Switch to Chinese':'Switch to English');
    $('#close-dialog').setAttribute('aria-label',language==='en'?'Close enlarged figure':'关闭放大图片');
    renderMethod();renderResults();renderGallery();
  }
  $('#language').addEventListener('click',()=>{language=language==='en'?'zh':'en';renderLanguage();});
  function delegatedTabs(selector,key,render){
    $(selector).addEventListener('click',event=>{
      const button=event.target.closest(`[data-${key}]`);if(!button)return;
      const index=Number(button.dataset[key]);
      if(key==='method')methodIndex=index;if(key==='dataset')datasetIndex=index;if(key==='gallery')galleryIndex=index;
      render();$(`${selector} [data-${key}="${index}"]`).focus({preventScroll:true});
    });
    $(selector).addEventListener('keydown',event=>{
      const buttons=$$(`${selector} button`),current=buttons.indexOf(event.target);
      if(current<0)return;let next;
      if(['ArrowRight','ArrowDown'].includes(event.key))next=(current+1)%buttons.length;
      if(['ArrowLeft','ArrowUp'].includes(event.key))next=(current+buttons.length-1)%buttons.length;
      if(event.key==='Home')next=0;if(event.key==='End')next=buttons.length-1;
      if(next!==undefined){event.preventDefault();buttons[next].click();}
    });
  }
  delegatedTabs('#method-tabs','method',renderMethod);delegatedTabs('#dataset-tabs','dataset',renderResults);delegatedTabs('#gallery-tabs','gallery',renderGallery);
  $$('input[name=metric]').forEach(radio=>radio.addEventListener('change',()=>{metric=radio.value;renderResults();}));
  const dialog=$('#image-dialog');let previousFocus=null;
  document.addEventListener('click',event=>{const trigger=event.target.closest('[data-image]');if(!trigger)return;previousFocus=trigger;$('#enlarged-image').src=trigger.dataset.image;$('#enlarged-image').alt=trigger.dataset.alt;$('#image-caption').textContent=trigger.dataset.alt;$('#image-download').href=trigger.dataset.image;dialog.showModal();document.body.classList.add('dialog-open');});
  $('#close-dialog').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{if(event.target!==dialog)return;const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();});
  dialog.addEventListener('close',()=>{document.body.classList.remove('dialog-open');previousFocus?.focus({preventScroll:true});});
  let toastTimer;
  function toast(message){$('#toast').textContent=message;$('#toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),2400);}
  $('#copy-citation').addEventListener('click',async()=>{
    let success=false;
    if(navigator.clipboard&&window.isSecureContext){try{await navigator.clipboard.writeText(content.bibtex);success=true;}catch{}}
    if(!success){const textarea=document.createElement('textarea');textarea.value=content.bibtex;textarea.style.cssText='position:fixed;left:-9999px;top:0';document.body.append(textarea);textarea.select();try{success=document.execCommand('copy');}catch{}textarea.remove();$('#copy-citation').focus({preventScroll:true});}
    toast(success?(language==='en'?'BibTeX copied to clipboard':'BibTeX 已复制'):(language==='en'?'Select the citation text to copy it manually':'请选择引用文本手动复制'));
  });
  function schematic(){
    let signals='';for(let row=0;row<4;row++){let d='';for(let x=0;x<=276;x+=3){const y=25+row*32+Math.sin(x/12+row*2)*7+Math.sin(x/5.5+row)*3;d+=`${x?'L':'M'}${x+5},${y.toFixed(2)} `;}signals+=`<path d="${d}" fill="none" stroke="${['#357a71','#789fb8','#5e8d59','#6f92a1'][row]}" stroke-width="1.5"/>`;}
    $('#signals').innerHTML=signals;
    let matrix='';for(let r=0;r<7;r++)for(let c=0;c<7;c++){const opacity=(r===c?.12:((r*11+c*7)%17)/20+.08);matrix+=`<rect x="${15+c*18}" y="${10+r*18}" width="14" height="14" rx="2" fill="#397f76" opacity="${opacity}"/>`;}
    $('#matrix').innerHTML=matrix;
    const nodes=[[43,43],[118,19],[214,36],[267,91],[180,126],[78,127],[137,78]];
    const links=[[0,1],[0,5],[1,2],[1,6],[2,3],[3,4],[6,2],[6,4],[5,6]];
    let graph='<defs><marker id="arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#527a76"/></marker></defs>';
    for(const [a,b] of links){const [ax,ay]=nodes[a],[bx,by]=nodes[b],dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy);graph+=`<line x1="${ax+dx/len*8}" y1="${ay+dy/len*8}" x2="${bx-dx/len*10}" y2="${by-dy/len*10}" stroke="#527a76" stroke-width="1.1" opacity=".7" marker-end="url(#arrowhead)"/>`;}
    nodes.forEach(([x,y],i)=>graph+=`<circle cx="${x}" cy="${y}" r="${i===6?8:6}" fill="${i===6?'#26695f':'#669f93'}"/><text x="${x}" y="${y+22}" text-anchor="middle" fill="#536d70" font-size="9" font-family="monospace">x${i+1}</text>`);
    $('#network').innerHTML=graph;
  }
  function updateProgress(){const max=document.documentElement.scrollHeight-innerHeight;$('#read-progress').style.width=`${max>0?scrollY/max*100:0}%`;}
  addEventListener('scroll',updateProgress,{passive:true});addEventListener('resize',updateProgress);
  if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){$$('nav a').forEach(link=>link.classList.toggle('active',link.hash===`#${entry.target.id}`));}}},{rootMargin:'-15% 0px -60% 0px'});$$('main section[id]').forEach(section=>observer.observe(section));}
  schematic();updateDimension();renderLanguage();updateProgress();window.__scdmReady=true;
})();
