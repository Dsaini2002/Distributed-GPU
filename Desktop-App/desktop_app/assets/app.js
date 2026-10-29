'use strict';
const $ = id => document.getElementById(id);
const state = {view:'dashboard', role:'host', host:'192.168.50.1', root:'', online:false, last:null,
               models:[], catalogErrors:{},mode:'multi_gpu',activeMode:'multi_gpu',activeJob:null, jobState:null, joinStatus:null, expireAt:0};
let toastTimer=null, refreshing=false, jobPolling=false;

function escaped(v){return String(v??'').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));}
function showToast(message,error=false){const el=$('toast');el.textContent=String(message);el.className='toast show'+(error?' error':'');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.className='toast',6500);}
async function call(method,...params){if(!window.pywebview||!window.pywebview.api){throw Error('Desktop bridge is not ready. Run this interface with the Windows application, not as an HTML file.');}
  const r=await window.pywebview.api[method](...params);if(!r||!r.ok){throw Error((r&&r.error)||'Unexpected desktop service response');}return r.data;}
function safeTask(task){return Promise.resolve().then(task).catch(e=>showToast(e.message||String(e),true));}
function badgeOnline(online){return online?'<span class="dot"></span><span class="device-status">Online</span>':'<span class="dot bad"></span><span class="device-status offline">Offline</span>';}
function view(name){state.view=name;document.querySelectorAll('.view').forEach(e=>e.classList.toggle('active',e.id==='view-'+name));document.querySelectorAll('[data-view]').forEach(e=>e.classList.toggle('selected',e.dataset.view===name));
 const titles={dashboard:['WORKSPACE OVERVIEW','Your GPU workspace'],cluster:['PAIRING & CONNECTIONS','Cluster manager'],train:['PROJECT COMMANDS & GPU DETAILS','Terminal'],models:['SUPPORTED ARCHITECTURES','Model Zoo'],monitor:['REAL-TIME HARDWARE','GPU monitoring'],history:['SAVED CHECKPOINTS','Training history'],settings:['LOCAL APPLICATION','Settings']};
 $('view-kicker').textContent=titles[name][0];$('view-heading').textContent=titles[name][1];
 if(name==='train')safeTask(refreshGpuPanel);
 if(name==='cluster')safeTask(refreshPairing);if(name==='train'||name==='models')safeTask(loadCatalog);if(name==='history')safeTask(loadHistory);
 if(name==='settings'){ $('settings-ip').value=state.host; $('settings-root').value=state.root; } }
function setRoleUI(){const host=state.role==='host';$('cluster-host-tab').classList.toggle('selected',host);$('cluster-worker-tab').classList.toggle('selected',!host);$('host-cluster').classList.toggle('hidden',!host);$('worker-cluster').classList.toggle('hidden',host);$('settings-host').classList.toggle('selected',host);$('settings-worker').classList.toggle('selected',!host);$('join-host').value=state.host;}
async function selectRole(role){const data=await call('settings_update',role,state.host,state.root);state.role=data.role;state.root=data.root;setRoleUI();showToast(role==='host'?'Host mode enabled':'Join mode enabled');await refreshDashboard();await refreshPairing();}
function deviceRow(name,desc,online){return `<div class="device-item"><div class="device-glyph">▣</div><div class="device-text"><strong>${escaped(name)}</strong><small>${escaped(desc)}</small></div><div>${badgeOnline(online)}</div></div>`;}
function renderDashboard(data){state.last=data;state.online=!!data.controller_online;state.host=data.host;state.role=data.role;state.root=data.root;
  setRoleUI();const gpu=data.local_gpu||{};const localGpu=!!gpu.name&&gpu.name!=='GPU status unavailable';const onlineWorkers=(data.workers||[]).filter(x=>x.online);
  const gpuCount=(localGpu?1:0)+onlineWorkers.length;
  $('metric-controller').textContent=data.controller_online?'Connected':'Offline';$('metric-controller-detail').textContent=data.controller_online?data.host+':45400':'Start the background Controller';
  $('metric-gpus').textContent=String(gpuCount);$('metric-training').textContent=state.activeJob&&state.jobState==='running'?'Running':'Idle';$('metric-training-sub').textContent=state.activeJob?'Job '+state.activeJob:'Ready for training';
  const connected=state.role==='host'?data.controller_online:data.agent_online;
  $('top-online').innerHTML='<span class="dot '+(connected?'':'bad')+'"></span> '+(connected?(state.role==='host'?`${gpuCount} GPUs ready`:'Join agent online'):'Service offline');
  let html='';if(localGpu){const detail=gpu.total?`${gpu.used} / ${gpu.total} MiB · ${gpu.util}% usage`:'NVIDIA GPU';html+=deviceRow('This computer · '+gpu.name,detail,true);}else html+=deviceRow('This computer','GPU telemetry unavailable',false);
  if(state.role==='host'){for(const worker of data.workers||[]){html+=deviceRow(worker.name||worker.ip,`${worker.gpu||'Remote GPU'} · ${worker.ip}:${worker.port}`,worker.online);}}
  else if(data.worker_status){html+=deviceRow('Worker service',String(data.worker_status.worker||'stopped'),String(data.worker_status.worker||'').includes('online'));}
  $('dashboard-devices').innerHTML=html;renderMonitor(data);
  if(state.role==='host')$('train-worker-count').innerHTML=`<span>↗</span> ${onlineWorkers.length} approved remote GPU worker(s) online`;
  else $('train-worker-count').innerHTML='<span>ⓘ</span> Start training from the Host computer';
  updateTrainButton();
  if(data.host_error){$('metric-controller-detail').textContent='Credential needs attention';}
}
function renderMonitor(data){const g=data.local_gpu||{};$('monitor-local').innerHTML=g.name!=='GPU status unavailable'?`<div class="detail-line"><span>Model</span><strong>${escaped(g.name)}</strong></div><div class="detail-line"><span>GPU utilization</span><strong>${escaped(g.util)}%</strong></div><div class="detail-line"><span>VRAM used</span><strong>${escaped(g.used)} / ${escaped(g.total)} MiB</strong></div>`:'<div class="placeholder">No NVIDIA GPU telemetry available on this PC.</div>';
 const workers=data.workers||[];$('monitor-workers').innerHTML=workers.length?workers.map(w=>`<div class="detail-line"><span>${escaped(w.name||w.ip)}<br><small>${escaped(w.gpu||'GPU')}</small></span><strong>${w.online?'Online':'Offline'}</strong></div>`).join(''):'<div class="placeholder">No approved remote workers in current host session.</div>';
}
async function refreshDashboard(){if(refreshing)return;refreshing=true;try{const d=await call('dashboard');renderDashboard(d);}catch(e){$('top-online').innerHTML='<span class="dot bad"></span> App not connected';$('metric-controller').textContent='Unavailable';}finally{refreshing=false;}}
function makeRequestCard(item){let html=`<div class="pending-entry"><strong>${escaped(item.name)} · ${escaped(item.gpu)}</strong><small>LAN ${escaped(item.ip)} · Fingerprint <b>${escaped(item.fingerprint)}</b></small>`;
  if(item.status==='pending')html+=`<button class="approve-btn" data-req="${escaped(item.id)}" data-fp="${escaped(item.fingerprint)}">Approve device</button>`;
  else html+='<span class="device-status">Approved ✓</span>';
 return html+'</div>';}
async function refreshPairing(){if(state.role==='worker'){await refreshWorker();return;}
  try{const d=await call('pairing_status');const pending=d.devices||[];$('pending-list').innerHTML=pending.length?pending.map(makeRequestCard).join(''):'<div class="placeholder">No pending join requests</div>';
    $('approved-list').innerHTML=(d.approved||[]).length?(d.approved||[]).map(x=>deviceRow(x.name,`${x.gpu} · ${x.ip}:${x.port}`,true)).join(''):'<div class="placeholder">No devices approved in this Controller session.</div>';
    if(d.pairing_open&&d.seconds_left>0&&state.expireAt===0){$('code-expiry').textContent=`Pairing open · ${d.seconds_left}s left`;}else if(!d.pairing_open){$('code-expiry').textContent='Code expired. Generate a new one.';}
  }catch(e){$('pending-list').innerHTML=`<div class="placeholder">${escaped(e.message)}</div>`;}
}
async function newCode(){const d=await call('create_cluster');$('pair-code').textContent=d.code||'—';state.expireAt=Date.now()+1000*(d.expires_seconds||300);showToast('Join code ready. Enter it on the other PC and verify its fingerprint.');await refreshPairing();}
async function approve(id,fp){if(!window.confirm('Approve ONLY if the other computer displays exactly this fingerprint:\n\n'+fp+'\n\nDo they match?'))return;
  const d=await call('approve',id,fp);showToast('Device approved: '+d.worker);await refreshPairing();await refreshDashboard();}
async function discover(){const results=await call('discover_hosts');if(results.length){state.host=results[0].ip;$('join-host').value=state.host;showToast('Host found: '+results[0].name+' ('+state.host+')');}else showToast('Host was not discovered. Enter its private LAN IP manually.',true);}
async function join(){const d=await call('join_cluster',$('join-host').value.trim(),$('join-code').value.trim());showToast(d.message||'Pairing request started');await refreshWorker();}
async function refreshWorker(){try{const d=await call('worker_status');state.joinStatus=d;$('join-message').textContent=d.message||d.status||'Unknown';$('join-fingerprint').textContent=d.fingerprint||'—';$('join-gpu').textContent=d.gpu||'Unknown';$('join-worker').textContent=d.worker||'stopped';}catch(e){$('join-message').textContent=e.message;}}
function prettyLog(log){return String(log||'').split('\n').map(line=>{
 if(/^P19 COMPLETE\s/i.test(line))return 'Training complete · checkpoint saved';
 if(/^P19 algorithm=/i.test(line))return 'Distributed training started (local SGD + weighted averaging)';
 return line.replace(/\bPhase\s*(18|19|20|21)\b/gi,'Training').replace(/phase19_runs/gi,'training_runs').replace(/model_zoo_runs/gi,'training_runs');
}).join('\n');}
function modelText(m){return `${m.family||''} ${m.key||''} ${m.description||''} ${m.task||''} ${m.dataset||''}`.toLowerCase();}
function renderModelCards(){const q=$('model-search').value.trim().toLowerCase(),filter=$('model-filter').value;
 const visible=state.models.filter(m=>modelText(m).includes(q)&&(filter==='all'||(filter==='experimental'&&m.experimental)||(m.modes||[]).includes(filter)));
 $('catalog-summary').textContent=`${visible.length} of ${state.models.length} models · ${state.models.filter(m=>(m.modes||[]).includes('multi_gpu')).length} available across multiple GPUs · ${state.models.filter(m=>m.experimental).length} experimental`;
 $('model-cards').innerHTML=visible.map(m=>`<div class="card model-card"><div class="model-icon">◈</div><h3>${escaped(m.family||'AI Model')}</h3><strong>${escaped(m.key)}</strong><div class="model-tags">${(m.modes||[]).map(mode=>`<span class="model-tag">${mode==='multi_gpu'?'Multiple GPUs':'This PC'}</span>`).join('')}${m.experimental?'<span class="model-tag experimental">Experimental · extra setup</span>':''}</div><p class="model-description">${escaped(m.description||'')}</p><p class="dataset-tip">Dataset: ${escaped(m.dataset||'See model documentation')}</p><button class="btn ghost choose-model" data-model="${escaped(m.key)}">Open terminal</button></div>`).join('')||'<div class="card placeholder">No models match this filter.</div>';
}
async function loadCatalog(force=false){if(state.models.length&&!force)return;try{
 const data=await call('catalog');state.models=data.models||[];state.catalogErrors=data.errors||{};
 if(!state.models.length)throw Error('No models were returned by the connected Host.');
 updateModelSelect();renderModelCards();
 if(Object.keys(state.catalogErrors).length)showToast('Some model training modes are unavailable. Check your Host setup.',true);
 }catch(e){$('model-select').innerHTML='<option value="">Model library unavailable</option>';$('model-cards').innerHTML=`<div class="card placeholder">${escaped(e.message)}</div>`;
 $('catalog-summary').textContent='Connect the Host and confirm its saved credentials. Try Refresh.';showToast(e.message,true);
 }}
function updateTrainButton(){const hasWorker=!!(state.last&&state.last.workers||[]).some(w=>w.online);
 const ready=state.role==='host'&&state.online&&state.models.some(m=>(m.modes||[]).includes(state.mode));
 $('start-train').disabled=!ready||(state.mode==='multi_gpu'&&!hasWorker)||state.jobState==='running';
 $('start-train').textContent=state.mode==='multi_gpu'?'▶   Start Multi-GPU Training':'▶   Start Training on This PC';
 $('train-worker-count').textContent=state.mode==='multi_gpu'?(hasWorker?'Connected GPU workers ready':'Pair an online GPU worker to use multiple GPUs'):'Trains with this computer’s GPU only';
 $('mode-help').textContent=state.mode==='multi_gpu'?'Uses separate GPUs with local SGD and weighted averaging; VRAM is not pooled.':'Runs on the Host GPU without requiring another computer.';
}
function updateModelSelect(preferred=''){const sel=$('model-select'), old=preferred||sel.value;
 const matching=state.models.filter(m=>(m.modes||[]).includes(state.mode));sel.replaceChildren();
 matching.forEach(m=>{const o=document.createElement('option');o.value=m.key;o.textContent=(m.family||'AI Model')+' · '+m.key;sel.append(o);});
 if(matching.some(m=>m.key===old))sel.value=old;
 else if(matching.some(m=>m.key==='tabular_mlp_regression'))sel.value='tabular_mlp_regression';
 updateModel();updateTrainButton();
}
function setTrainingMode(mode, preferred=''){state.mode=mode;$('training-mode').value=mode;updateModelSelect(preferred);}
function suggestDataset(m){const key=m.key||'';
 if(key==='tabular_mlp_regression')return 'data/zoo_demo/tabular_regression.csv';
 if(key==='tabular_mlp_classification')return 'data/zoo_demo/tabular_classification.csv';
 if(key==='text_gru'||key==='text_transformer')return 'data/zoo_demo/text.csv';
 if(key==='tiny_transformer_lm'||key==='distilgpt2_lora')return 'data/zoo_demo/lm.txt';
 if(key==='time_series_lstm')return 'data/zoo_demo/timeseries.csv';
 if(key==='unet_segmentation')return 'data/zoo_demo/segmentation';
 if(key==='audio_spectrogram_cnn')return 'data/zoo_demo/spectrograms';
 if(m.task==='image_classification')return 'data/zoo_demo/images';
 return '';
}
function updateModel(){const m=state.models.find(m=>m.key===$('model-select').value);
 $('model-info').textContent=m?(m.description+' · Dataset: '+(m.dataset||'See requirements')+(m.experimental?' · Experimental: extra dependencies and downloads may be needed':'')):'No supported model available in this mode.';
 if(m){$('batch').value=m.suggested_batch||4;const demo=suggestDataset(m);if(demo)$('dataset').value=demo;}
 updateTrainButton();
}
async function browseDataset(kind){const d=await call('browse_dataset',kind);if(!d.cancelled){$('dataset').value=d.path;showToast('Dataset selected');}}
async function startTraining(){const spec={mode:state.mode,model:$('model-select').value,dataset:$('dataset').value.trim(),epochs:Number($('epochs').value),batch_size:Number($('batch').value),learning_rate:Number($('lr').value)};
 const m=state.models.find(x=>x.key===spec.model);if(!m)throw Error('Select a supported model.');
 if(m.experimental&&!confirm('This is experimental training. Extra packages, downloads or GPU memory may be required. Continue?'))return;
 if(!confirm(`Train ${spec.model} for ${spec.epochs} epoch(s) on ${spec.mode==='multi_gpu'?'multiple GPUs':'this PC'}?`))return;
 const d=await call('start_training',spec);state.activeJob=d.id;state.activeMode=d.mode||spec.mode;state.jobState='running';$('job-badge').textContent='RUNNING';$('job-badge').className='status-tag good';$('job-label').textContent='Job '+d.id;$('job-log').textContent='Preparing training…';$('checkpoint').textContent='';updateTrainButton();showToast('Training job started');await pollJob();}
function graphLoss(log){const regex=/Epoch\s+\d+\/\d+\s+(?:weighted_loss|loss)=([0-9.]+)/g, values=[];for(const m of log.matchAll(regex)){const n=Number(m[1]);if(Number.isFinite(n))values.push(n);}if(!values.length){$('loss-line').setAttribute('points','');$('latest-loss').textContent='Awaiting loss data';return;}
 const max=Math.max(...values),min=Math.min(...values),span=Math.max(max-min,.001);const pts=values.map((v,i)=>`${12+(values.length===1?0:i/(values.length-1)*526)},${20+(max-v)/span*168}`).join(' ');$('loss-line').setAttribute('points',pts);$('latest-loss').textContent='Weighted loss: '+values.at(-1).toFixed(6);}
async function pollJob(){if(!state.activeJob||jobPolling)return;jobPolling=true;try{const d=await call('training_status',state.activeJob,state.activeMode);state.jobState=d.state;
 const log=prettyLog(d.log||'');$('job-log').textContent=log||'Training is starting…';$('monitor-log').textContent=log||'Training is starting…';$('job-state').textContent=d.state||'Unknown';$('job-badge').textContent=String(d.state||'Unknown').toUpperCase();$('job-badge').className='status-tag '+(d.state==='failed'?'bad':d.state==='completed'||d.state==='running'?'good':'');
 $('checkpoint').textContent=d.checkpoint?'Saved checkpoint · model.pt':'';graphLoss(d.log||'');$('metric-training').textContent=d.state==='running'?'Running':d.state==='completed'?'Complete':d.state==='failed'?'Failed':'Idle';updateTrainButton();
 if(d.state!=='running'){showToast('Training '+d.state+(d.checkpoint?' · checkpoint saved':''),d.state==='failed');await loadHistory();}
 }catch(e){$('job-state').textContent=e.message;}finally{jobPolling=false;}}
async function loadHistory(){try{const jobs=await call('training_history');$('history-list').innerHTML=jobs.length?jobs.map(x=>`<div class="history-item"><div><strong>${escaped(x.model||'Training job')}</strong><small>${x.mode==='multi_gpu'?'Multiple GPUs':'This PC'} · ${escaped(x.dataset)} · ${escaped(x.id)}</small><small>${x.checkpoint?'✓ Checkpoint saved':'No checkpoint yet'}</small></div><div class="right"><strong>${escaped(x.state)}</strong><small>${new Date(x.time*1000).toLocaleString()}</small></div></div>`).join(''):'<div class="placeholder">No saved training jobs yet.</div>';
 }catch(e){$('history-list').innerHTML=`<div class="placeholder">${escaped(e.message)}</div>`;}}
async function saveSettings(){const role=$('settings-worker').classList.contains('selected')?'worker':'host';const d=await call('settings_update',role,$('settings-ip').value.trim(),$('settings-root').value.trim());state.role=d.role;state.host=d.host;state.root=d.root;state.models=[];setRoleUI();showToast('Settings saved');await refreshDashboard();await loadCatalog(true);}
async function startServices(){const d=await call('start_services');showToast(d.message);}
let gpuPanelBusy=false;
function metric(value,unit=''){return value===null||value===undefined?'Unavailable':String(value)+unit;}
async function refreshGpuPanel(){if(gpuPanelBusy)return;gpuPanelBusy=true;
 try{const data=await call('cluster_gpu_status');const lines=[];
 if(data.error)lines.push(data.error,'');
 for(const node of data.nodes){lines.push(node.name+' | '+node.address,node.error||('Metrics received | '+node.sampled_at));
 for(const g of node.gpus){lines.push('  GPU '+g.id+': '+g.name,'  Load: '+metric(g.util,'%')+' | Temperature: '+metric(g.temperature,' °C'),'  VRAM used/free/total: '+metric(g.used)+' / '+metric(g.free)+' / '+metric(g.total)+' MiB');}lines.push('');}
 $('cluster-gpu-console').textContent=lines.join('\n');$('gpu-summary').textContent='Last refresh: '+new Date().toLocaleTimeString();
 }catch(e){$('cluster-gpu-console').textContent='Metrics unavailable: '+e.message;$('gpu-summary').textContent='Refresh failed';}finally{gpuPanelBusy=false;}}
function wire(){
 $('open-native-terminal').onclick=()=>safeTask(async()=>{const d=await call('open_terminal');showToast(d.message);});
 $('gpu-refresh').onclick=()=>safeTask(refreshGpuPanel);
 setInterval(()=>{if(state.view==='train')safeTask(refreshGpuPanel);},6000);
document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>view(el.dataset.view)));
 document.querySelectorAll('[data-go]').forEach(el=>el.addEventListener('click',()=>view(el.dataset.go)));
 $('refresh-button').onclick=()=>safeTask(async()=>{await refreshDashboard();await refreshPairing();});
 document.querySelector('[data-action="create"]').onclick=()=>safeTask(async()=>{view('cluster');if(state.role==='host')await newCode();else showToast('Switch to Host role to create a cluster.');});
 $('cluster-host-tab').onclick=()=>safeTask(()=>selectRole('host'));$('cluster-worker-tab').onclick=()=>safeTask(()=>selectRole('worker'));
 $('settings-host').onclick=()=>{$('settings-host').classList.add('selected');$('settings-worker').classList.remove('selected');};
 $('settings-worker').onclick=()=>{$('settings-worker').classList.add('selected');$('settings-host').classList.remove('selected');};
 $('create-code').onclick=()=>safeTask(newCode);$('refresh-pending').onclick=()=>safeTask(refreshPairing);
 $('pending-list').addEventListener('click',e=>{const t=e.target.closest('.approve-btn');if(t)safeTask(()=>approve(t.dataset.req,t.dataset.fp));});
 $('discover-host').onclick=()=>safeTask(discover);$('request-join').onclick=()=>safeTask(join);
 $('model-select').onchange=updateModel;$('training-mode').onchange=e=>setTrainingMode(e.target.value);$('model-search').oninput=renderModelCards;$('model-filter').onchange=renderModelCards;$('refresh-catalog').onclick=()=>safeTask(()=>loadCatalog(true));$('browse-file').onclick=()=>safeTask(()=>browseDataset('file'));$('browse-folder').onclick=()=>safeTask(()=>browseDataset('folder'));$('train-form').onsubmit=e=>{e.preventDefault();safeTask(startTraining);};
 $('model-cards').addEventListener('click',e=>{const t=e.target.closest('.choose-model');if(t){const m=state.models.find(x=>x.key===t.dataset.model);if(m){const mode=(m.modes||[]).includes('multi_gpu')?'multi_gpu':'single_gpu';setTrainingMode(mode,m.key);view('train');}}});
 $('refresh-history').onclick=()=>safeTask(loadHistory);$('save-settings').onclick=()=>safeTask(saveSettings);$('start-service').onclick=()=>safeTask(startServices);
}
async function init(){wire();try{const d=await call('dashboard');renderDashboard(d);$('settings-root').value=state.root;$('settings-ip').value=state.host;
  if(!state.root){view('settings');showToast('Choose your Distributed-GPU project folder to start.');}else{await loadCatalog();await refreshPairing();}}
  catch(e){view('settings');showToast(e.message,true);}setInterval(()=>safeTask(refreshDashboard),5500);setInterval(()=>{if(state.view==='cluster')safeTask(refreshPairing);},3200);setInterval(()=>{if(state.activeJob&&state.jobState==='running')safeTask(pollJob);},2300);
  setInterval(()=>{if(state.expireAt>0){const n=Math.max(0,Math.ceil((state.expireAt-Date.now())/1000));$('code-expiry').textContent=n?`Expires in ${n}s`:'Code expired. Generate a new one.';if(n===0)state.expireAt=0;}},1000);
}
window.addEventListener('pywebviewready',()=>{init();},{once:true});
if(!window.pywebview){setTimeout(()=>{if(!window.pywebview)showToast('This is a desktop client. Open it using DistributedGPUStudio.exe, not a browser.',true);},1400);}
