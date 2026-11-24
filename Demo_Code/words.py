"""Metadata word search and on-demand aligned F1465 comparisons."""
from pathlib import Path
from collections import OrderedDict
import base64, hashlib, io, json, re, sys, math

INDEX=None
ALIGNER=None
CACHE=OrderedDict()

def catalog(base):
 global INDEX
 if INDEX is not None:return INDEX
 index={}
 for folder in ['all_came_f1465_check','noncame_f1465_check']:
  path=base/folder/'results.json'
  if not path.is_file():raise ValueError('The saved corpus analysis is missing: '+folder)
  for r in json.loads(path.read_text()):
   if r.get('error'):continue
   for word in set(re.findall(r"[a-z]+(?:'[a-z]+)?",r['text'].lower())):
    index.setdefault(word,[]).append(r)
 INDEX=index
 return index

def compare(payload,base,models):
 global ALIGNER
 import numpy as np,torch,soundfile as sf
 from scipy.signal import resample_poly
 terms=payload.get('words',['came','gave'])
 if not isinstance(terms,list) or not 1<=len(terms)<=4:raise ValueError('Choose one to four words.')
 if any(not isinstance(w,str) or not re.fullmatch(r"[a-z]+(?:'[a-z]+)?",w.lower().strip()) for w in terms):raise ValueError('Enter individual English words separated by commas.')
 terms=list(dict.fromkeys(w.lower().strip() for w in terms))
 count=payload.get('count',3)
 if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=3:raise ValueError('Choose 1 to 3 recordings per word.')
 mode=payload.get('selection','sample')
 if mode not in ['sample','strongest']:raise ValueError('Unknown selection rule.')
 snapshot=base/'data/word_examples.json'
 if snapshot.is_file() and not (base/'all_came_f1465_check/results.json').is_file():
  import copy
  saved=json.loads(snapshot.read_text())[mode]
  available={g['word']:g for g in saved['groups']}
  groups=[]
  for word in terms:
   if word not in available:raise ValueError('The public example collection contains came and gave. Use the local dataset setup to search additional words.')
   group=copy.deepcopy(available[word]);group['examples']=group['examples'][:count];groups.append(group)
  return {**saved,'groups':groups,'count_per_word':count,'public_snapshot':True}
 idx=catalog(base);groups=[];h,s,device=models
 sys.path.insert(0,str(base))
 from came_feature_check import ctc_word_alignment
 for word in terms:
  records=idx.get(word,[])
  ordered=sorted(records,key=(lambda r:(-r['F1465_global_peak'],r['recording_id'])) if mode=='strongest' else (lambda r:hashlib.sha256(('word-demo-v1:'+r['recording_id']).encode()).hexdigest()))
  group={'word':word,'recordings':len(records),'active_anywhere':sum(r['F1465_global_peak']>0 for r in records),'examples':[]}
  for r in ordered[:count]:
   key=(r['recording_id'],word)
   if key in CACHE:
    CACHE.move_to_end(key);group['examples'].append(CACHE[key]);continue
   row={'recording':r['recording_id'],'emotion':r['emotion'],'transcript':r['text'],'greedy_asr':r.get('greedy_asr',''),'error':None}
   try:
    wave,sr=sf.read(base.parent/'Data/train_extracted/vmc2026-track2/wav'/r['recording_id'],dtype='float32',always_2d=True);wave=wave.mean(1)
    if sr!=16000:
     g=math.gcd(sr,16000);wave=resample_poly(wave,16000//g,sr//g)
    wave=np.asarray(wave,dtype=np.float32)
    if len(wave)>16000*30:raise ValueError('This recording exceeds the interactive 30-second limit.')
    with torch.inference_mode():v=s.encode(h(torch.from_numpy(wave)[None].to(device)),use_threshold=True)[0,:,1465].cpu().numpy()
    spans=r.get('word_alignment')
    if not spans:
     if ALIGNER is None:
      from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
      pr=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h');m=Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval();ALIGNER=(pr,m)
     pr,m=ALIGNER
     with torch.inference_mode():lp=torch.log_softmax(m(pr(wave,sampling_rate=16000,return_tensors='pt').input_values.to(device)).logits[0],-1).cpu().numpy()
     spans=ctc_word_alignment(lp,r['text'],pr.tokenizer,m.config.pad_token_id,float(np.prod(m.config.conv_stride))/16000)
    matches=[w for w in spans if w['word'].lower()==word]
    if not matches:raise ValueError('No aligned occurrence found for this metadata word.')
    times=np.arange(len(v))/50;peak=float(v.max());pt=float(times[v.argmax()]) if peak>0 else None
    occurrences=[]
    for span in matches:
     start,end=span['start'],span['end'];mask=(times>=start)&(times<end)
     lo=max(0,int((start-.25)*16000));hi=min(len(wave),int((end+.25)*16000));buf=io.BytesIO();sf.write(buf,wave[lo:hi],16000,format='WAV',subtype='PCM_16')
     trace=(times>=lo/16000)&(times<hi/16000)
     occurrences.append({'start':start,'end':end,'character_posterior':span['mean_character_posterior'],'word_peak':float(v[mask].max()) if mask.any() else None,'word_frequency':float((v[mask]>0).mean()) if mask.any() else None,'global_peak_overlaps':bool(pt is not None and start<=pt<end),'relative_times':(times[trace]-start).tolist(),'values':v[trace].tolist(),'audio':'data:audio/wav;base64,'+base64.b64encode(buf.getvalue()).decode(),'context_start':lo/16000,'context_end':hi/16000})
    row.update(global_peak=peak,global_peak_time=pt,occurrences=occurrences)
   except Exception as e:row['error']='Could not analyze this example: '+str(e)
   if not row['error']:
    CACHE[key]=row
    while len(CACHE)>32:CACHE.popitem(last=False)
   group['examples'].append(row)
  groups.append(group)
 return {'feature':1465,'selection':mode,'count_per_word':count,'groups':groups,'alignment':'Approximate metadata-forced CTC word alignment; boundaries are not human-verified. Character posterior is a diagnostic, not calibrated confidence.','scope':'Corpus totals mean F1465 active anywhere in the recording, not necessarily during the searched word. Matching recordings may share sentences, speakers, and systems.'}
