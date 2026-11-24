"""Five preselected came-transcript recordings: HuBERT SAE F1465 plus approximate CTC word timing."""
from pathlib import Path
import re,json,csv,hashlib,sys,math,os
import numpy as np
REFERENCE='vmc2026-track2-sys014-spk006-utt136.wav'
ALIGNMENT_MODEL='facebook/wav2vec2-base-960h'

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def ctc_word_alignment(log_probs,transcript,tokenizer,blank,seconds_per_frame):
 """Full-transcript CTC Viterbi alignment; returns approximate character-supported word spans."""
 text=' '.join(re.sub("[^A-Z' ]",' ',transcript.upper()).split())
 chars=text.replace(' ','|');tokens=np.array(tokenizer.convert_tokens_to_ids(list(chars)),dtype=int)
 assert len(tokens) and not np.any(tokens==tokenizer.unk_token_id),'Unsupported transcript token'
 states=np.full(2*len(tokens)+1,blank,dtype=int);states[1::2]=tokens
 T,V=log_probs.shape;S=len(states);assert T>=len(tokens),'Audio shorter than transcript label sequence'
 prev=np.full(S,-np.inf);prev[0]=log_probs[0,blank]
 if S>1:prev[1]=log_probs[0,states[1]]
 parents=np.zeros((T,S),dtype=np.int8)
 can_skip=np.zeros(S,dtype=bool);can_skip[2:]=(states[2:]!=blank)&(states[2:]!=states[:-2])
 for t in range(1,T):
  options=np.stack([prev,np.r_[-np.inf,prev[:-1]],np.r_[-np.inf,-np.inf,prev[:-2]]])
  options[2,~can_skip]=-np.inf
  step=options.argmax(0);parents[t]=step;prev=options[step,np.arange(S)]+log_probs[t,states]
 end=S-1 if prev[-1]>=prev[-2] else S-2
 assert np.isfinite(prev[end]),'No valid transcript alignment'
 path=np.empty(T,dtype=int);path[-1]=end
 for t in range(T-1,0,-1):path[t-1]=path[t]-parents[t,path[t]]
 char_spans=[]
 for j,char in enumerate(chars):
  frames=np.flatnonzero(path==2*j+1);assert len(frames)>0
  char_spans.append((char,int(frames[0]),int(frames[-1])+1,float(np.exp(log_probs[frames,tokens[j]]).mean())))
 words=[];cursor=0
 for word in text.split():
  chunk=char_spans[cursor:cursor+len(word)]
  words.append({'word':word.lower(),'start':chunk[0][1]*seconds_per_frame,'end':chunk[-1][2]*seconds_per_frame,'mean_character_posterior':float(np.mean([c[3] for c in chunk]))})
  cursor+=len(word)+1
 return words

def selection(base,data):
 metadata=[]
 with open(data/'metadata.csv',newline='') as f:
  for r in csv.reader(f,delimiter='|'):
   if len(r)>=3 and re.search(r'\bcame\b','|'.join(r[2:]),re.I) and (data/'wav'/r[0]).is_file():
    metadata.append({'recording_id':r[0],'emotion':r[1],'text':'|'.join(r[2:]),'system':r[0].split('-')[2],'speaker':r[0].split('-')[3]})
 excluded={REFERENCE}
 if (base/'selected_recordings.json').exists():excluded.update(r['wavID'] for r in json.loads((base/'selected_recordings.json').read_text()))
 # Also exclude explicitly named examples anywhere in the current learning notebook.
 notebook=json.loads((base/'undestandingSAE.ipynb').read_text())
 for cell in notebook['cells']:
  src=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
  excluded.update(re.findall(r'vmc2026-track2-sys\d+-spk\d+-utt\d+\.wav',src))
 rng=np.random.default_rng(20261123);chosen=[];systems=set();speakers=set()
 for emotion in ['Angry','Happy','Neutral','Sad','Surprise']:
  pool=sorted([r for r in metadata if r['emotion']==emotion and r['recording_id'] not in excluded],key=lambda r:r['recording_id'])
  assert pool,emotion
  order=rng.permutation(len(pool));pool=[pool[i] for i in order]
  r=next((r for r in pool if r['system'] not in systems and r['speaker'] not in speakers),pool[0])
  chosen.append(r);systems.add(r['system']);speakers.add(r['speaker'])
 manifest={'selection_seed':20261123,'selection_rule':'One new recording per intended emotion; deterministic metadata-only selection, preferring distinct systems and speakers; no activation screening','available_came_recordings':len(metadata),'unique_came_transcripts':sorted(set(r['text'] for r in metadata)),'excluded_recordings':sorted(excluded),'selected':chosen,'reference':next(r for r in metadata if r['recording_id']==REFERENCE)}
 return manifest

def run_came_checks(base=None):
 base=Path(base or Path(__file__).resolve().parent)
 data=Path(os.environ.get("SPEECHSAE_DATA",str(base/"Data/vmc2026-track2")));out=base/'came_f1465_check';out.mkdir(exist_ok=True);figs=out/'figures';figs.mkdir(exist_ok=True)
 sys.path.insert(0,str(base/'vendor'))
 import os
 os.environ.setdefault('HF_HOME',str(base/'hf_cache'));os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
 import torch,soundfile as sf,librosa
 import matplotlib.pyplot as plt
 from scipy.signal import resample_poly
 from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
 from IPython.display import display,Image,Audio,Markdown
 from audio_sae import BatchTopKSAE
 from audio_sae.models import MyHubert
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 device='cuda:0' if torch.cuda.is_available() else 'cpu';SR=16000
 # Reuse the precommitted metadata-only choices on reruns.
 manifest_path=out/'selection.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else selection(base,data)
 if not manifest_path.exists():manifest_path.write_text(json.dumps(manifest,indent=2))
 print('Selected five new recordings without inspecting F1465 activations:',flush=True)
 for row in manifest['selected']:print(row['emotion'],row['recording_id'],flush=True)
 print('All available came examples share this transcript:',manifest['unique_came_transcripts'],flush=True)
 hubert=MyHubert('facebook/hubert-base-ls960',sae_after_layer=11).to(device).eval()
 sae=BatchTopKSAE.from_pretrained(str(base/'ae_layer11.pt'),device=device).eval()
 print('Loading independent ASR model for approximate word alignment...',flush=True)
 processor=Wav2Vec2Processor.from_pretrained(ALIGNMENT_MODEL)
 aligner=Wav2Vec2ForCTC.from_pretrained(ALIGNMENT_MODEL).to(device).eval()
 seconds_per_frame=float(np.prod(aligner.config.conv_stride))/SR
 results=[]
 for record in [manifest['reference'],*manifest['selected']]:
  path=data/'wav'/record['recording_id'];wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
  if sr!=SR:
   g=math.gcd(sr,SR);wave=resample_poly(wave,SR//g,sr//g)
  wave=np.asarray(wave,dtype=np.float32)
  with torch.inference_mode():
   acts=hubert(torch.from_numpy(wave)[None].to(device));features=sae.encode(acts,use_threshold=True)[0,:,1465].cpu().numpy()
   inputs=processor(wave,sampling_rate=SR,return_tensors='pt').input_values.to(device)
   logits=aligner(inputs).logits[0]
   log_probs=torch.log_softmax(logits,dim=-1).cpu().numpy()
  greedy=processor.decode(logits.argmax(-1).cpu().numpy())
  words=ctc_word_alignment(log_probs,record['text'],processor.tokenizer,aligner.config.pad_token_id,seconds_per_frame)
  came=next(w for w in words if w['word']=='came')
  times=np.arange(len(features))/50.;mask=(times>=came['start'])&(times<came['end'])
  peak=float(features.max());peak_time=float(times[features.argmax()]) if peak>0 else None
  metrics={**record,'is_reference':record['recording_id']==REFERENCE,'duration_seconds':len(wave)/SR,'greedy_asr':greedy,'greedy_asr_contains_came':bool(re.search(r'\bcame\b',greedy,re.I)),'estimated_came_start':came['start'],'estimated_came_end':came['end'],'came_alignment_mean_character_posterior':came['mean_character_posterior'],'F1465_global_peak':peak,'F1465_global_peak_seconds':peak_time,'peak_in_estimated_came_span':bool(peak_time is not None and came['start']<=peak_time<came['end']),'maximum_F1465_within_came':float(features[mask].max()) if mask.any() else None,'maximum_F1465_outside_came':float(features[~mask].max()) if (~mask).any() else None,'positive_frames_total':int((features>0).sum()),'positive_frames_within_came':int((features[mask]>0).sum()),'audio_sha256':digest(path),'word_alignment':words}
  mel=librosa.power_to_db(librosa.feature.melspectrogram(y=wave,sr=SR,n_mels=80,hop_length=160),ref=np.max)
  results.append({'metrics':metrics,'wave':wave,'times':times,'features':features,'mel':mel})
  print(record['recording_id'],'came',round(came['start'],2),round(came['end'],2),'peak',peak_time,round(peak,3),'overlap',metrics['peak_in_estimated_came_span'],flush=True)
  np.savez_compressed(out/(record['recording_id']+'.npz'),waveform=wave,feature_response=features,feature_times=times,log_mel=mel)
 exported=[r['metrics'] for r in results];(out/'results.json').write_text(json.dumps(exported,indent=2))
 fields=[k for k in exported[0] if k!='word_alignment']
 with open(out/'results.csv','w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows([{k:r[k] for k in fields} for r in exported])
 maximum=max(1,max(r['metrics']['F1465_global_peak'] for r in results))*1.12
 def draw_rows(cases,filename):
  fig,axes=plt.subplots(len(cases),2,figsize=(14,3*len(cases)),squeeze=False,gridspec_kw={'width_ratios':[1.5,1]})
  for (left,right),r in zip(axes,cases):
   m=r['metrics'];start=m['estimated_came_start'];end=m['estimated_came_end']
   left.imshow(r['mel'],origin='lower',aspect='auto',extent=[0,m['duration_seconds'],0,80],cmap='magma',vmin=-80,vmax=0)
   left.set(ylabel='Mel band',title=f"{m['emotion']} | {m['recording_id']}")
   right.plot(r['times'],r['features'],color='#c36b20',lw=1.8);right.set(ylabel='F1465 response',ylim=(0,maximum),title=f"Peak {m['F1465_global_peak']:.2f}; estimated came {start:.2f}–{end:.2f} s")
   for ax in (left,right):
    ax.axvspan(start,end,color='#37a16f',alpha=.22,label='Estimated came interval');ax.set(xlabel='Time in original recording (seconds)',xlim=(0,m['duration_seconds']))
    if m['F1465_global_peak_seconds'] is not None:ax.axvline(m['F1465_global_peak_seconds'],color='#b83232',ls='--',lw=1)
   right.legend(loc='upper right',fontsize=8)
  fig.suptitle('Green: approximate CTC-aligned came; red dashed: F1465 global peak',y=.998);fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(figs/filename,dpi=130,bbox_inches='tight');plt.close(fig)
 draw_rows(results[1:],'five_new_came_examples.png');draw_rows(results[:1],'reference_came_alignment.png')
 for i,r in enumerate(results[1:],1):draw_rows([r],f'new_example_{i}.png')
 display(Image(filename=str(figs/'five_new_came_examples.png')))
 display(Markdown('**Original reference with the same approximate word alignment:**'))
 display(Image(filename=str(figs/'reference_came_alignment.png')))
 rows='| Example | Emotion | Estimated came (s) | F1465 peak (s) | Peak activation | Peak overlaps came? | Greedy ASR contains came? |\n|---|---|---:|---:|---:|---|---|\n'
 for i,r in enumerate(exported):
  peak='inactive' if r['F1465_global_peak_seconds'] is None else f"{r['F1465_global_peak_seconds']:.2f}"
  rows+=f"| {'Reference' if i==0 else str(i)} | {r['emotion']} | {r['estimated_came_start']:.2f}–{r['estimated_came_end']:.2f} | {peak} | {r['F1465_global_peak']:.3f} | {r['peak_in_estimated_came_span']} | {r['greedy_asr_contains_came']} |\n"
 display(Markdown(rows))
 for i,r in enumerate(results[1:],1):
  m=r['metrics'];wave=r['wave']
  display(Markdown(f"**New example {i}: {m['recording_id']} — {m['emotion']}**\n\nTranscript: {m['text']}\n\nIndependent greedy ASR: {m['greedy_asr']}"))
  display(Markdown('Full recording:'));display(Audio(wave,rate=SR,normalize=False))
  lo=max(0,int((m['estimated_came_start']-.15)*SR));hi=min(len(wave),int((m['estimated_came_end']+.15)*SR))
  display(Markdown(f'Estimated came plus context: {lo/SR:.2f}–{hi/SR:.2f}s'));display(Audio(wave[lo:hi],rate=SR,normalize=False))
  if m['F1465_global_peak_seconds'] is not None:
   lo=max(0,int((m['F1465_global_peak_seconds']-.30)*SR));hi=min(len(wave),int((m['F1465_global_peak_seconds']+.30)*SR))
   display(Markdown(f'F1465 peak plus context: {lo/SR:.2f}–{hi/SR:.2f}s'));display(Audio(wave[lo:hi],rate=SR,normalize=False))
 nactive=sum(r['F1465_global_peak']>0 for r in exported[1:]);noverlap=sum(r['peak_in_estimated_came_span'] for r in exported[1:])
 summary={'new_examples':5,'new_examples_with_nonzero_F1465':nactive,'new_examples_with_global_peak_in_estimated_came_span':noverlap,'selection_was_metadata_only':True,'all_came_candidates_share_one_sentence':True,'word_alignment_is_approximate':True,'feature_meaning_confirmed':False,'alignment_model':ALIGNMENT_MODEL,'alignment_model_revision':getattr(aligner.config,'_commit_hash',None),'sae_checkpoint_sha256':digest(base/'ae_layer11.pt'),'helper_sha256':digest(Path(__file__)),'reference_peak_overlaps_estimated_came':exported[0]['peak_in_estimated_came_span']}
 summary['new_examples_greedy_asr_contains_came']=sum(r['greedy_asr_contains_came'] for r in exported[1:])
 summary['new_examples_needing_manual_transcript_review']=[r['recording_id'] for r in exported[1:] if not r['greedy_asr_contains_came']]
 summary['asr_came_examples_with_inactive_feature']=[r['recording_id'] for r in exported[1:] if r['greedy_asr_contains_came'] and r['F1465_global_peak']==0]
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 message=f"Of five new, preselected recordings, {nactive} have a nonzero F1465 response and {noverlap} have their global activation peak inside the estimated came interval."
 print(message)
 display(Markdown('**Interpretation:** '+message+' These are repeated renditions of the same sentence, so this does not distinguish the word came from its phonemes, nearby context, or another correlated property. Alignment is approximate; listen to both the estimated word and peak excerpts. No feature meaning is confirmed.'))
 if summary['new_examples_needing_manual_transcript_review']:
  display(Markdown('**Alignment caution:** greedy ASR did not recognize came in '+', '.join(summary['new_examples_needing_manual_transcript_review'])+'. The transcript is metadata, and forced alignment assumes that transcript. Listen to this example before treating its estimated word interval or absent F1465 response as a clear word-level counterexample.'))
 return summary

if __name__=='__main__':run_came_checks()
