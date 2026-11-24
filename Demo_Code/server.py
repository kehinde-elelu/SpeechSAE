"""Private, loopback-only HuBERT AudioSAE demo; no web dependencies required."""
from pathlib import Path
import argparse, base64, io, json, os, subprocess, sys, tempfile, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('SAE_BASE', str(ROOT.parent)))
os.environ.setdefault('HF_HOME', str(BASE/'hf_cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(BASE/'vendor'))
MODEL = None
ERROR = None
LOCK = threading.Lock()
CACHE = {}
EXAMPLES = {
 'own': ('Your voice · a new sentence', BASE/'audio_recordbyme/me_1.wav'),
 'angry': ('Angry · Rat came…', BASE.parent/'Data/train_extracted/vmc2026-track2/wav/vmc2026-track2-sys005-spk011-utt046.wav'),
 'neutral': ('Neutral · Rat came…', BASE.parent/'Data/train_extracted/vmc2026-track2/wav/vmc2026-track2-sys004-spk009-utt016.wav'),
 'inactive': ('Surprise · inactive F1465 example', BASE.parent/'Data/train_extracted/vmc2026-track2/wav/vmc2026-track2-sys011-spk028-utt136.wav'),
}

def load():
 global MODEL, ERROR
 try:
  import torch
  from audio_sae import BatchTopKSAE
  from audio_sae.models import MyHubert
  torch.set_num_threads(4)
  device = os.environ.get('SAE_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
  checkpoint=BASE/'ae_layer11.pt'
  if not checkpoint.is_file():
   from huggingface_hub import hf_hub_download
   checkpoint=Path(hf_hub_download(repo_id='Egorgij21/Audio-SAE-hubert-base',filename='layer_11/ae.pt'))
  h = MyHubert('facebook/hubert-base-ls960', sae_after_layer=11).to(device).eval()
  s = BatchTopKSAE.from_pretrained(str(checkpoint), device=device).eval()
  MODEL = (h,s,device)
 except Exception:
  traceback.print_exc(); ERROR = 'Could not load the models. Check the server log and SAE_BASE configuration.'

def analyze(raw, name):
 import numpy as np
 import soundfile as sf
 import torch, librosa
 from audio_sae import reconstruction_metrics
 from scipy.signal import resample_poly
 import math
 try:
  wave,sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
 except Exception:
  # Decode containers such as M4A and browser WebM without trusting their extension.
  with tempfile.TemporaryDirectory(prefix='sae-demo-') as temp:
   inp=Path(temp)/'input'; out=Path(temp)/'decoded.wav';inp.write_bytes(raw)
   try:
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(inp),'-t','31','-ac','1','-ar','16000',str(out)], check=True, timeout=25, capture_output=True)
    wave,sr=sf.read(out,dtype='float32',always_2d=True)
   except (subprocess.SubprocessError,FileNotFoundError):
    raise ValueError('This audio could not be decoded. Try WAV, FLAC, MP3, M4A, or WebM.')
 if len(wave)/sr > 30:
  raise ValueError('Please use a recording of 30 seconds or less. The recording was not analyzed.')
 wave=wave.mean(axis=1)
 if sr!=16000:
  g=math.gcd(sr,16000);wave=resample_poly(wave,16000//g,sr//g)
 wave=np.asarray(wave,dtype=np.float32)
 if len(wave)<400 or not np.isfinite(wave).all():raise ValueError('Please provide at least 25 ms of valid audio.')
 h,s,device=MODEL
 with torch.inference_mode():
  hidden=h(torch.from_numpy(wave)[None].to(device));z=s.encode(hidden,use_threshold=True);rec=s.decode(z)
  metrics=reconstruction_metrics(hidden,z,rec)
 f=z[0].cpu().numpy();peak=f.max(axis=0);freq=(f>0).mean(axis=0)
 ids=np.argsort(-peak,kind='stable')[:20].tolist()
 if 1465 not in ids:ids.append(1465)
 features=[]
 for i in ids:
  active=int((f[:,i]>0).sum());t=int(f[:,i].argmax())/50 if active else None
  features.append({'id':i,'peak':float(peak[i]),'peak_time':t,'frequency':float(freq[i]),'active_frames':active,'values':f[:,i].tolist(),'label':'Came-associated candidate' if i==1465 else 'Unmapped feature','candidate':i==1465})
 # Raster mel image generated without global pyplot state.
 from matplotlib.figure import Figure
 from matplotlib.backends.backend_agg import FigureCanvasAgg
 mel=librosa.power_to_db(librosa.feature.melspectrogram(y=wave,sr=16000,n_mels=80,hop_length=160),ref=np.max)
 fig=Figure(figsize=(12,2),dpi=110);FigureCanvasAgg(fig);ax=fig.add_axes([0,0,1,1]);ax.imshow(mel,origin='lower',aspect='auto',cmap='magma',vmin=-80,vmax=0);ax.set_axis_off();png=io.BytesIO();fig.savefig(png,format='png')
 wavbuf=io.BytesIO();sf.write(wavbuf,wave,16000,format='WAV',subtype='PCM_16')
 # Peak-preserving envelope rather than plotting every audio sample.
 stride=max(1,int(np.ceil(len(wave)/1200)));envelope=[]
 for start in range(0,len(wave),stride):
  block=wave[start:start+stride];envelope.append([start/16000,float(block.min()),float(block.max())])
 active_total=int((peak>0).sum())
 return {'name':name,'duration':len(wave)/16000,'frames':len(f),'active_total':active_total,'mean_l0':float(metrics.l0),'fve':float(metrics.frac_var_explained),'relative_l2':float(metrics.normalized_l2),'threshold':float(s.threshold.item()),'features':features,'times':(np.arange(len(f))/50).tolist(),'waveform':envelope,'mel':'data:image/png;base64,'+base64.b64encode(png.getvalue()).decode(),'audio':'data:audio/wav;base64,'+base64.b64encode(wavbuf.getvalue()).decode(),'candidate_active':bool(peak[1465]>0),'model':'facebook/hubert-base-ls960','checkpoint':'layer_11/ae.pt','hidden_index':10}

class Handler(BaseHTTPRequestHandler):
 def reply(self,status,data,kind='application/json'):
  body=json.dumps(data,allow_nan=False).encode() if kind=='application/json' else data
  self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
 def do_GET(self):
  path=urlparse(self.path).path
  if path=='/api/health':return self.reply(200,{'ready':MODEL is not None,'error':ERROR})
  if path=='/api/examples':
   bundled=BASE/'data/examples.json'
   available=json.loads(bundled.read_text()) if bundled.is_file() else {}
   return self.reply(200,[{'id':k,'label':v[0]} for k,v in EXAMPLES.items() if v[1].is_file() or k in available])
  files={'/':'index.html','/results':'index.html','/app.js':'app.js','/style.css':'style.css'}
  if path not in files:return self.reply(404,{'error':'Not found'})
  mime='text/html; charset=utf-8' if files[path].endswith('html') else ('text/css' if files[path].endswith('css') else 'text/javascript')
  return self.reply(200,(ROOT/'static'/files[path]).read_bytes(),mime)
 def do_POST(self):
  if not MODEL:return self.reply(503,{'error':ERROR or 'Models are warming up. Please try again shortly.'})
  path=urlparse(self.path).path
  if path not in ['/api/analyze','/api/example','/api/words/compare']:return self.reply(404,{'error':'Not found'})
  try:length=int(self.headers.get('Content-Length','0'))
  except ValueError:return self.reply(400,{'error':'Invalid content length'})
  if length<=0 or length>20*1024*1024:return self.reply(413,{'error':'Choose an audio file under 20 MB.'})
  # Accept same-origin browser requests only; no CORS is enabled.
  origin=self.headers.get('Origin')
  if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.reply(403,{'error':'Use this page to analyze audio.'})
  if not LOCK.acquire(blocking=False):return self.reply(429,{'error':'Another recording is being analyzed. Please try again shortly.'})
  try:
   raw=self.rfile.read(length)
   if path=='/api/words/compare':
    from words import compare
    result=compare(json.loads(raw),BASE,MODEL)
   elif path=='/api/example':
    key=json.loads(raw).get('id')
    if key not in EXAMPLES:return self.reply(404,{'error':'Example not available'})
    if key not in CACHE:
     if EXAMPLES[key][1].is_file():CACHE[key]=analyze(EXAMPLES[key][1].read_bytes(),EXAMPLES[key][1].name)
     elif (BASE/'data/examples.json').is_file() and key in json.loads((BASE/'data/examples.json').read_text()):CACHE[key]=json.loads((BASE/'data/examples.json').read_text())[key]
     else:return self.reply(404,{'error':'Example not available. Upload your own recording.'})
    result=CACHE[key]
   else:
    if self.headers.get('Content-Type','').split(';')[0]!='application/octet-stream':return self.reply(415,{'error':'Send an audio file as binary data.'})
    result=analyze(raw,'Uploaded recording')
   self.reply(200,result)
  except (ValueError,json.JSONDecodeError) as e:self.reply(400,{'error':str(e)})
  except Exception:
   traceback.print_exc();self.reply(500,{'error':'Analysis failed. See the server log for details.'})
  finally:LOCK.release()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7865);p.add_argument('--host',default='127.0.0.1');args=p.parse_args()
 threading.Thread(target=load,daemon=True).start()
 print(f'Speech Scope is listening at http://127.0.0.1:{args.port}',flush=True)
 ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()
