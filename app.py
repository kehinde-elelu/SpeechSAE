"""Gradio-compatible Space entry point serving the Speech Scope frontend."""
import os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('SAE_DEVICE','cpu')
sys.path.insert(0,str(ROOT/'Demo_Code'))
import server
import gradio as gr
import json
from fastapi import FastAPI,Request
from fastapi.responses import Response
import uvicorn

threading.Thread(target=server.load,daemon=True).start()

def inspect_file(path):
 if server.MODEL is None:raise gr.Error('Models are warming up. Please try again shortly.')
 if path is None:raise gr.Error('Choose an audio recording.')
 if Path(path).stat().st_size>20*1024*1024:raise gr.Error('Use a file below 20 MB.')
 with server.LOCK:
  result=server.analyze(Path(path).read_bytes(),'Uploaded recording')
 return {k:v for k,v in result.items() if k not in ['audio','mel','waveform','times','features']}|{'features':[{'id':f['id'],'peak':f['peak'],'peak_time':f['peak_time'],'frequency':f['frequency']} for f in result['features']]}

studio=gr.Interface(inspect_file,inputs=gr.Audio(type='filepath'),outputs=gr.JSON(),title='SpeechSAE',description='A native Gradio interface. Open the main page for interactive plots and word comparisons.',flagging_mode='never')
app=gr.mount_gradio_app(FastAPI(),studio,path='/gradio')

@app.api_route('/{path:path}',methods=['GET','POST'])
async def proxy(path:str,request:Request):
 from starlette.concurrency import run_in_threadpool
 if request.method=='GET':
  if path=='api/health':return {'ready':server.MODEL is not None,'error':server.ERROR}
  if path=='api/examples':
   bundled=ROOT/'data/examples.json';available=json.loads(bundled.read_text()) if bundled.is_file() else {}
   return [{'id':k,'label':v[0]} for k,v in server.EXAMPLES.items() if v[1].is_file() or k in available]
  files={'':'index.html','results':'index.html','style.css':'style.css','app.js':'app.js'}
  if path not in files:return Response('Not found',status_code=404)
  name=files[path];mime='text/html' if name.endswith('html') else ('text/css' if name.endswith('css') else 'text/javascript')
  return Response((ROOT/'Demo_Code/static'/name).read_bytes(),media_type=mime)
 if path not in ['api/analyze','api/example','api/words/compare']:return Response('Not found',status_code=404)
 if server.MODEL is None:return Response(json.dumps({'error':server.ERROR or 'Models are warming up. Please try again shortly.'}),status_code=503,media_type='application/json')
 body=bytearray()
 async for chunk in request.stream():
  body.extend(chunk)
  if len(body)>20*1024*1024:return Response('Upload limit is 20 MB.',status_code=413)
 if not body:return Response(json.dumps({'error':'Choose an audio file or comparison.'}),status_code=400,media_type='application/json')
 if not server.LOCK.acquire(blocking=False):return Response(json.dumps({'error':'Another recording is being analyzed. Please try again shortly.'}),status_code=429,media_type='application/json')
 def work():
  try:
   if path=='api/analyze':
    if request.headers.get('content-type','').split(';')[0]!='application/octet-stream':raise ValueError('Send an audio file as binary data.')
    return server.analyze(bytes(body),'Uploaded recording')
   payload=json.loads(body)
   if path=='api/words/compare':
    from words import compare
    return compare(payload,ROOT,server.MODEL)
   key=payload.get('id');bundled=ROOT/'data/examples.json';examples=json.loads(bundled.read_text()) if bundled.is_file() else {}
   if key in examples:return examples[key]
   if key in server.EXAMPLES and server.EXAMPLES[key][1].is_file():return server.analyze(server.EXAMPLES[key][1].read_bytes(),server.EXAMPLES[key][1].name)
   raise ValueError('Example unavailable. Upload your own recording.')
  finally:server.LOCK.release()
 try:
  result=await run_in_threadpool(work)
  return Response(json.dumps(result,allow_nan=False),media_type='application/json',headers={'Cache-Control':'no-store'})
 except ValueError as e:return Response(json.dumps({'error':str(e)}),status_code=400,media_type='application/json')
 except Exception:
  import traceback;traceback.print_exc()
  return Response(json.dumps({'error':'Analysis failed. Please try another recording.'}),status_code=500,media_type='application/json')

if __name__=='__main__':uvicorn.run(app,host='0.0.0.0',port=int(os.environ.get('PORT','7860')))
