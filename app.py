"""Gradio-compatible Space entry point serving the Speech Scope frontend."""
import os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('SAE_DEVICE','cpu')
sys.path.insert(0,str(ROOT/'Demo_Code'))
import server
import gradio as gr
import httpx
from fastapi import FastAPI,Request
from fastapi.responses import Response
import uvicorn

threading.Thread(target=server.load,daemon=True).start()
threading.Thread(target=server.ThreadingHTTPServer(('127.0.0.1',7865),server.Handler).serve_forever,daemon=True).start()

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
 body=bytearray()
 async for chunk in request.stream():
  body.extend(chunk)
  if len(body)>20*1024*1024:return Response('Upload limit is 20 MB.',status_code=413)
 headers={k:v for k,v in request.headers.items() if k.lower() not in ['content-length','connection','transfer-encoding','accept-encoding']}
 async with httpx.AsyncClient(timeout=300) as client:
  r=await client.request(request.method,'http://127.0.0.1:7865/'+path,params=request.query_params,content=bytes(body),headers=headers)
 return Response(r.content,status_code=r.status_code,headers={'Content-Type':r.headers.get('content-type','application/json'),'Cache-Control':'no-store'})

if __name__=='__main__':uvicorn.run(app,host='0.0.0.0',port=int(os.environ.get('PORT','7860')))
