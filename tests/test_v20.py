import io, tempfile
import torch
import torch.nn as nn
from PIL import Image
from fastapi.testclient import TestClient
from modelforge import open as mf_open
from modelforge.api import app
from modelforge.store import DB_PATH

client=TestClient(app)

def png_bytes(v=128):
    im=Image.new('RGB',(32,32),(v,v,v)); b=io.BytesIO(); im.save(b,format='PNG'); return b.getvalue()

def test_arbitrary_nn_module_session():
    m=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,2))
    s=mf_open(m)
    r=s.run(torch.ones(1,4),probes=[{'path':'0'}],interventions=[{'path':'0','kind':'scale','value':0.5}])
    assert r['output']['shape']==[1,2]
    assert '0' in r['captures']

def test_real_input_research_endpoint():
    fd={'model':(None,'resnet18'),'name':(None,'real input'),'image_size':(None,'32'),'seed':(None,'1'),'capture_gradients':(None,'false'),'probes_json':(None,'[{"path":"conv1","kind":"activation"}]'),'interventions_json':(None,'[]'),'text':(None,'') ,'image':('x.png',png_bytes(), 'image/png')}
    r=client.post('/api/research/run-input',files=fd)
    assert r.status_code==200,r.text
    j=r.json(); assert j['result']['input']['consumed_modalities']==['image']; assert 'conv1' in j['result']['captures']

def test_run_persistence_and_codegen():
    runs=client.get('/api/runs').json()['runs']; assert runs
    rid=runs[0]['id']; code=client.get(f'/api/runs/{rid}/code').json()['content']; compile(code,'run.py','exec')
    assert DB_PATH.exists()

def test_patch_real_images():
    files={'clean':('clean.png',png_bytes(220),'image/png'),'corrupted':('corrupt.png',png_bytes(20),'image/png')}
    data={'model':'resnet18','path':'layer1.0.conv1','image_size':'32'}
    r=client.post('/api/patch/image',files=files,data=data)
    assert r.status_code==200,r.text
    assert 'l2_delta' in r.json()

def test_sweep_reuses_input_seed():
    r=client.post('/api/sweep',json={'model':'resnet18','path':'layer1','kind':'scale','values':[0.5,1.0],'image_size':32})
    assert r.status_code==200; assert r.json()['input_seed']==0
