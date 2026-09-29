import io
import torch
from torchvision import models
from fastapi.testclient import TestClient
from modelforge.api import app

client=TestClient(app)

def test_health_v03():
    r=client.get('/api/health'); assert r.status_code==200; assert r.json()['version']=='3.6.0'

def test_units_conv_and_linear():
    r=client.post('/api/units',json={'model':'resnet18','path':'conv1','count':4}); assert r.status_code==200; assert r.json()['kind']=='channel'; assert len(r.json()['units'])==4
    r=client.post('/api/units',json={'model':'resnet18','path':'fc','count':4}); assert r.status_code==200; assert r.json()['kind']=='neuron'; assert len(r.json()['units'])==4

def test_import_state_dict():
    buf=io.BytesIO(); torch.save(models.resnet18(weights=None).state_dict(),buf); buf.seek(0)
    r=client.post('/api/import',files={'file':('r18.pth',buf.getvalue(),'application/octet-stream')}); assert r.status_code==200; j=r.json(); assert j['architecture']=='resnet18'
    h=client.post('/api/hierarchy',json={'model':j['model'],'image_size':64,'root':''}); assert h.status_code==200; assert any(x['name']=='layer2' for x in h.json()['children'])
