import io, os, tempfile
from pathlib import Path
import pandas as pd

# Isolate persistent state before importing app modules.
os.environ['MODELFORGE_HOME'] = tempfile.mkdtemp(prefix='mf_v30_test_')
from fastapi.testclient import TestClient
from modelforge.api import app

client = TestClient(app)

def test_signup_and_dataset_workflow():
    r=client.post('/api/auth/signup',json={'email':f"researcher_{os.getpid()}@example.com",'password':'strongpass123'})
    assert r.status_code==200
    assert r.json()['quota']['quota_bytes']==1024**3
    csv=b'a,b,label\n1,2,x\n2,,y\n2,,y\n3,9,x\n'
    r=client.post('/api/datasets/upload',files={'file':('tiny.csv',csv,'text/csv')})
    assert r.status_code==200, r.text
    did=r.json()['dataset']['id']
    p=client.get(f'/api/datasets/{did}/preview')
    assert p.status_code==200 and len(p.json()['rows'])==4
    st=client.get(f'/api/datasets/{did}/stats').json()
    col={x['name']:x for x in st['columns']}
    assert col['b']['missing']==2
    f=client.post(f'/api/datasets/{did}/filter',json={'column':'a','op':'gte','value':'2','limit':20})
    assert f.status_code==200 and f.json()['matched']==3
    c=client.post(f'/api/datasets/{did}/clean',json={'drop_duplicates':True,'drop_missing':True,'drop_columns':[]})
    assert c.status_code==200
    assert c.json()['dataset']['rows']==2
    ch=client.get(f'/api/datasets/{did}/chart?column=a')
    assert ch.status_code==200 and ch.json()['kind']=='histogram'

def test_auth_required_for_datasets():
    other=TestClient(app)
    r=other.get('/api/datasets')
    assert r.status_code==401

def test_beginner_assets_present():
    root=Path(__file__).resolve().parents[1]
    assert (root/'beginner'/'cnn_model.json').exists()
    html=(root/'beginner'/'index.html').read_text()
    for token in ['Conv2d','ReLU','MaxPool2d','BatchNorm2d','LayerNorm','MultiheadAttention','ResidualBlock']:
        assert token in html
