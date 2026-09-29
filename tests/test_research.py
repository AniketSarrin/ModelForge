from fastapi.testclient import TestClient
from modelforge.api import app

client=TestClient(app)

def test_research_run_probe_and_intervention():
    payload={
        'model':'resnet18','image_size':64,'name':'test',
        'probes':[{'path':'layer2.0.conv1','kind':'activation','top_k':4}],
        'interventions':[{'path':'layer2.0.conv1','kind':'scale','value':0.5}],
        'capture_gradients':False,'seed':1
    }
    r=client.post('/api/research/run',json=payload)
    assert r.status_code==200, r.text
    j=r.json()
    assert 'id' in j
    assert 'layer2.0.conv1' in j['result']['captures']
    assert j['result']['output']['shape']==[1,1000]

def test_parameter_inventory_and_profile():
    r=client.get('/api/parameters/resnet18?q=conv1')
    assert r.status_code==200
    assert r.json()['count']>=1
    p=client.post('/api/profile',json={'model':'resnet18','image_size':64,'root':''})
    assert p.status_code==200
    assert p.json()['total_ms']>0
    assert len(p.json()['modules'])>0

def test_experiment_and_compare_runs():
    base={'model':'resnet18','image_size':64,'probes':[],'interventions':[],'capture_gradients':False,'seed':0}
    a=client.post('/api/research/run',json={**base,'name':'a'}).json()['id']
    b=client.post('/api/research/run',json={**base,'name':'b','interventions':[{'path':'layer4','kind':'scale','value':0.1}]}).json()['id']
    c=client.post('/api/runs/compare',json={'run_a':a,'run_b':b})
    assert c.status_code==200
    e=client.post('/api/experiments',json={'name':'branch','baseline_run':a,'graph_diff':{'scale':0.1}})
    assert e.status_code==200

def test_plugins_and_python_export():
    p=client.get('/api/plugins')
    assert p.status_code==200 and len(p.json()['plugins'])>=5
    e=client.get('/api/export/python/resnet18')
    assert e.status_code==200
    assert 'def build_model()' in e.json()['content']
    assert 'def load_weights' in e.json()['content']
    compile(e.json()['content'], 'model.py', 'exec')
    b=client.get('/api/export/python-bundle/resnet18')
    assert b.status_code==200
    import io, zipfile
    z=zipfile.ZipFile(io.BytesIO(b.content))
    assert {'model.py','weights.pth','README.txt'} <= set(z.namelist())
