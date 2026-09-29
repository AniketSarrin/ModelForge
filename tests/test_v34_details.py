from fastapi.testclient import TestClient
from modelforge.api import app

client = TestClient(app)

def test_resnet_paper_details():
    r = client.post('/api/hierarchy', json={'model':'resnet18','image_size':64,'root':''})
    assert r.status_code == 200
    by_name = {x['name']: x for x in r.json()['children']}
    assert by_name['conv1']['paper_label'].startswith('Conv 7×7, 64 /2')
    assert by_name['conv1']['details']['in_channels'] == 3
    assert by_name['conv1']['details']['out_channels'] == 64
    assert by_name['layer1']['paper_label'] == 'BasicBlock ×2'
