import torch
from fastapi.testclient import TestClient
from modelforge.api import app
from modelforge.models import load_model
from modelforge.stitch import build_hybrid


def test_hybrid_forward():
    a = load_model("resnet18")
    b = load_model("resnet34")
    model, plan = build_hybrid(a, "layer2", b, "layer3", image_size=64)
    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1000)
    assert plan.source_channels > 0
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(p.numel() for p in model.adapter.parameters())


def test_hierarchy_drilldown():
    client = TestClient(app)
    top = client.post("/api/hierarchy", json={"model":"resnet18","image_size":64,"root":""})
    assert top.status_code == 200
    names = [x["name"] for x in top.json()["children"]]
    assert "layer2" in names
    deep = client.post("/api/hierarchy", json={"model":"resnet18","image_size":64,"root":"layer2"})
    assert deep.status_code == 200
    assert len(deep.json()["children"]) == 2
    assert deep.json()["children"][0]["has_children"] is True


def test_api_plan_and_train():
    client = TestClient(app)
    payload={"source":"resnet18","source_cut":"layer2","target":"resnet34","target_entry":"layer3","image_size":64}
    plan=client.post("/api/plan",json=payload)
    assert plan.status_code==200
    assert plan.json()["params"]>0
    train=client.post("/api/train",json={**payload,"steps":1,"batch_size":1})
    assert train.status_code==200
    assert train.json()["training"]["output_shape"]==[1,1000]
