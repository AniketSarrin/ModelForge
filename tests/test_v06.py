import io
from PIL import Image
from fastapi.testclient import TestClient
from modelforge.api import app

client=TestClient(app)

def test_activation_endpoint_image_and_unused_modalities():
    im=Image.new("RGB",(64,64),(128,64,32))
    buf=io.BytesIO(); im.save(buf,format="PNG")
    r=client.post("/api/activate",data={"model":"resnet18","text":"hello","image_size":"64"},files={"image":("x.png",buf.getvalue(),"image/png")})
    assert r.status_code==200
    j=r.json()
    assert "image" in j["consumed_modalities"]
    assert "text" in j["unused_modalities"]
    assert j["output"]["shape"]==[1,1000]
    assert "conv1" in j["activations"]
    assert 0 <= j["activations"]["conv1"]["normalized"] <= 1
