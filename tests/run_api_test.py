import sys
from pathlib import Path
# ensure project root on path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print('GET /')
resp = client.get('/')
print(resp.status_code, resp.json())

wav_path = project_root / 'Demo_samples' / 'demo.wav'
print('Posting:', wav_path)
with open(wav_path, 'rb') as f:
    files = {'file': ('demo.wav', f, 'audio/wav')}
    resp = client.post('/audio/predict', files=files)
    print('Status:', resp.status_code)
    try:
        print('JSON:', resp.json())
    except Exception as e:
        print('Response text:', resp.text)

# also test /audio/upload
with open(wav_path, 'rb') as f:
    files = {'file': ('demo.wav', f, 'audio/wav')}
    resp = client.post('/audio/upload', files=files)
    print('/audio/upload status:', resp.status_code)
    try:
        print('/audio/upload json:', resp.json())
    except Exception:
        print('/audio/upload text:', resp.text)
