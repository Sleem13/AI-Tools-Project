import http.client
import json
import glob
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

imgs = glob.glob(r'C:\Users\Admin\Documents\GitHub\AI-Tools-Project\data\processed\Master_Plate_Dataset\images\test\*')
if not imgs:
    print("No test images found")
    exit(1)

detected = 0
total = 0
for img_path in imgs[:20]:
    fname = os.path.basename(img_path)[:60]
    boundary = '----FormBoundary7MA4YWxkTrZu0gW'
    with open(img_path, 'rb') as f:
        file_data = f.read()

    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(img_path)}"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode() + file_data + (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="conf"\r\n\r\n0.05\r\n--{boundary}--\r\n'
    ).encode()

    total += 1
    try:
        conn = http.client.HTTPConnection('localhost', 8000, timeout=10)
        conn.request('POST', '/api/detect', body, {'Content-Type': f'multipart/form-data; boundary={boundary}'})
        resp = conn.getresponse()
        raw = resp.read()
        data = json.loads(raw)
        
        n = len(data['detections'])
        ms = data['processing_time_ms']
        if n > 0:
            detected += 1
            plates = [d['plate_text'] for d in data['detections']]
            src = [d.get('text_source', '?') for d in data['detections']]
            print(f"[OK] {fname}: {n} det, {ms:.0f}ms, plates={plates}, source={src}")
        else:
            print(f"[--] {fname}: 0 det, {ms:.0f}ms")
    except Exception as e:
        print(f"[ERR] {fname}: {e}")
    finally:
        try: conn.close()
        except: pass

print(f"\nSummary: {detected}/{total} images had detections")
