import http.client
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

conn = http.client.HTTPConnection('localhost', 8000, timeout=10)
img = r'C:\Users\Admin\Documents\GitHub\AI-Tools-Project\data\processed\Master_Plate_Dataset\images\test\egyptian_motorcycles__138297047_2868057803476869_7008108065211707713_n_jpg.rf.cba07483b4f0bfeccc0e4f46f77b690b.jpg'
boundary = '----FormBoundary7MA4YWxkTrZu0gW'
with open(img, 'rb') as f:
    data = f.read()
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + data + (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="conf"\r\n\r\n0.05\r\n--{boundary}--\r\n').encode()
conn.request('POST', '/api/detect', body, {'Content-Type': f'multipart/form-data; boundary={boundary}'})
resp = conn.getresponse()
print(f'Status: {resp.status}')
print(f'Body: {resp.read().decode(errors="replace")[:500]}')
