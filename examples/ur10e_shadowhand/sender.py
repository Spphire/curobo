import json
import requests

q = requests.post("http://10.53.21.95:8000/getJoints")
print(list(json.loads(q.content).keys()))
print(list(json.loads(q.content).values()))