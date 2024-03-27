import json
import requests
# q = {"valid": False}
#while not q["valid"]:
q = requests.post("http://10.9.11.1:8000/get_tip_tcp", json.dumps({"name": "indexTip"}))
    #q=json.loads(q.content)
    #print(q)
#q = requests.post("http://10.9.11.1:8000/getJoints")
print(q.content)
print(list(json.loads(q.content).keys()))
print(list(json.loads(q.content).values()))