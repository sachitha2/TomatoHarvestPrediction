import requests
BASE = "http://127.0.0.1:5000/"

res = requests.post(BASE+'hello')

print(res.json())

