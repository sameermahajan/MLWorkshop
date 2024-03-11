import requests
import sys

file = "example1.txt"

if len(sys.argv) > 1:
    file = sys.argv[1]

with open(file) as f:
    print (file, " opened")
    text = "\n".join(f.readlines())

print (file, " read")
data = {"text": text}
response = requests.post("http://127.0.0.1:8000/analyze", json=data)
print ("request made")
print(response.text)