
import json

list = []

for i in range(100):
    i += 7
    print(i)

'''
with open('E:/logData/test-with label/bd33-40/20190109.log', 'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        print(data['severity_label'] == 'Informational')
'''