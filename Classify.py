import json

f = open('QS', 'r',encoding = 'latin-1')
c = ''
s = ''
data = {}
for i in range(28910):
    x = f.readline()
    if ('--' in x):
        x = x.split('--')
        c = x[1]
    else:
        s = x
    if (c != '' and s != ''):
        try:
            blah = data[c]
        except:
            data[c] = []
        data[c].append(s)
        c = ''
        s = ''

with open('dump.json', 'w') as fp:
    json.dump(data, fp)
