import csv
import numpy as np


with open('/home/fangsh/tianchi/tianchi_dataset/submit/submit0725.csv') as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
        data.append(row)



filename = []
res1 = []
res2 = []

for i in data:
    filename.append(i[0])
    res1.append(i[1])
    res2.append(i[2])
filename.pop(0)
res1.pop(0)
res2.pop(0)
res1 = [float('%.6f'%float(ele)) for ele in res1]
res2 = [float('%.6f'%float(ele)) for ele in res2]
print(res1)
p1 = [1 if ele > 0.5 else 0 for ele in res1]
p2 = [1 if ele > 0.5 else 0 for ele in res2]
p = np.equal(p1,p2)
acc = np.sum(p)/len(p)
print(acc)
res2 = [0.999880 if ele ==1 else ele for ele in res2]
res2 = [0.000112 if ele ==0 else ele for ele in res2]
print(res2)
with open('/home/fangsh/test/sub1.csv','a',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(('filename','probability'))
    for idx in range(len(res2)):
        writer.writerow((filename[idx],res2[idx]))