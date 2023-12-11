import os
import matplotlib.pyplot as plt

infos = os.listdir("info")
data = []
for i in infos:
    b = []
    l = []
    with open("info/"+i, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            b.append(int(line[0]))
            l.append(float(line[1]))
    data.append((b,l))
print(len(data))
avloss = [sum(x[1])/len(x[1]) for x in data]

res = 10
for i , d in enumerate(data):
    plt.subplot(3,4,i+1)
    plt.title(f"batch {i+1}")
    plt.plot(d[0][::res],d[1][::res])   

plt.subplot(3,4,12)
plt.title("avrege to all")
plt.plot(range(1,len(avloss)+1),avloss) 
plt.show() 