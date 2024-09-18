import numpy as np

# 4.1
N = int(1e6)
x = np.random.uniform(size=N)
y = 2*np.random.uniform(size=N)
res = 2*1*np.mean(np.sin(x**2+y**2))
print(res)

# 4.2
N = int(1e6)
r  = np.random.uniform(size=N)
th = 2*np.pi*np.random.uniform(size=N)
x,y = np.sqrt(r)*np.cos(th),np.sqrt(r)*np.sin(th)
res = np.pi*1**2*np.mean(np.exp(-(x**2+y**2)))
print(res)

# 4.3
N = int(1e6)
x = np.random.uniform(size=N)
y = np.random.uniform(size=N)
cond = y < 1-x
res = 1*1*np.mean((x+y)*cond)
print(res)

# 4.4
N  = int(1e6)
x1 = np.random.uniform(size=N)
x2 = np.random.uniform(size=N)
x3 = np.random.uniform(size=N)
x4 = np.random.uniform(size=N)
res = 1*1*1*1*np.mean(np.exp(-(x1**2 + x2**2 + x3**2 + x4**2)))
print(res)

# 4.5
N  = int(1e6)
x1 = np.random.uniform(size=N)
x2 = np.random.uniform(size=N)
x3 = np.random.uniform(size=N)
x4 = np.random.uniform(size=N)
cond = (x1**2 + x2**2 + x3**2 + x4**2) < 1
res = 2*2*2*2*np.mean(cond*(x1**2 + x2**2 + x3**2 + x4**2)*np.exp(-(x1**2 + x2**2 + x3**2 + x4**2)))
print(res)
