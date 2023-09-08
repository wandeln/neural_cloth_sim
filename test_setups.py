from setups import Dataset
import matplotlib.pyplot as plt

h,w = 64,64

dataset = Dataset(h,w)

x_v,M,_ = dataset.ask()

for i in range(3):
	plt.subplot(1,3,i+1)
	plt.imshow(x_v[0,i])
plt.show()
