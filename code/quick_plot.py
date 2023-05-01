import matplotlib.pyplot as plt

seed = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
val = [3.60, 7.34, 7.04, 6.13, 3.78, 2.90, 4.19, 5.23, 5.93, 6.71, 6.18, 5.44]

plt.scatter(seed,val)
plt.xlabel('Seed Number')
plt.ylabel('Pixel Error on Real Dataset')
plt.title('Impact of Random Seed on Generalization to Real Dataset')
plt.grid()
plt.show()
