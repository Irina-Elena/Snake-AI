import matplotlib.pyplot as plt




if __name__ == '__main__':
    x  = plt.plot([1,2,3,2,4,1])
    plt.axis([0,4,min([1,2,3,2,4,1]),max([1,2,3,2,4,1])])
    plt.show()