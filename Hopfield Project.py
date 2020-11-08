import numpy as np
import matplotlib.pyplot as plt

M=np.array([[1,1,-1,1,1],[1,-1,1,-1,1],[1,-1,1,-1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1],[1,-1,-1,-1,1]])
A=np.array([[-1,-1,1,-1,-1],[-1,1,-1,1,-1],[-1,1,-1,1,-1],[-1,1,1,1,-1],[-1,1,-1,1,-1],[-1,1,-1,1,-1],[-1,1,-1,1,-1]])

plt.subplot(1,2,1)
plt.imshow(M, cmap=plt.cm.binary)
plt.title("Character M")

plt.subplot(1,2,2)
plt.imshow(A, cmap=plt.cm.binary)
plt.title("Character A")
plt.title("Character A")

plt.show()


M_vector=np.reshape(M,(1,-1))
A_vector=np.reshape(A,(1,-1))

W=np.multiply(M_vector.T,M_vector)+np.multiply(A_vector.T,A_vector)
np.fill_diagonal(W, 0)
print("weight matrix is:",W,sep='\n')


def detector(X,W):
    x=X
    y=+X
    t= list(np.random.permutation(35))
    Temp_old=np.zeros((1,35))
    Temp_new=y
    Count=0

    while np.sum(Temp_new-Temp_old) != 0:
        Temp_old=Temp_new
        for i in t:
            y_in=x[0,i]+np.dot(W[i,:],y[0,:].T)

            if y_in > 0:
                y[0,i]=+1
            elif y_in<0:
                y[0,i]=-1
            else:
                y[0,i]=y[0,i]
        Count+=1
        Temp_new=y

    print("Number of iterations:",Count)
    
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(x,(7,5)), cmap=plt.cm.binary)
    plt.title("Input")
    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(y,(7,5)), cmap=plt.cm.binary)
    plt.title("Detected Output")
    plt.show()
    
    return

detector(M_vector,W)
detector(A_vector,W)

M_noise=np.array([[-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,-1,-1,-1,1]])
detector(M_noise,W)

A_noise=np.array([[1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1]])
detector(A_noise,W)

M_noise_random=np.array([[1,1,-1,1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]])+ 0.6*np.random.randn(1,35)
detector(M_noise_random,W)

A_noise_random=np.array([[-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1]])+ 1.3*np.random.randn(1,35)
detector(A_noise_random,W)
