import numpy as np
import cv2
import scipy as sc
from matplotlib import pyplot as plt
from scipy import misc
from scipy import signal
from numpy.linalg import inv
from tempfile import TemporaryFile
import settings
import sys

f = file("face.txt").read()
i = 0
rows = cols = 0
X = D = None
n = 0 #no of images in training set

######################  form the vector X

for word in f.split():
	img = cv2.imread(word,0)
	print word
	img = cv2.resize(img,(125,125),interpolation = cv2.INTER_AREA)
	
	#apply dft and shift the dc component from top left to the centre
	dft = np.fft.fft2(img)
	dft = np.fft.fftshift(dft)

	#dft = 20*np.log(cv2.magnitude(dft[:,:,0],dft[:,:,1]))
	if X is None:
		rows,cols = img.shape
		X = dft.flatten()
	else:
		X = np.c_[ X, dft.flatten()]
	n = n + 1

print X.shape

######################   find inverse of matrix D
d = rows * cols
Dinv = np.zeros((d,1))

for i in range(d):
	a = 0
	for j in range(n):
		a = a +  np.absolute(X[i,j])*np.absolute(X[i,j])
	Dinv[i] = 1/(a/n)

###################### find Dinv * X

dim = (d,n)


mul = X
for i in range(d):     #for i in range(d):  if x1[d]!=0 H[1
	mul[i,:]= mul[i,:]*Dinv[i]
###################### find  conjugate of X

Xcon = np.transpose(np.conjugate(X))

##################### now the H filter
print "Xshape",X.shape," ",Xcon.shape," ",mul.shape

u = np.ones((n,1))
H = np.dot (  np.dot(mul, np.linalg.inv(np.dot(Xcon, mul)) ),  u)
#H = (mul *  np.linalg.inv(Xcon*mul) )* u

H = np.reshape(H, (rows, cols))

#################### save the filter in folder and add in filter.txt
print "filter name : ",sys.argv[1]
x = np.save('./filters/'+sys.argv[1]+'.npy',H)

text_file = open("filter_list.txt", "a")
text_file.write("%s\n" %sys.argv[1])
print H[0,0]

#plt.subplot(2,4,1),plt.imshow(H, cmap = 'gray')
#plt.title('Spectrum of training image'), plt.xticks([]), plt.yticks([])

#plt.show()
