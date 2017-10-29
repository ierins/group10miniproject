from sklearn import datasets 
from sklearn.svm import SVC
from scipy import misc
import numpy as np


digits = datasets.load_digits()
features = digits.data
labels = digits.target
print("feature....................................................................................................")
print(features)
print("label.........................................................................................................")
print(labels)

print("Rows are..........................................................")
extractedData = features[1,:]
#extractedData = features[:,[1,4]]

print(extractedData)
clf = SVC(gamma=0.001)
clf.fit(features,labels)

print(features.shape)

img = misc.imread("img4.jpg")
img = misc.imresize(img, (8,8))
print("Image is ..........................................................................................")
print(img)
print("Datatype of image is ...............................................................................")
print(img.dtype)


img = img.astype(digits.images.dtype)
print("New Datatype is ....................................................................................")
print(img.dtype)
img = misc.bytescale(img, high=16, low=0)
print(img)

arr=np.array(img)
b = np.reshape(arr, (1,np.product(arr.shape)))
#print(b.dtype)
print(clf.predict(b))
#print(b.dtype)
#print(b)

#x_test = []
#for eachRow in arr:
 #for eachPixel in eachRow:
	#x_test.append(sum(eachRow)/8.0)
#print(x_test)

#print(clf.predict([x_test]))

#print(clf.predict(b))
