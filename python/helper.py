import cv2
import numpy as np
import random
def hog(img):
	bin_n = 2
	gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)     # hist is a 64 bit vector
	return hist
def array_hog(arr):
	result = []
	for img in arr:
		result.append(hog(img))
	return result
		
def display(img,txt="image"):
	cv2.imshow(txt,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def separados(imgray):
	imgray = cv2.medianBlur(imgray,3)
	thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,55,15)
	#display(thresh)
	_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	out = np.zeros_like(imgray.copy()).astype('uint8')
	cv2.drawContours(out, contours, -1, (255,0,255), 1)
	#display(out)
	result = []
	boxes=imgray.copy()
	for t in range(len(contours)) :
		x,y,w,h = cv2.boundingRect(contours[t])
		result.append((x,y,w,h))
		cv2.rectangle(boxes,(x,y),(x+w,y+h),128,1)
	#display(boxes)
	random.shuffle(result)
	return result, contours
	
