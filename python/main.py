import matplotlib.pyplot as plt
import cv2
import sys
from helper import *
from train import *
#carregar imagem com referencias
test_img = cv2.imread('digits.png',0)
test_cropped=[]
test_result=[]
for t in range(0,10):
	test_last=test_img[:,80*t:80*(t+1)]
	test_cropped.append(test_last)
	test_boxes, test_contornos = separados(test_last)
	test_result.append(test_contornos[0])
#carregar imagem a ser analizada
query_img = cv2.imread(sys.argv[1],0)
#display(query_img)
query_boxes, query_contornos = separados(query_img)
#min_size e o minimo tamanho do ruido 
for query_t in range(0,len(query_contornos)):
	query_c=query_contornos[query_t]
	query_b=query_boxes[query_t]
	min_dist=sys.float_info.max
	tmin=0
	for t in range(0,10):
		#d1=cv2.matchShapes(query_c,test_result[t], 1 ,0.0)
		#d2=cv2.matchShapes(query_c,test_result[t], 2 ,0.0)
		d3=cv2.matchShapes(query_c,test_result[t], 3 ,0.0)
		actual_dist=d3
		if actual_dist<min_dist:
			min_dist=actual_dist
			tmin=t
	print tmin, ' na posicao (',query_b[0],',',query_b[1],')'
cv2.destroyAllWindows()


