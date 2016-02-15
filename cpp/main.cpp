#include <iostream>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/shape/shape.hpp> 
#include <opencv2/ml/ml.hpp> 

std::vector<std::vector<cv::Mat> > gridImage(cv::Mat totalImage,int rows,int cols){
	cv::Size size  = totalImage.size();
	std::vector<std::vector<cv::Mat> > matrix = std::vector<std::vector<cv::Mat> >();
	for(int i=0 ; i<rows ; i++){
		std::vector<cv::Mat> sameDigits = std::vector<cv::Mat>();
		for(int j=0 ; j<cols ; j++){
				cv::Rect_<int> rect = cv::Rect_<int>(
					j*(size.width/(cols)),
					i*(size.height/(rows)),
					(size.width/cols),
					(size.height/rows)
				);
				sameDigits.push_back(totalImage(rect));
				//cv::imshow("img",sameDigits[j]);
				//cv::waitKey(0);
				//std::cout<<"gridImage::bloco "<<i/5<<std::endl;
		}
		matrix.push_back(sameDigits);
	}
	return matrix;
}

/*float shapeDistance(cv::Mat i1,cv::Mat i2){
	cv::ShapeDistanceExtractor extractor;
	
	std::vector<std::vector<cv::Point> >  contours1;
	cv::findContours( i1, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	
	std::vector<std::vector<cv::Point> >  contours2;
	cv::findContours( i2, contours2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


}*/

bool secondComparission(std::pair<int,double> a,std::pair<int,double> b){
	return a.second<b.second;
}

std::vector<std::vector<double> > computeDistances(std::vector<std::vector<cv::Mat> > matrix,cv::Mat inputImage){
	std::vector<std::vector<double> > distances = std::vector<std::vector<double> >();
	std::cout<<"computeDistance::distancias criadas"<<std::endl;
	for(int i=0 ; i<matrix.size() ; i++){
		std::vector<double> row = std::vector<double>();
		for(int j=0 ; j<matrix[0].size() ; j++){
			row.push_back(cv::matchShapes(inputImage,matrix[i][j],3,0));
//			if(cv::matchShapes(inputImage,matrix[i][j],1,0)<.0001){
//				std::cout<<"computeDistances::valor pequeno de: "<<cv::matchShapes(inputImage,matrix[i][j],1,0)<<std::endl;
//				cv::imshow("deu zero!",matrix[i][j]);
//				cv::waitKey(0);
//			}
		}
		distances.push_back(row);
	}	
	std::cout<<"computeDistance::distancias calculadas"<<std::endl;
	return distances;
}
/*
int neuralNetClassifier(std::vector<std::vector<cv::Mat> > matrix, std::vector<std::vector<double> > distances,int k){
	cv::CvAnn_MLP mlp;
	
	cv::CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;
	criteria.type = cv::CV_TERMCRIT_ITER | cv::CV_TERMCRIT_EPS;

	cv::CvANN_MLP_TrainParams params;
	params.train_method = cv::CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	cv::Mat layers = cv::Mat(4,1,cv::CV_32SC1);
	layers.row(0) = cv::Scalar(2);
	layers.row(0) = cv::Scalar(10);
	layers.row(0) = cv::Scalar(15);
	layers.row(0) = cv::Scalar(1);

	mlp.create(layers);
	
	mlp.train(trainData,trainingClasses, cv::Mat(), cv::Mat(), params);

	cv::Mat response(1,1,cv::CV_32SC1);
	cv::Mat predicted(testClasses.rows,1,cv::CV_32SC1);
	for(int i=0;i<testData.rows,i++){
		cv::Mat response(1,1,cv::CV_32SC1);
		cv::Mat sample = testData.row(i);

		mlp.predict(sample,response);
		predicted.at<float>(i,0)=response.at<float>(0,0);

	}



}
*/
int knnClassifier(std::vector<std::vector<cv::Mat> > matrix, std::vector<std::vector<double> > distances,int k){
	std::vector< std::pair<int,double> > ret = std::vector< std::pair<int,double> >(); 
	for(int i=0 ; i<distances.size() ; i++){
		for(int j=0 ; j<distances[0].size() ; j++){
			ret.push_back(std::pair<int,double>(i/5,distances[i][j]));
			//verificar incoerência em pair
			if(i*distances[0].size()+j!=ret.size()-1){
				std::cout<<i*distances[0].size()+j<<"("<<i<<"*w+"<<j<<") == "<<ret.size()-1<<std::endl;
				std::cout<<"ERRO: INCOERÊNCIA"<<std::endl;
				return 0;
			}
		}
	}
	for(int t=0;t<ret.size();t++){
		if(ret[t].second==0){
			std::cout<<ret[t].first<<" , "<<ret[t].second<<std::endl;
		}
	}
	std::sort(ret.begin(),ret.end(),secondComparission);
	std::cout<<"knnClassifier::distancias ordenadas"<<std::endl;
	double looksLike[20000];
	for(int t=0;t<20000;t++){
		looksLike[t]=0;	
	}
	for(int t=0;t<ret.size();t++){
		if(ret[t].second==0){
			std::cout<<ret[t].first<<" , "<<ret[t].second<<" : "<<t<<std::endl;
		}
	}
	double max=0;
	int maxId=0;
	for(int t=0; t<k; t++){
		std::cout<<ret[t].first<<" , "<<ret[t].second<<std::endl;
		looksLike[ret[t].first]+=ret[t].second/((t+1)*(t+1));
		if(looksLike[ret[t].first]>max){
			max=looksLike[ret[t].first];
			maxId = ret[t].first;
		}
	}
	//cv::imshow("Resultado do knn",matrix[5*maxId][0]);
	//cv::waitKey(0);
	return maxId;

}

int main(int argc, char** argv){
	int thresh = 0;
	cv::Mat inputImage = cv::imread(argv[1],2);
	cv::Mat edited ;
	cv::GaussianBlur(inputImage,edited,cv::Size_<int>(21,41),0,0);
	cv::imwrite("gaussian.png",edited);
	cv::medianBlur(inputImage,edited,21);
	cv::imwrite("median.png",edited);
	cv::blur(inputImage,edited,cv::Size_<int>(21,41));
	cv::imwrite("mean.png",edited);
	cv::GaussianBlur(inputImage,edited,cv::Size_<int>(21,41),0,0);
	cv::imwrite("gaussian.png",edited);
	cv::adaptiveThreshold(inputImage,edited,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,201,0);
	cv::imwrite("adaptiveThreshold.png",edited);
	//cv::resize(inputImage,edited,cv::Size(20,20));
	cv::Sobel(inputImage,edited,-1,1,1);
	cv::Laplacian(inputImage,edited,-1,5,1,0,cv::BORDER_DEFAULT);
	cv::threshold(inputImage, edited,thresh, 255,cv::THRESH_BINARY);
	cv::imwrite("threshold.png",edited);
	cv::Canny(inputImage,inputImage, thresh, thresh);
	
	
	cv::Mat scaled;
	cv::resize(inputImage,scaled,cv::Size(500,500),0,0,cv::INTER_NEAREST);
	cv::imshow("imagem original",scaled);	
	cv::waitKey(0);
	
	cv::Mat trainImage = cv::imread("../../data/train3.png",2);
	//cv::Laplacian(trainImage,trainImage,-1,5,1,0,cv::BORDER_DEFAULT);
	cv::threshold(trainImage, trainImage,thresh, 255,cv::THRESH_BINARY);
	//cv::Canny(trainImage,trainImage, thresh, thresh);
	cv::imshow("casos de teste",trainImage);
	cv::waitKey(0);

	int rows = 50;
	int cols = 100;
	
	std::vector<std::vector<cv::Mat> > matrix = gridImage(trainImage,rows,cols);
	std::cout<<"main::grade criada"<<std::endl;
	
	for(int i=0 ; i<distances.size() ; i++){
		for(int j=0 ; j<distances[0].size() ; j++){
			std::cout<<cv::HuMoments(cv::moments(matrix[i][j]),);
		}
	}
	
	std::vector<std::vector<double> > distances = computeDistances(matrix,inputImage);
	std::cout<<"main::distâncias calculadas"<<std::endl;
	
	assert(matrix.size()==50);
	assert(matrix[0].size()==100);
	assert(distances.size()==50);
	assert(distances[0].size()==100);

	std::cout<<"RESPOSTA: "<<knnClassifier(matrix,distances,5000)<<std::endl;
	return 0;

}
