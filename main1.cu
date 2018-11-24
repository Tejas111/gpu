#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <fstream>

#include "pavan.cu"
#include "strtk.hpp"

int main(int argc, char** argv)
{

  	//Taking graph inputs
  	char file_name[] = "testri";
  	char file_name1[] = "testci";
	for(int i=1;i<=1;i++)
	{
		std::ifstream input_file;
		std::cout << "Test File " << i <<std::endl;
		file_name[5]=i+'0';
		std::cout<<file_name;
		input_file.open(file_name);
		if(!input_file)
		{
			std::cout << "Error in reading file" << std::endl;
			exit(4);
		}

		//Reading file
		std::string line;

		std::vector<int> R_vec;
		std::getline(input_file,line);
	   	strtk::parse(line,",",R_vec);
	   	int* R = &R_vec[0];
	   	

	   	std::ifstream input_file1;
		std::cout << "Test File " << i <<std::endl;
		file_name1[5]=i+'0';
		std::cout<<file_name1;
		input_file1.open(file_name1);
		if(!input_file1)
		{
			std::cout << "Error in reading file" << std::endl;
			exit(4);
		}

		std::string line1;

	   	std::vector<int> C_vec;
	   	std::getline(input_file1,line1);
	   	strtk::parse(line1,",",C_vec);
	   	int* C = &C_vec[0];


	   	std::cout<<"Processing" << std::endl;
		//int R[10] = {0,3,5,8,12,16,20,24,27,28};
		//int C[28] = {1,2,3,0,2,0,1,3,0,2,4,5,3,5,6,7,3,4,6,7,4,5,7,8,4,5,6,6};
		int* sigma;
		float* delta;
		int* dr;
		int* dc;
		int* dsigma;
		float* ddelta;
		int* d;
		sigma = (int*) malloc( R_vec.size() * sizeof(int));
		delta = (float*) malloc( R_vec.size() * sizeof(float));
		cudaMalloc((void **)&dr,  R_vec.size() * sizeof(int));
		cudaMalloc((void **)&dc,  C_vec.size() * sizeof(int));
		cudaMalloc((void **)&dsigma,  R_vec.size() * sizeof(int));
		cudaMalloc((void **)&ddelta,  R_vec.size() * sizeof(float));
		cudaMalloc((void **)&d,  R_vec.size() * sizeof(int));
		cudaMemcpy(dr, R, R_vec.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dc, C, C_vec.size() * sizeof(int), cudaMemcpyHostToDevice);
		Betweenness<<<1,R_vec.size()>>>(dr,dc,0,d,dsigma,R_vec.size(),ddelta);
	  	cudaMemcpy(sigma, dsigma, R_vec.size() * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(delta, ddelta, R_vec.size() * sizeof(float), cudaMemcpyDeviceToHost);
	  
	  	std::cout<<"\nResults" << std::endl;
	  	printf("\n");
	  	std::cout<<"\nSigma" << std::endl;
	  	for(int i=0;i<R_vec.size();i++)
			printf("%d ", sigma[i]);
		printf("\n ");
		std::cout<<"\nDelta" << std::endl;
		for(int i=0;i<R_vec.size();i++)
			printf("%d ", delta[i]);
		printf("\n");
		//return 0;
	}
}
