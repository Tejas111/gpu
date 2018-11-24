#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <fstream>

#include "strtk.hpp"

__global__ void Betweenness(int* R,int* C,int s,int* d,int* sigma,int n,float* delta,float* bc)
{

    int idx = threadIdx.x;
	int w,v;
	float ds,dd;
    for(v=idx; v<n; v+=blockDim.x)
    {
      if(v == s)
      {
        d[v] = 0;
        sigma[v] = 1;
      }
      else
      {
        d[v] = INT_MAX;
        sigma[v] = 0;
      }
		delta[v]=0;
    }
    __shared__ int current_depth;
    __shared__ bool done;

    if(idx == 0)
    {  
      done = false;
      current_depth = 0;
    }
    __syncthreads();

    //Calculate the number of shortest paths and the 
    // distance from s (the root) to each vertex
    while(!done)
    {
      __syncthreads();
      done = true;
      __syncthreads();

      for(v=idx; v<n; v+=blockDim.x) //For each vertex...
      {
        if(d[v] == current_depth)
        {
          for(int r=R[v]; r<R[v+1]; r++) //For each neighbor of v
          {
            int w = C[r];
            if(d[w] == INT_MAX)
            {
              d[w] = d[v] + 1;
              done = false;
            }
            if(d[w] == (d[v] + 1))
            {
              atomicAdd(&sigma[w],sigma[v]);
            }
          }
        }
      }
      __syncthreads();
      
      if(idx == 0){
        current_depth++;
      }
    }
	__syncthreads();
	for(v=idx;v<n;v+=blockDim.x)
	{
		ds=0;
		for(int r=R[v]; r<R[v+1]; r++)
		{
			w=C[r];
			if(d[v]==d[w]+1)
				ds=ds+(float)(1+delta[w])*sigma[v]/sigma[w];
		}
		delta[v]=ds;
	}
	__syncthreads();
	for(v=idx;v<n;v+=blockDim.x)
	{
		dd=0;
		for(int r=R[v];r<R[v+1];r++)
		{
			w=C[r];
			if(d[v]==d[w]+1){
				dd=dd+delta[w];
				printf("%f %f\n",dd,delta[w]);}
		}
		bc[v]=dd+delta[v];
	}
}

int main(int argc, char** argv)
{
  	//Taking graph inputs
  	char file_name[] = "testi.txt";
	for(int i=1;i<=3;i++)
	{
		std::ifstream input_file;
		std::cout << "Test File " << i <<std::endl;
		file_name[4]=i+'0';
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
	   	for(int p=0;p<R_vec.size();p++)
	   		std::cout<<R[p]<<" ";

	   	std::vector<int> C_vec;
	   	std::getline(input_file,line);
		//std::cout << line << std::endl;
	   	strtk::parse(line,",",C_vec);
	   	int* C = &C_vec[0];
	   	for(int p=0;p<C_vec.size();p++)
	   		std::cout<<C[p]<<" ";


	   	//Timers
	   	cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

	   	std::cout<<"\nProcessing" << std::endl;
		
		int* sigma;
		float* delta;
		float* bc;
		int* dr;
		int* dc;
		int* dsigma;
		float* ddelta;
		float* dbc;
		int* d;
		sigma = (int*) malloc( R_vec.size() * sizeof(int));
		delta = (float*) malloc( R_vec.size() * sizeof(float));
		bc = (float*) malloc( 10 * sizeof(float));
		cudaMalloc((void **)&dr,  R_vec.size() * sizeof(int));
		cudaMalloc((void **)&dc,  C_vec.size() * sizeof(int));
		cudaMalloc((void **)&dsigma,  R_vec.size() * sizeof(int));
		cudaMalloc((void **)&ddelta,  R_vec.size() * sizeof(float));
		cudaMalloc((void **)&dbc,  10 * sizeof(float));
		cudaMalloc((void **)&d,  R_vec.size() * sizeof(int));
		cudaMemcpy(dr, R, R_vec.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dc, C, C_vec.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaEventRecord(start);
		Betweenness<<<1,R_vec.size()>>>(dr,dc,0,d,dsigma,R_vec.size(),ddelta,dbc);
		cudaEventRecord(stop);
	  	cudaMemcpy(sigma, dsigma, R_vec.size() * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(delta, ddelta, R_vec.size() * sizeof(float), cudaMemcpyDeviceToHost);
	  	cudaMemcpy(bc, dbc, 10 * sizeof(float), cudaMemcpyDeviceToHost);
	  	
	  	cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

	  	std::cout<<"\nResults" << std::endl;
	  	std::cout << "Time:" << milliseconds <<std::endl;

	  	printf("\n");
	  	std::cout<<"\nSigma" << std::endl;
	  	for(int i=0;i<R_vec.size();i++)
			printf("%.2f ", sigma[i]);
		printf("\n ");
		std::cout<<"\nDelta" << std::endl;
		for(int i=0;i<R_vec.size();i++)
			printf("%.2lf ", delta[i]);
		printf("\n");
		std::cout<<"\nBC" << std::endl;
		for(int i=0;i<10;i++)
		printf("%.3f ", bc[i]);
		cudaFree(dr);
		cudaFree(dc);
		cudaFree(dsigma);
		cudaFree(ddelta);
		cudaFree(d);
		//return 0;
	}
}
