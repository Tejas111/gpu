#include<stdio.h>
#include<cstdlib>
#include<iostream>
#include<cuda.h>

__global__ void Betweenness(int* R,int* C,int s,int* d,int* sigma,int n,float* delta)
{  //printf("dfmk");

    int idx = threadIdx.x;
	int w,v;
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
		float ds = 0;
		for(int r=R[v]; r<R[v+1]; r++)
		{
			w=C[r];
			if(d[w]+1==d[v]){
				ds=ds+(float)sigma[v]*(1+delta[w])/sigma[w];
				printf("%d ",(float)sigma[v]*(1+delta[w])/sigma[w]);}
		}
		delta[v]=ds;
	}
	__syncthreads();
}
