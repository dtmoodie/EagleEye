#include "Segmentation_impl.h"
#include <cuda.h>


using namespace cv;
using namespace cv::cuda;


// Calculates the closest center for each point and updates labels
template<typename T, typename labelType> __global__ void calculateClosestCenter(T* samples, int k, labelType* labels, T* centers, int N, int D, T* weights = 0)
{
    int tid = threadIdx.x * blockDim.x;
    if(tid >= N)
        return;
    T distance;
    T minDistance = 100000000; // Need to set to numeric limits of T
    labelType minCenter = 0;

    for(int c = 0; c < k; ++c)
    {
        distance = 0;
        for(int i = 0; i < D; ++i)
        {
            if(weights)
                distance += weights[i] * sqrt(samples[tid*D + i] - centers[c*D + i]);
            else
                distance += samples[tid*D + i] - centers[c*D + i];
        }
        if(distance < minDistance)
        {
            minDistance = distance;
            minCenter = c;
        }
    }
    labels[tid] = minCenter;
}

template<typename T, typename labelType> __global__ void sumCentroid(T* samples, int k, labelType* labels, T* centerBuf, int N, int D, unsigned int* countBuf)
{
    int tid = threadIdx.x * blockDim.x;
    if(tid >= N)
        return;
    labelType c = labels[tid];
    atomicAdd(countBuf + c, 1);
    for(int i = 0; i < D; ++i)
    {
        atomicAdd((centerBuf + c*D + i), samples[tid*D + i]);
    }
}
template<typename T> __global__ void normalizeCentroids(T* centerBuf, int K, int D, unsigned int* countBuf)
{
    int tid = threadIdx.x * blockDim.x;
    if(tid >= K)
        return;
    for(int i = 0; i < D; ++i)
    {
        centerBuf[tid*D + i] /= T(countBuf);
    }
}
template<typename T> __global__ void calculateCentroidShift(T* centerBuf, T* sharedBuf, int K, int D, T* totalShift)
{
    int tid = threadIdx.x * blockDim.x;
    T dist = 0;
    for(int i = 0; i < D; ++i)
    {
        dist += sqr(centerBuf[tid*D + i] + sharedBuf[tid*D + i]);
    }
    atomicAdd(totalShift, dist);
    for(int i = 0; i < D; ++i)
    {
        centerBuf[tid*D + i] = sharedBuf[tid*D + i];
    }
}

template<typename T, typename labelType> __global__ void calculateKMeans(T* samples, int K, labelType* labels, T* centers, int N, int D, int maxIters, T* weights)
{
    extern __shared__ T sharedBuf[];

    for(int i = 0; i < K*D; ++i)
    {
        sharedBuf[i] = 0;
    }

    unsigned int* countBuf = (unsigned int*)&sharedBuf[K*D];

    __syncthreads();

    int blocks = N / 1024;
    bool converged = false;
    int iters = 0;
    T centroidShift;
    while(!converged && iters < maxIters)
    {
        calculateClosestCenter<T,labelType><<<blocks,1024>>>(samples, K, labels, centers, N, D, weights);

        for(int i = 0; i < K; ++i)
        {
            countBuf[i] = 0;
        }

        sumCentroid<T, labelType><<<blocks, 1024>>>(samples, K, labels, sharedBuf, N, D, countBuf);

        normalizeCentroids<T><<<blocks, 1024>>>(sharedBuf, K, D, countBuf);

        // Calculate the distance between the old and new centroids
        centroidShift = 0;

        calculateCentroidShift<T><<<1, K>>>(centers, sharedBuf,K, D, &centroidShift);




    }



}





double CV_EXPORTS kmeans(GpuMat samples, int K, GpuMat& labels,
            TermCriteria termCrit, int attempts, int flags,
            GpuMat& centers, Stream stream = Stream::Null(), GpuMat weights = cv::cuda::GpuMat())
{
    int dims = samples.cols;
    int N = samples.rows;
    int type = samples.type();

    centers.create(K, dims, type);
    labels.create(N,1, CV_8U);

    // Random initialization of centers

    // For each sample, calculate the closest center

    // For each center, estimate a new centroid (requires a shared buffer of size K * dims)

    // Calculate movement of the centroid


    return 0.0;
}
