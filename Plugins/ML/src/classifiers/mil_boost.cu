#include "mil_boost.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include <opencv2/cudaarithm.hpp>

#include "EagleLib/Thrust_interop.hpp"
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
using namespace EagleLib;
using namespace EagleLib::ML;
using namespace EagleLib::ML::classifiers;
using namespace EagleLib::ML::classifiers::MIL;
using namespace EagleLib::ML::classifiers::MIL::device;
#define  sign(s)  ((s > 0 ) ? 1 : ((s<0) ? -1 : 0))

template <typename T>
struct KernelArray
{
    T*  _array;
    int _size;
    KernelArray(thrust::device_vector<T>& dev_vec)
    {
        _array = thrust::raw_pointer_cast(&dev_vec[0]);
        _size = (int)dev_vec.size();
    }
    __device__ T& operator[](int index)
    {
        return _array[index];
    }
};

__device__ __host__ float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}
// This is used for running ALL of the different weak classifiers in training
__global__ void classify_set(
    const cv::cuda::PtrStepSzf X, 
    KernelArray<stump> stumps, 
    cv::cuda::PtrStepSzf result, bool use_sigmoid)
{
    // Two dimensional grid stride loop stuffs
    // X dimension = classifier
    // Y dimension = sample
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < stumps._size; i += blockDim.x * gridDim.x)
    {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < X.rows; ++j)
        {
            if(use_sigmoid)
            {
                result(j,i) = sigmoid(stumps[i].classifyF(X,j));
            }else
            {
                result(j, i) = stumps[i].classifyF(X, j);
            }
        }
    }
}

__global__ void update_set(
    const cv::cuda::PtrStepSzf pos, 
    const cv::cuda::PtrStepSzf neg, 
    KernelArray<stump> stumps)
{
    for(int i =  blockIdx.x * blockDim.x + threadIdx.x; i < stumps._size; i += blockDim.x * gridDim.x)
    {
        stumps[i].update(pos,neg);
    }
}


__device__ __host__ stump::stump(int index):
    _ind(index)
{   
    init();
}

__device__ __host__ void stump::init()
{
    mu0 = 0.0f;
    mu1 = 0.0f;
    sig0 = 1.0f;
    sig1 = 1.0f;
    lRate = 0.85f;
    _trained = false;
}

__device__ void stump::update(cv::cuda::PtrStepSzf positive, cv::cuda::PtrStepSzf negative)
{
    float posmu = 0.0f;
    float negmu = 0.0f;
    if (positive.cols)
    {
        for (int i = 0; i < positive.rows; ++i)
        {
            posmu += positive(i, _ind);
        }
        posmu /= (float)positive.rows;
    }
    if(negative.cols)
    {
        for (int i = 0; i < negative.rows; ++i)
        {
            negmu += negative(i, _ind);
        }
        negmu /= (float)negative.rows;
    }
    if(_trained)
    {
        if (positive.cols > 0)
        {
            mu1 = (lRate * mu1 + (1 - lRate) * posmu);
            float mean_diff_sqr = 0;
            for(int i = 0; i < positive.rows; ++i)
            {
                float diff = positive(i, _ind);
                mean_diff_sqr += diff*diff;
            }
            mean_diff_sqr /= positive.rows;
            sig1 = lRate * sig1 + (1- lRate)* mean_diff_sqr;
        }
        if (negative.cols > 0)
        {
            mu0 = (lRate * mu0 + (1 - lRate) * posmu);
            float mean_diff_sqr = 0;
            for (int i = 0; i < negative.rows; ++i)
            {
                float diff = negative(i, _ind);
                mean_diff_sqr += diff*diff;
            }
            mean_diff_sqr /= negative.rows;
            sig0 = lRate * sig0 + (1 - lRate)* mean_diff_sqr;
        }
        q = (mu1 - mu0) / 2.0f;
        s = sign(q);
        log_n0 = log(1.0f / pow(sig0, 0.5f));
        log_n1 = log(1.0f / pow(sig1, 0.5f));
        e0 = -1.0f / (2.0f * sig0 + 1.17549e-38f /* std::numeric_limits<float>::min() */);
        e1 = -1.0f / (2.0f * sig1 + 1.17549e-38f /* std::numeric_limits<float>::min() */);
    }else
    {
        _trained = true;
        if(positive.cols > 0)
        {
            mu1 = posmu;
            // Calculate mean and variance of this feature
            float sum = 0.0f;
            float sumSq = 0.0f;
            for(int i = 0; i < positive.rows; ++i)
            {
                float val = positive(i, _ind);
                sum += val;
                sumSq += val*val;
            }
            sig1 = (sumSq - sum*sum / (float)positive.rows)/(positive.rows - 1) + 1e-9f;
        }
        if (negative.cols > 0)
        {
            mu0 = posmu;
            // Calculate mean and variance of this feature
            float sum = 0.0f;
            float sumSq = 0.0f;
            for (int i = 0; i < negative.rows; ++i)
            {
                float val = negative(i, _ind);
                sum += val;
                sumSq += val*val;
            }
            sig0 = (sumSq - sum*sum / (float)negative.rows) / (negative.rows - 1) + 1e-9f;
        }
        q = (mu1 - mu0) / 2.0f;
        s = sign(q);
        log_n0 = std::log(1.0f / pow(sig0, 0.5f));
        log_n1 = std::log(1.0f / pow(sig1, 0.5f));
        //_e1 = -1.0f/(2.0f*_sig1+1e-99f);
        //_e0 = -1.0f/(2.0f*_sig0+1e-99f);
        e1 = -1.0f / (2.0f * sig1 + 1.17549e-38f);
        e0 = -1.0f / (2.0f * sig0 + 1.17549e-38f);
    }
}

__host__   void stump::update(cv::Mat_<float> positive, cv::Mat_<float> negative)
{

}

__device__ bool stump::classify(const cv::cuda::PtrStepSzf& X, int i)
{
    float xx = X(i, _ind);
    float log_p0 = (xx - mu0) * (xx - mu0) * e0 + log_n0;
    float log_p1 = (xx - mu1) * (xx - mu1) * e1 + log_n1;
    return log_p1 > log_p0;
}

__host__   bool stump::classify(const cv::Mat& X, int i)
{
    float xx = X.at<float>(i, _ind);
    float log_p0 = (xx - mu0) * (xx - mu0) * e0 + log_n0;
    float log_p1 = (xx - mu1) * (xx - mu1) * e1 + log_n1;
    return log_p1 > log_p0;
}

__host__   float stump::classifyF(const cv::Mat& X, int i)
{
    float xx = X.at<float>(i, _ind);
    float log_p0 = (xx - mu0) * (xx - mu0) * e0 + log_n0;
    float log_p1 = (xx - mu1) * (xx - mu1) * e1 + log_n1;
    return log_p1 - log_p0;
}

__device__ float stump::classifyF(const cv::cuda::PtrStepSzf& X, int i)
{
    float xx = X(i, _ind);
    float log_p0 = (xx - mu0) * (xx - mu0) * e0 + log_n0;
    float log_p1 = (xx - mu1) * (xx - mu1) * e1 + log_n1;
    return log_p1 - log_p0;
}


void mil_tree::update(const cv::Mat& pos, const cv::Mat& neg)
{

}

__global__ void compute_likelihood(
    cv::cuda::PtrStepSzf pos_pred,
    cv::cuda::PtrStepSzf neg_pred,
    cv::cuda::PtrStepSzf H_pos,
    cv::cuda::PtrStepSzf H_neg,
    cv::cuda::PtrStepSzf likelihood)
{
    // For each stump
    for(int w = blockIdx.x * blockDim.x + threadIdx.x; w < pos_pred.cols; w += blockDim.x * gridDim.x)
    {
        // For each example
        float pos_lll = 1.0f;
        for(int j = 0; j < pos_pred.rows; ++j)
        {
            pos_lll *= ( 1.0f - sigmoid(H_pos(0,j) + pos_pred(j,w)));
        }
        pos_lll = (float) -log( 1.0f - pos_lll + 1e-5f ) / (float)pos_pred.rows;

        float neg_lll = 0.0f;
        for(int j = 0; j < neg_pred.rows; ++j)
        {
            neg_lll += -log( 1e-5f + 1.0f - sigmoid( H_neg(0,j) + neg_pred(j,w) ) );
        }
        likelihood(0,w) = pos_lll + neg_lll;
    }
}
__global__ void select_classifier(KernelArray<int> selectors, cv::cuda::PtrStepSzf likelihood, cv::cuda::PtrStepSzi ordering)
{
    // Find the highest ranked (first in likelihood) that doesn't already exist in the selctors set
    __shared__ bool found;
    for(int i = 0; i < likelihood.cols; ++i)
    {
        if(threadIdx.y == 0)
        {
            found = false;
        }
        __syncthreads();
        int classifier_index = ordering(0, i);
        for (int j = threadIdx.y; j < selectors._size; j += blockDim.y)
        {
            if(selectors[j] == classifier_index)
                found = true;
        }
        __syncthreads();
        if(threadIdx.y == 0 && !found)
        {
            // This classifier doesn't already exist in the selection, add it
            for(int j = 0; j < selectors._size; ++j)
            {
                if(selectors[j] == -1)
                    selectors[j] = i;
            }
        }
        if(!found)
            return;
    }
}
__global__ void update_H(KernelArray<int> selectors, cv::cuda::PtrStepSzf H, cv::cuda::PtrStepSzf prediction, int index)
{
    int idx = selectors[index];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H.cols; i += blockDim.x * gridDim.x)
    {
        H(0,i) += prediction(i, idx);
    }
}
// This is used for calulating predictions for only the selected weak classifiers during testing
__global__ void classifyF_kernel(KernelArray<int> selectors, const cv::cuda::PtrStepSzf X, KernelArray<stump> stumps, cv::cuda::PtrStepSzf output)
{
    // Shared memory summation of response for speed
    //__extern__ float smem[];

    // sample_index is grid stride over each one of the samples
    for (int sample_index = blockIdx.y * blockDim.y + threadIdx.y; sample_index < X.rows; sample_index += blockDim.y * gridDim.y)
    {
        // For each selected weak classifier
        for(int classifier_index = blockIdx.x * blockDim.x + threadIdx.x; classifier_index < selectors._size; classifier_index += blockDim.x * gridDim.x)
        {
            output(sample_index, classifier_index) += stumps[selectors[classifier_index]].classifyF(X, sample_index);
        }
    }
}
// per element Y = sigmoid(X) 
__global__ void sigmoid_kernel(cv::cuda::PtrStepSzf X, cv::cuda::PtrStepSzf Y)
{
    for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < X.rows; y += blockDim.y * gridDim.y)
    {
        for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < X.cols; x += blockDim.x * gridDim.x)
        {
            Y(y,x) = sigmoid(X(y,x));
        }
    }
}

void mil_tree::update(const cv::cuda::GpuMat& pos, const cv::cuda::GpuMat& neg, cv::cuda::Stream& stream)
{
    // first update all the classifiers with the new data
    cudaStream_t stream_ = cv::cuda::StreamAccessor::getStream(stream);
    update_set<<<1, 1, 0, stream_>>>(pos, neg, d_stumps);

    cv::cuda::GpuMat pos_results = cv::cuda::createContinuous(pos.rows, num_features, CV_32F);
    cv::cuda::GpuMat neg_results = cv::cuda::createContinuous(neg.rows, num_features, CV_32F);
    cv::cuda::GpuMat H_pos = cv::cuda::createContinuous(1, pos.rows, CV_32F);
    cv::cuda::GpuMat H_neg = cv::cuda::createContinuous(1, neg.rows, CV_32F);
    // compute errors/likl for all weak clfs
    cv::cuda::GpuMat likl = cv::cuda::createContinuous(1, num_features, CV_32F);
    cv::cuda::GpuMat ordering = cv::cuda::createContinuous(1, num_features, CV_32F);
    H_pos.setTo(cv::Scalar(0.0), stream);
    H_neg.setTo(cv::Scalar(0.0), stream);
    d_selectors = thrust::device_vector<int>(num_weak_classifiers, -1);

    for(int i = 0; i < num_weak_classifiers; ++i)
    {
        // run classification on the negative and positive cases
        classify_set<<<1, 1, 0, stream_ >>>(pos, d_stumps, pos_results, false);
        classify_set<<<1, 1, 0, stream_ >>>(neg, d_stumps, neg_results, false);
        compute_likelihood<<<1, 1, 0, stream_ >>>(pos_results, neg_results, H_pos, H_neg, likl);

        // Sort the likelihood and pick the best classifier
        thrust::sequence(thrust::system::cuda::par.on(stream_), GpuMatBeginItr<int>(ordering), GpuMatEndItr<int>(ordering));
        
        thrust::sort_by_key(thrust::system::cuda::par.on(stream_), GpuMatBeginItr<float>(likl), GpuMatEndItr<float>(likl), 
            GpuMatBeginItr<int>(ordering),thrust::greater<float>());// Sort into descending order

        select_classifier<<<1, 1, 0, stream_>>>(d_selectors, likl, ordering);
        
        // Update H_pos / H_neg
        update_H<<<1, 1, 0, stream_>>>(d_selectors, H_pos, pos_results, i);
        update_H<<<1, 1, 0, stream_>>>(d_selectors, H_neg, neg_results, i);
    }
    ++counter;
}

void mil_tree::classifyF(const cv::Mat& X, cv::Mat& output, bool logR)
{

}

void mil_tree::classifyF(
    const cv::cuda::GpuMat& X, 
    cv::cuda::GpuMat& output, 
    bool logR, 
    cv::cuda::Stream& stream)
{
    auto stream_ = cv::cuda::StreamAccessor::getStream(stream);
    //output.create(X.rows, 1, CV_32F);
    buffer.create(X.rows, num_weak_classifiers, CV_32F);
    classifyF_kernel<<<1, 1, 0, stream_>>>(d_selectors, X, d_stumps, buffer);
    // Sum the predictions of all classifiers
    cv::cuda::reduce(buffer, output, 1, CV_REDUCE_SUM, -1, stream);   
    if(logR)
    {
        sigmoid_kernel<<<1, 1, 0, stream_>>>(output, output);
    }
}