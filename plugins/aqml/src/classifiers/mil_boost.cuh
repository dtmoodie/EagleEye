#pragma once
#include <opencv2/core/cuda.hpp>
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"


#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
#include "RuntimeObjectSystem/RuntimeInclude.h"

RUNTIME_MODIFIABLE_INCLUDE;// If this file changes, update files that include this
RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("mil_boost", ".cu");// If the cuda implementation file changes, recompile dependents


namespace aq {
    namespace ML {
        namespace classifiers {
            namespace MIL {
                namespace device
                {
                    // Multiple instance learning boostded decision tree
                    class stump
                    {
                        bool _trained;
                        int _ind;
                    public:
                        __device__ __host__ stump(int index);
                        __device__ __host__ void init();
                        __device__ void update(cv::cuda::PtrStepSzf positive, cv::cuda::PtrStepSzf negative);
                        __host__   void update(cv::Mat_<float> positive, cv::Mat_<float> negative);

                        __device__ bool classify(const cv::cuda::PtrStepSzf& X, int i);
                        __host__   bool classify(const cv::Mat& X, int i);

                        __host__   float classifyF(const cv::Mat& X, int i);
                        __device__ float classifyF(const cv::cuda::PtrStepSzf& X, int i);

                        // This should be a kernel function so each classification can happen separately
                        //__device__ void classify_set(const cv::cuda::PtrStepSzf& X, cv::cuda::PtrStepSzf& result);
                        //__host__   void classify_set(const cv::cuda::GpuMat& X, cv::cuda::GpuMat& result, cv::cuda::Stream& stream);
                        //__host__   void classify_set(const cv::Mat& X, cv::Mat& result);

                        float mu0, mu1, sig0, sig1;
                        float q;
                        int s;
                        float log_n1, log_n0;
                        float e1, e0;
                        float lRate;

                    };
                } // namespace device
            } // namespace MIL
        } // namespace classifiers
    } // namespace ML
} // namespace aq