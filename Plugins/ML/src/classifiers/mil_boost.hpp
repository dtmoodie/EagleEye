#pragma once

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include "cuda.h"
#include "cuda_runtime.h"

//#include "Aquila/rcc/external_includes/parameters.hpp"
#include "Aquila/rcc/external_includes/cv_core.hpp"
#include "Aquila/rcc/external_includes/cv_cudev.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>

//#include "Aquila/core/Algorithm.hpp"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

RUNTIME_MODIFIABLE_INCLUDE; // If this file changes, update files that include this
RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("mil_boost",
                                       ".cu"); // If the cuda implementation file changes, recompile dependents

namespace aq
{
    namespace ML
    {
        namespace classifiers
        {
            namespace MIL
            {
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
                        __host__ void update(cv::Mat_<float> positive, cv::Mat_<float> negative);

                        __device__ bool classify(const cv::cuda::PtrStepSzf& X, int i);
                        __host__ bool classify(const cv::Mat& X, int i);

                        __host__ float classifyF(const cv::Mat& X, int i);
                        __device__ float classifyF(const cv::cuda::PtrStepSzf& X, int i);

                        // This should be a kernel function so each classification can happen separately
                        //__device__ void classify_set(const cv::cuda::PtrStepSzf& X, cv::cuda::PtrStepSzf& result);
                        //__host__   void classify_set(const cv::cuda::GpuMat& X, cv::cuda::GpuMat& result,
                        //cv::cuda::Stream& stream);
                        //__host__   void classify_set(const cv::Mat& X, cv::Mat& result);

                        float mu0, mu1, sig0, sig1;
                        float q;
                        int s;
                        float log_n1, log_n0;
                        float e1, e0;
                        float lRate;
                    };
                } // namespace device

                class mil_tree // : public Aquila::Algorithm
                {
                    thrust::device_vector<device::stump> d_stumps;
                    thrust::host_vector<device::stump> h_stumps;
                    thrust::device_vector<int> d_selectors;
                    thrust::host_vector<int> h_selectors;
                    cv::cuda::GpuMat buffer;
                    unsigned int num_samples;
                    unsigned int counter;
                    // This is the total number of weak classifiers
                    int num_features;
                    // The top best performing weak classifiers are used for final classification.  This number controls
                    // how many from the top performing classifiers will be used
                    int num_weak_classifiers;

                  public:
                    void nodeInit(bool firstInit);

                    // virtual std::vector<Parameters::Parameter::Ptr> GetParameters();

                    void update(const cv::Mat& pos, const cv::Mat& neg);
                    void update(const cv::cuda::GpuMat& pos,
                                const cv::cuda::GpuMat& neg,
                                cv::cuda::Stream& stream = cv::cuda::Stream::Null());

                    void classifyF(const cv::Mat& X, cv::Mat& output, bool logR = false);
                    void classifyF(const cv::cuda::GpuMat& X,
                                   cv::cuda::GpuMat& output,
                                   bool logR = false,
                                   cv::cuda::Stream& stream = cv::cuda::Stream::Null());
                };
            } // namespace MIL
        }     // namespace classifiers
    }         // namespace ML
} // namespace aq
