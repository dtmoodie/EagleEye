#pragma once
#include "EagleLib/Defs.hpp"
#include <EagleLib/rcc/external_includes/cv_core.hpp>

#include <opencv2/core/cuda.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>
#include <map>

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"

RUNTIME_MODIFIABLE_INCLUDE;
RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("ColorMapping",".cpp");

namespace EagleLib
{
    class IColorMapper;
    class EAGLE_EXPORTS ColorMapperFactory
    {
        std::map<std::string, std::function<IColorMapper*()>> _registered_functions;
    public:
        IColorMapper* Create(std::string color_mapping_scheme_);
        void Register(std::string colorMappingScheme, std::function<IColorMapper*()> creation_function_);
        std::vector<std::string> ListSchemes();
        static ColorMapperFactory* Instance();
    };

    class EAGLE_EXPORTS IColorMapper
    {
    public:
        virtual ~IColorMapper();

        // Apply a colormap to an input cpu or gpu image, with the output being a passed in buffer
        virtual void Apply(cv::InputArray input, cv::OutputArray output, cv::InputArray mask = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null()) = 0;

        // Apply a colormap to an input cpu or gpu image with the output being a returned gpu mat
        virtual cv::cuda::GpuMat Apply(cv::cuda::GpuMat input, cv::InputArray mask = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual cv::Mat Apply(cv::Mat input, cv::InputArray mask = cv::noArray());

        // Returns a matrix where
        // Column(0) = x location
        // Column(1) = r location
        // Column(2) = g location
        // Column(3) = b location
        // input min is the min x location
        // input max is the max x location
        // resolution is the number of samples to estimate
        virtual cv::Mat_<float> GetMat(float min, float max, int resolution) = 0;
    };

    struct EAGLE_EXPORTS ColorScale
    {
	    __host__ __device__ ColorScale(double start_ = 0, double slope_ = 1, bool symmetric_ = false);
	    // Defines where this color starts to take effect, between zero and 1000
	    double start;
	    // Defines the slope of increase / decrease for this color between 1 and 255
	    double slope;
	    // Defines if the slope decreases after it peaks
	    bool	symmetric;
	    // Defines if this color starts high then goes low
	    bool	inverted;
	    bool flipped;
	    __host__ __device__ float operator()(float location);
	    __host__ __device__ float GetValue(float location_);
    };

    // Uses the ColorScale object to create linear mappings for red, green, and blue separately
    class EAGLE_EXPORTS LinearColormapper: public IColorMapper
    {
    public:
        LinearColormapper(const ColorScale& red, const ColorScale& green, const ColorScale& blue);
        virtual void Apply(cv::InputArray input, cv::OutputArray output, cv::InputArray mask = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual cv::Mat_<float> GetMat(float min, float max, int resolution);
    private:
        ColorScale _red;
        ColorScale _green;
        ColorScale _blue;
    };

}