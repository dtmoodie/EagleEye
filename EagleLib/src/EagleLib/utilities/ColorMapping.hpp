#pragma once
#include "EagleLib/Defs.hpp"
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include "IColorMapper.hpp"
#include "ColorScale.hpp"
#include <parameters/Persistence/cereal.hpp>
#include <opencv2/core/cuda.hpp>


#include <functional>
#include <map>

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"

RUNTIME_MODIFIABLE_INCLUDE;
RUNTIME_COMPILER_SOURCEDEPENDENCY;


namespace EagleLib
{
    // Uses the ColorScale object to create linear mappings for red, green, and blue separately
    class EAGLE_EXPORTS LinearColormapper: public IColorMapper
    {
    public:
        LinearColormapper();
        LinearColormapper(const ColorScale& red, const ColorScale& green, const ColorScale& blue);
        virtual void Apply(cv::InputArray input, cv::OutputArray output, cv::InputArray mask = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual cv::Mat_<float> GetMat(float min, float max, int resolution);
        template<typename A> void serialize(A& ar)
        {
            ar(CEREAL_NVP(_red));
            ar(CEREAL_NVP(_green));
            ar(CEREAL_NVP(_blue));
        }
    private:
        ColorScale _red;
        ColorScale _green;
        ColorScale _blue;
    };
}