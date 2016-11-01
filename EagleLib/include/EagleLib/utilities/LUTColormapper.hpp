#pragma once

#include "IColorMapper.hpp"

namespace EagleLib
{
    class EAGLE_EXPORTS LUTColorMapper: public IColorMapper
    {
        public:
        LUTColorMapper();
        LUTColorMapper(cv::Mat LUT_);
        virtual void Apply(cv::InputArray input, cv::OutputArray output, cv::InputArray mask = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void Rescale(float alpha, float beta);
        virtual cv::Mat_<float> GetMat(float min, float max, int resolution);
        template<typename A> void serialize(A& ar)
        {

        }
        cv::Vec3f Interpolate(float x);
    private:
        cv::Mat_<float> _LUT;
    };
}
