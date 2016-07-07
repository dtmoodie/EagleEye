#include "ColorMapping.hpp"
#include "ColorMapperFactory.hpp"

#define HAVE_CEREAL
#include <parameters/Persistence/cereal.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/map.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
using namespace EagleLib;

//CEREAL_REGISTER_TYPE(LinearColormapper);





LinearColormapper::LinearColormapper(const ColorScale& red, const ColorScale& green, const ColorScale& blue):
    _red(red),
    _green(green),
    _blue(blue)
{

}
LinearColormapper::LinearColormapper()
{
}
void LinearColormapper::Apply(cv::InputArray input, cv::OutputArray output, cv::InputArray mask, cv::cuda::Stream& stream)
{
    CV_Assert(false && "Not implemented yet");
}

cv::Mat_<float> LinearColormapper::GetMat(float min, float max, int resolution)
{
    cv::Mat_<float> output(resolution, 4);
    int idx = 0;
    for(float i = min; i < max && idx < resolution; i += (max - min) / float(resolution), ++idx)
    {
        output(idx, 0) = i;
        output(idx, 1) = _red(i);
        output(idx, 2) = _blue(i);
        output(idx, 3) = _green(i);
    }
    return output;
}

struct LinearColormapper_registerer
{
    LinearColormapper_registerer()
    {
        ColorMapperFactory::Instance()->Register("Jet Linear", std::bind(&LinearColormapper_registerer::create, std::placeholders::_1, std::placeholders::_2));
    }
    static IColorMapper* create(float alpha, float beta)
    {
        return new LinearColormapper(ColorScale(50.0/255.0*alpha - beta, 0.4*alpha, true), 
                                     ColorScale(alpha*50 / (255.0*3) - beta, 0.4*alpha, true), 
                                     ColorScale(-beta, 0.4*alpha, true));
    }
};
LinearColormapper_registerer g_registerer;