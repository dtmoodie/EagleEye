#include "EagleLib/utilities/LinearColormapper.hpp"
#include "EagleLib/utilities/ColorMapperFactory.hpp"
using namespace EagleLib;

LinearColorMapper::LinearColorMapper(const ColorScale& red, const ColorScale& green, const ColorScale& blue):
    _red(red),
    _green(green),
    _blue(blue)
{

}
LinearColorMapper::LinearColorMapper()
{
}
void LinearColorMapper::Apply(cv::InputArray input, cv::OutputArray output, cv::InputArray mask, cv::cuda::Stream& stream)
{
    CV_Assert(false && "Not implemented yet");
}

cv::Mat_<float> LinearColorMapper::GetMat(float min, float max, int resolution)
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
void LinearColorMapper::Rescale(float, float)
{
    
}

/*struct LinearColormapper_registerer
{
    LinearColormapper_registerer()
    {
        ColorMapperFactory::Instance()->Register("Jet Linear", std::bind(&LinearColormapper_registerer::create, std::placeholders::_1, std::placeholders::_2));
    }
    static IColorMapper* create(float alpha, float beta)
    {
        return new LinearColormapper(ColorScale(, , true), 
                                     ColorScale(, , true), 
                                     ColorScale(, , true));
    }
};
LinearColormapper_registerer g_registerer;*/