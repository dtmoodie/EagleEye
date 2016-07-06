#include "ColorMapping.hpp"

using namespace EagleLib;

ColorMapperFactory* ColorMapperFactory::Instance()
{
    static ColorMapperFactory inst;
    return &inst;
}

IColorMapper* ColorMapperFactory::Create(std::string color_mapping_scheme_)
{
    auto itr = _registered_functions.find(color_mapping_scheme_);
    if(itr != _registered_functions.end())
    {
        return itr->second();
    }
    return nullptr;
}

void ColorMapperFactory::Register(std::string colorMappingScheme, std::function<IColorMapper*()> creation_function_)
{
    _registered_functions[colorMappingScheme] = creation_function_;
}
std::vector<std::string> ColorMapperFactory::ListSchemes()
{
    std::vector<std::string> output;
    for(auto& scheme : _registered_functions)
    {
        output.push_back(scheme.first);
    }
    return output;
}

IColorMapper::~IColorMapper()
{

}

cv::cuda::GpuMat IColorMapper::Apply(cv::cuda::GpuMat input, cv::InputArray mask, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat output;
    Apply(input, output, mask, stream);
    return output;
}

cv::Mat IColorMapper::Apply(cv::Mat input, cv::InputArray mask)
{
    cv::Mat output;
    Apply(input, output, mask);
    return output;
}

LinearColormapper::LinearColormapper(const ColorScale& red, const ColorScale& green, const ColorScale& blue):
    _red(red),
    _green(green),
    _blue(blue)
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
        ColorMapperFactory::Instance()->Register("Jet Linear", std::bind(&LinearColormapper_registerer::create));
    }
    static IColorMapper* create()
    {
        return new LinearColormapper(ColorScale(50, 255 / 25, true), ColorScale(50 / 3, 255 / 25, true), ColorScale(0, 255 / 25, true));
    }
};
LinearColormapper_registerer g_registerer;