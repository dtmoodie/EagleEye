#include "ColorMapperFactory.hpp"
#include "ColorScale.hpp"
#include <boost/filesystem.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>
using namespace EagleLib;


ColorMapperFactory* ColorMapperFactory::Instance()
{
    static ColorMapperFactory inst;
    return &inst;
}

IColorMapper* ColorMapperFactory::Create(std::string color_mapping_scheme_, float alpha, float beta)
{
    auto itr = _registered_functions.find(color_mapping_scheme_);
    if(itr != _registered_functions.end())
    {
        return itr->second(alpha, beta);
    }
    return nullptr;
}

void ColorMapperFactory::Register(std::string colorMappingScheme, std::function<IColorMapper*(float, float)> creation_function_)
{
    _registered_functions[colorMappingScheme] = creation_function_;
}
void ColorMapperFactory::Load(std::string definition_file_xml)
{
    if(boost::filesystem::is_regular_file(definition_file_xml))
    {
        std::ifstream ifs(definition_file_xml);
        cereal::XMLInputArchive ar(ifs);
        std::map<std::string, std::tuple<ColorScale, ColorScale, ColorScale>> color_scales;
        ar(CEREAL_NVP(color_scales));
    }
}

void ColorMapperFactory::Save(std::string definition_file_xml)
{
    std::ofstream ofs(definition_file_xml);
    cereal::XMLOutputArchive ar(ofs);
    std::map<std::string, std::tuple<ColorScale, ColorScale, ColorScale>> color_scales;
    color_scales["example1"] = std::tuple<ColorScale, ColorScale, ColorScale>();
    color_scales["example2"] = std::tuple<ColorScale, ColorScale, ColorScale>();
    ar(CEREAL_NVP(color_scales));
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