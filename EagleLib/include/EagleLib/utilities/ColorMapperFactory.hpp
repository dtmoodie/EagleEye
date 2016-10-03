#pragma once
#include "EagleLib/Detail/Export.hpp"
#include <functional>
#include <map>
#include <vector>
#include <string>

namespace EagleLib
{
    class IColorMapper;
    class EAGLE_EXPORTS ColorMapperFactory
    {
        std::map<std::string, std::function<IColorMapper*(float, float)>> _registered_functions;
    public:
        // Give the name of a scheme as well as an alpha and beta value
        // each colormap is defined in the range 0,1.  Alpha will first be used to scale the range and then beta offsets
        IColorMapper* Create(std::string color_mapping_scheme_, float alpha, float beta);
        void Register(std::string colorMappingScheme, std::function<IColorMapper*(float, float)> creation_function_);
        void Load(std::string definition_file_xml);
        void Save(std::string definition_file_xml);
        std::vector<std::string> ListSchemes();
        static ColorMapperFactory* Instance();
    };
}