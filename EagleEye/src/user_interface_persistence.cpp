#ifndef OPENCV_FOUND
#define OPENCV_FOUND
#endif
#include "Persistence/OpenCV.hpp"
#include "user_interface_persistence.h"


void user_interface_persistence::variable_storage::load_parameters(EagleLib::ParameteredObject* This, Loki::TypeInfo type)
{
    auto& params = loaded_parameters[type.name()];
    for (auto& param : params)
    {
        if (This->exists(param->GetName()))
        {
            // Update the variable with data from file
            This->getParameter(param->GetName())->Update(param);
        }
    }
}
void user_interface_persistence::variable_storage::save_parameters(EagleLib::ParameteredObject* This, Loki::TypeInfo type)
{
    auto& params = loaded_parameters[type.name()];
    for(auto& param:This->parameters)
    {
        params.push_back(param->DeepCopy());
    }
}
void user_interface_persistence::variable_storage::save_parameters(const std::string& file_name)
{
    cv::FileStorage fs;
    fs.open(file_name, cv::FileStorage::WRITE);
    int index = 0;
    fs << "Count" << (int)loaded_parameters.size();
    for(auto itr: loaded_parameters)
    {
        fs << "Widget-" + boost::lexical_cast<std::string>(index) << "{";
        fs << "WidgetType" << itr.first;
        fs << "Parameters" << "{";
        for(auto itr2: itr.second)
        {
            Parameters::Persistence::cv::Serialize(&fs, itr2.get());            
        }
        fs << "}"; // End parameters
        fs << "}"; // End widgets
    }
    
}
void user_interface_persistence::variable_storage::load_parameters(const std::string& file_name)
{
    try
    {
        cv::FileStorage fs;
        fs.open(file_name, cv::FileStorage::READ);
        int count = (int)fs["Count"];
        for(int i = 0; i < count; ++i)
        {
            auto widget_node = fs["Widget-" +boost::lexical_cast<std::string>(i)];
            auto type = (std::string)(widget_node)["WidgetType"];
            auto parameter_node = widget_node["Parameters"];
            auto& param_vec = loaded_parameters[type];
            for (auto itr2 = parameter_node.begin(); itr2 != parameter_node.end(); ++itr2)
            {
                auto param = Parameters::Persistence::cv::DeSerialize(&(*itr2));
                if (param)
                {
                    param_vec.push_back(Parameters::Parameter::Ptr(param));
                }
            }
        }
        
    }catch(...)
    {
    
    }
    
    
}
user_interface_persistence::variable_storage::variable_storage()
{
    load_parameters();
}

user_interface_persistence::variable_storage::~variable_storage()
{
    save_parameters();
}

user_interface_persistence::variable_storage& user_interface_persistence::variable_storage::instance()
{
    static variable_storage inst;
    return inst;
}

user_interface_persistence::user_interface_persistence()
{
    //variable_storage::instance().load_parameters(this);
}

user_interface_persistence::~user_interface_persistence()
{
    //variable_storage::instance().save_parameters(this);
}