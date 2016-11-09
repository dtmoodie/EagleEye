#ifndef HAVE_OPENCV
#define HAVE_OPENCV
#endif

#include "user_interface_persistence.h"
#include <MetaObject/Parameters/IParameter.hpp>

void user_interface_persistence::variable_storage::load_parameters(mo::IMetaObject* This, mo::TypeInfo type)
{
    auto& params = loaded_parameters[type.name()];
    for (auto& param : params)
    {
        auto param_ = This->GetParameter(param.first);
        if (param_)
        {
            // Update the variable with data from file
            param_->Update(param.second);
            //This->getParameter(param.first)->Update(param.second.get());
        }
    }
}
void user_interface_persistence::variable_storage::save_parameters(mo::IMetaObject* This, mo::TypeInfo type)
{
    /*auto& params = loaded_parameters[type.name()];
    auto all_params = This->getParameters();
    for(auto& param: all_params)
    {
        params[param->GetName()] = param->DeepCopy();
    }*/
}
void user_interface_persistence::variable_storage::save_parameters(const std::string& file_name)
{
    /*cv::FileStorage fs;
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
            Parameters::Persistence::cv::Serialize(&fs, itr2.second.get());
        }
        fs << "}"; // End parameters
        fs << "}"; // End widgets
        ++index;
    }*/
    
}
void user_interface_persistence::variable_storage::load_parameters(const std::string& file_name)
{
    /*try
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
                auto node = *itr2;
                auto param = Parameters::Persistence::cv::DeSerialize(&node);
                if (param)
                {
                    param_vec[param->GetName()] = std::shared_ptr<mo::IParameter>(param);
                }
            }
        }
        
    }catch(...)
    {
    
    }*/
    
    
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
