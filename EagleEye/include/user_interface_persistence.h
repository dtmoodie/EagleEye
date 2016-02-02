#pragma once
#ifndef OPENCV_FOUND
#define OPENCV_FOUND
#endif
#ifndef PARAMTERS_GENERATE_PERSISTENCE
#define PARAMTERS_GENERATE_PERSISTENCE
#endif

#include "EagleLib/ParameteredObject.h"
#include <LokiTypeInfo.h>
#include <map>



class user_interface_persistence: public EagleLib::ParameteredObject
{
public:
    class variable_storage
    {
        std::map<std::string, std::map<std::string, Parameters::Parameter::Ptr>> loaded_parameters;
        variable_storage();
        ~variable_storage();
    public:
        static variable_storage& instance();
        void save_parameters(const std::string& file_name = "user_preferences.yml");
        void load_parameters(const std::string& file_name = "user_preferences.yml");
        void load_parameters(EagleLib::ParameteredObject* This, Loki::TypeInfo type);
        void save_parameters(EagleLib::ParameteredObject* This, Loki::TypeInfo type);

        template<typename T> void load_parameters(T* This)
        {
            load_parameters(This, Loki::TypeInfo(typeid(T)));
        }
        
        template<typename T> void save_parameters(T* This)
        {
            save_parameters(This, Loki::TypeInfo(typeid(T)));
        }
    };
    user_interface_persistence();
    ~user_interface_persistence();
};