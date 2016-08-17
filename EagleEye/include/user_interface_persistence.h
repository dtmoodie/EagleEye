#pragma once
#ifndef HAVE_OPENCV
#define HAVE_OPENCV
#endif
#ifndef PARAMTERS_GENERATE_PERSISTENCE
#define PARAMTERS_GENERATE_PERSISTENCE
#endif

#include "EagleLib/ParameteredIObject.h"
#include "parameters/ParameteredObjectImpl.hpp"
#include <parameters/LokiTypeInfo.h>
#include <map>



class user_interface_persistence: public Parameters::ParameteredObject
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
        void load_parameters(Parameters::ParameteredObject* This, mo::TypeInfo type);
        void save_parameters(Parameters::ParameteredObject* This, mo::TypeInfo type);

        template<typename T> void load_parameters(T* This)
        {
            load_parameters(This, mo::TypeInfo(typeid(T)));
        }
        
        template<typename T> void save_parameters(T* This)
        {
            save_parameters(This, mo::TypeInfo(typeid(T)));
        }
    };
    user_interface_persistence();
    ~user_interface_persistence();
};