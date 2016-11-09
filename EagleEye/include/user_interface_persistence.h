#pragma once

#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Parameters/ParameterMacros.hpp>
#include <map>

class UIPersistence
{
public:
    UIPersistence();
    virtual ~UIPersistence();

    class variable_storage
    {
    public:
        static variable_storage& instance();
        void save_parameters(const std::string& file_name = "user_preferences.yml");
        void load_parameters(const std::string& file_name = "user_preferences.yml");
        void load_parameters(mo::IMetaObject* This, mo::TypeInfo type);
        void save_parameters(mo::IMetaObject* This, mo::TypeInfo type);

        template<typename T> void load_parameters(T* This)
        {
            load_parameters(This, mo::TypeInfo(typeid(T)));
        }

        template<typename T> void save_parameters(T* This)
        {
            save_parameters(This, mo::TypeInfo(typeid(T)));
        }
    private:
        std::map<std::string, std::map<std::string, mo::IParameter*>> loaded_parameters;
        variable_storage();
        ~variable_storage();
    };
};
