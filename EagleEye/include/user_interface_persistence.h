#pragma once

#include <MetaObject/object/IMetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <map>

class UIPersistence
{
public:
    UIPersistence();
    virtual ~UIPersistence(){}
    virtual std::vector<mo::IParam*> GetParameters() = 0;
};

class VariableStorage
{
public:
    static VariableStorage* Instance();
    void SaveUI(const std::string& file_name = "user_preferences.yml");
    void LoadUI(const std::string& file_name = "user_preferences.yml");
    void LoadParams(UIPersistence* This, const std::string& name);
    void SaveParams(UIPersistence* This, const std::string& name);
private:
    VariableStorage();
    ~VariableStorage();
    std::map<std::string, std::map<std::string, mo::IParam*>> loaded_parameters;
};

