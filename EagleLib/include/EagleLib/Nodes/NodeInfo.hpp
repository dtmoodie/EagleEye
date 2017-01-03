#pragma once
#include <EagleLib/Nodes/NodeInfo.hpp>
#include <EagleLib/Detail/Export.hpp>

#include <MetaObject/Detail/HelperMacros.hpp>
#include <MetaObject/MetaObjectInfo.hpp>

#include <vector>
#include <string>

namespace mo
{
    class IVariableManager;
}
namespace EagleLib
{
    namespace Nodes
    {
        struct EAGLE_EXPORTS NodeInfo : virtual public mo::IMetaObjectInfo
        {
            std::string Print() const;
            // Get the organizational hierarchy of this node, ie Image -> Processing -> ConvertToGrey
            virtual std::vector<std::string> GetNodeCategory() const = 0;

            // List of nodes that need to be in the direct parental tree of this node, in required order
            virtual std::vector<std::vector<std::string>> GetParentalDependencies() const = 0;

            // List of nodes that must exist in this data stream, but do not need to be in the direct parental tree of this node
            virtual std::vector<std::vector<std::string>> GetNonParentalDependencies() const = 0;

            // Given the variable manager for a datastream, look for missing dependent variables and return a list of candidate nodes that provide those variables
            virtual std::vector<std::string> CheckDependentVariables(mo::IVariableManager* var_manager_) const = 0;
        };
    }
}

template<class T> struct GetNodeCategoryHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasNodeCategory, GetNodeCategory, std::vector<std::string>(*)(void));
    template<class U> 
    static std::vector<std::string> helper(typename std::enable_if<HasNodeCategory<U>::value, void>::type* = 0)
    { 
        return U::GetNodeCategory(); 
    }
    template<class U> 
    static std::vector<std::string> helper(typename std::enable_if<!HasNodeCategory<U>::value, void>::type* = 0)
    { 
        return std::vector<std::string>(1, std::string(U::GetTypeNameStatic()));
    }

    static std::vector<std::string> Get()
    {
        return helper<T>();
    }
};

template<class T> struct GetParentDepsHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasParentDeps, GetParentalDependencies, std::vector<std::vector<std::string>>(*)(void));
    template<class U> 
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<HasParentDeps<U>::value, void>::type* = 0)
    { 
        return U::GetParentalDependencies(); 
    }
    template<class U> 
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<!HasParentDeps<U>::value, void>::type* = 0)
    { 
        return std::vector<std::vector<std::string>>(); 
    }

    static std::vector<std::vector<std::string>> Get()
    {
        return helper<T>();
    }
};

template<class T> struct GetNonParentDepsHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasNonParentDeps, GetNonParentalDependencies, std::vector<std::vector<std::string>>(*)(void));
    template<class U> 
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<HasNonParentDeps<U>::value, void>::type* = 0)
    { 
        return U::GetParentalDependencies(); 
    }
    template<class U> 
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<!HasNonParentDeps<U>::value, void>::type* = 0)
    { 
        return std::vector<std::vector<std::string>>(); 
    }

    static std::vector<std::vector<std::string>> Get()
    {
        return helper<T>();
    }
};


template<class T> struct GetDepVarHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasDepVar, CheckDependentVariables, std::vector<std::string>(*)(mo::IVariableManager*));
    template<class U> 
    static std::vector<std::vector<std::string>> helper(mo::IVariableManager* mgr, typename std::enable_if<HasDepVar<U>::value, void>::type* = 0)
    { 
        return U::CheckDependentVariables(mgr); 
    }
    template<class U> 
    static std::vector<std::string> helper(mo::IVariableManager* mgr, typename std::enable_if<!HasDepVar<U>::value, void>::type* = 0)
    { 
        return std::vector<std::string>();
    }

    static std::vector<std::string> Get(mo::IVariableManager* mgr)
    {
        return helper<T>(mgr);
    }
};

namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template<class Type>
    struct MetaObjectInfoImpl<Type, EagleLib::Nodes::NodeInfo>: public EagleLib::Nodes::NodeInfo
    {
        std::vector<std::string> GetNodeCategory() const
        {
            return GetNodeCategoryHelper<Type>::Get();
        }

        // List of nodes that need to be in the direct parental tree of this node, in required order
        std::vector<std::vector<std::string>> GetParentalDependencies() const
        {
            return GetParentDepsHelper<Type>::Get();
        }

        // List of nodes that must exist in this data stream, but do not need to be in the direct parental tree of this node
        std::vector<std::vector<std::string>> GetNonParentalDependencies() const
        {
            return GetNonParentDepsHelper<Type>::Get();
        }

        // Given the variable manager for a datastream, look for missing dependent variables and return a list of candidate nodes that provide those variables
        std::vector<std::string> CheckDependentVariables(mo::IVariableManager* var_manager_) const
        {
            return GetDepVarHelper<Type>::Get(var_manager_);
        }
    };
}
