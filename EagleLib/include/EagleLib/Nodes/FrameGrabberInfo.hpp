#pragma once
#include "NodeInfo.hpp"


namespace EagleLib
{
    namespace Nodes
    {
        class FrameGrabberInfo;
    }
}


// I tried placing these as functions inside of the MetaObjectInfoImpl specialization, but msvc doesn't like that. :(
template<class T> struct GetLoadableDocumentsHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasLoadableDocuments, ListLoadableDocuments, std::vector<std::string>(*)(void));
    template<class U> 
    static std::vector<std::string> helper(typename std::enable_if<HasLoadableDocuments<U>::value, void>::type* = 0)
    { 
        return U::ListLoadableDocuments(); 
    }
    template<class U> 
    static std::vector<std::string> helper(typename std::enable_if<!HasLoadableDocuments<U>::value, void>::type* = 0)
    { 
        return std::vector<std::string>(); 
    }

    static std::vector<std::string> Get()
    {
        return helper<T>();
    }
};

template<class T> struct GetTimeoutHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasTimeout, LoadTimeout, int(*)(void));
    template<class U> 
    static int helper(typename std::enable_if<HasTimeout<U>::value, void>::type* = 0)
    { 
        return U::LoadTimeout(); 
    }
    template<class U> 
    static int helper(typename std::enable_if<!HasTimeout<U>::value, void>::type* = 0)
    { 
        return 1000;
    }

    static int Get()
    {
        return helper<T>();
    }
};

template<class T> struct GetCanLoadHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasCanLoad, CanLoadDocument, int(*)(const std::string&));
    template<class U> 
    static int helper(const std::string& doc, typename std::enable_if<HasCanLoad<U>::value, void>::type* = 0)
    { 
        return U::CanLoadDocument(doc); 
    }
    template<class U> 
    static int helper(const std::string& doc, typename std::enable_if<!HasCanLoad<U>::value, void>::type* = 0)
    { 
        return 0;
    }

    static int Get(const std::string& doc)
    {
        return helper<T>(doc);
    }
};


namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template<class Type>
    struct MetaObjectInfoImpl<Type, EagleLib::Nodes::FrameGrabberInfo>: public EagleLib::Nodes::FrameGrabberInfo
    {
        int LoadTimeout() const
        {
            return GetTimeoutHelper<Type>::Get();
        }

        std::vector<std::string> ListLoadableDocuments() const
        {
            return GetLoadableDocumentsHelper<Type>::Get();
        }

        int CanLoadDocument(const std::string& document) const
        {
            return GetCanLoadHelper<Type>::Get(document);
        }

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
