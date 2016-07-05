#pragma once
#include "EagleLib/rcc/shared_ptr.hpp"
#include <parameters/LokiTypeInfo.h>
#include <map>
#include <memory>
struct ISingleton
{
    virtual ~ISingleton(){}
};

template<typename T> struct Singleton: public ISingleton
{
    Singleton(T* ptr_): ptr(ptr_){}
    ~Singleton()
    {
        delete ptr;
    }
    T* ptr;
    operator T*() const {return ptr;}
};
template<typename T> struct IObjectSingleton: public ISingleton
{
    rcc::shared_ptr<T> ptr;
    IObjectSingleton(T* ptr_): ptr(ptr_){}
    IObjectSingleton(const rcc::shared_ptr<T>& ptr_): ptr(ptr_){}
    operator T*() const {return ptr.get(); }
};
struct SystemTable
{
    SystemTable();
    // These are per stream singletons
    template<typename T> T* GetSingleton()
    {
        auto g_itr = g_singletons.find(Loki::TypeInfo(typeid(T)));
        if(g_itr != g_singletons.end())
        {
            return static_cast<Singleton<T>*>(g_itr->second.get())->ptr;
        }
        return nullptr;
    }
    template<typename T> T* SetSingleton(T* singleton, int stream_id = -1)
    {
        
        g_singletons[Loki::TypeInfo(typeid(T))] = std::unique_ptr<ISingleton>(new Singleton<T>(singleton));
        return singleton;
    }
    void DeleteSingleton(Loki::TypeInfo type);
    template<typename T> void DeleteSingleton()
    {
        DeleteSingleton(Loki::TypeInfo(typeid(T)));
   }
private:
    std::map<Loki::TypeInfo, std::unique_ptr<ISingleton>> g_singletons;
};