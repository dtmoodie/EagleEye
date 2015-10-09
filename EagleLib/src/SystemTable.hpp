#pragma once

#include <LokiTypeInfo.h>
#include <map>
namespace EagleLib
{
	class EventHandler;
}
namespace Freenect
{
	class Freenect;
}
struct SystemTable
{
	SystemTable();

	EagleLib::EventHandler* eventHandler;
	Freenect::Freenect* freenect;
	std::map<Loki::TypeInfo, void*> singletons;

	template<typename T> T* GetSingleton()
	{
		auto itr = singletons.find(Loki::TypeInfo(typeid(T)));
		if (itr != singletons.end())
		{
			return static_cast<T*>(itr->second);
		}
		return nullptr;
	}
	template<typename T> void SetSingleton(T* singleton)
	{
		auto itr = singletons.find(Loki::TypeInfo(typeid(T)));
		if (itr == singletons.end())
		{
			singletons[Loki::TypeInfo(typeid(T))] = static_cast<void*>(singleton);
		}
	}

};