#pragma once

#include <boost/signals2.hpp>
#include <memory>
#include "IObject.h"
#include "IRuntimeObjectSystem.h"
#include "LokiTypeInfo.h"
#include <opencv2/core/base.hpp>

namespace EagleLib
{

    class ISignalManager
    {
    public:
        virtual Loki::TypeInfo GetType() = 0;
    };

    template<typename T> class SignalManager: public ISignalManager
    {
        std::map<std::string, T> sig;
    public:
        T* GetSignal(const std::string& sigName)
        {
            return &sig[sigName];
        }
        virtual Loki::TypeInfo GetType()
        {
            return Loki::TypeInfo(typeid(T));
        }
    };
    class ISignalHandler : public TInterface<IID_SignalHandler, IObject>
    {
    public:
        virtual ISignalManager* GetSignalManager(Loki::TypeInfo type) = 0;
        virtual ISignalManager* AddSignalManager(ISignalManager* manager) = 0;

        template<typename T> T* GetSignal(const std::string& name)
        {
            auto manager = GetSignalManager(Loki::TypeInfo(typeid(T)));
            if (manager)
            {
                auto typedManager = dynamic_cast<SignalManager<T>*>(manager);
                if (typedManager)
                {
                    return typedManager->GetSignal(name);
                }
            }
            return nullptr;
        }
        template<typename T> T* GetSignalSafe(const std::string& name)
        {
            auto manager = GetSignalManager(Loki::TypeInfo(typeid(T)));
            if (!manager)
            {
                manager = AddSignalManager(new SignalManager<T>());
            }
            CV_Assert(manager);
            if (manager)
            {
                auto typedManager = dynamic_cast<SignalManager<T>*>(manager);
                if (typedManager)
                {
                    return typedManager->GetSignal(name);
                }
            }
            return nullptr;
        }
    };

    class SignalHandler: public ISignalHandler
    {
    public:
        SignalHandler();
        virtual ISignalManager* GetSignalManager(Loki::TypeInfo type);
        virtual ISignalManager* AddSignalManager(ISignalManager* manager);

    private:
        std::map<Loki::TypeInfo, ISignalManager*> signalManagers;
    };
}
