#pragma once
#include "EagleLib/Defs.hpp"
#include "IObject.h"
#include <map>
#include <mutex>
#include <shared_ptr.hpp>

namespace EagleLib
{
    class WindowCallbackHandler;
    class SignalManager;
    // Manages instances of handler for each stream
    /*class EAGLE_EXPORTS WindowCallbackHandlerManager : public TInterface<IID_IOBJECT, IObject>
    {
        std::map<int, rcc::shared_ptr<WindowCallbackHandler>> instances;
        std::mutex mtx;
    public:
        WindowCallbackHandler* instance(SignalManager* mgr);
        WindowCallbackHandlerManager();
        void Init(bool firstInit);
        void Serialize(ISimpleSerializer* pSerializer);
    };*/
}