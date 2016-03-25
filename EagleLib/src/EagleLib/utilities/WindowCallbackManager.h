#pragma once
#include "EagleLib/Defs.hpp"
#include "IObject.h"
#include <map>
#include <mutex>
#include "EagleLib/rcc/shared_ptr.hpp"

namespace EagleLib
{
    class WindowCallbackHandler;
    // Manages instances of handler for each stream
    class EAGLE_EXPORTS WindowCallbackHandlerManager : public TInterface<IID_IOBJECT, IObject>
    {
        std::map<size_t, rcc::shared_ptr<WindowCallbackHandler>> instances;
        std::mutex mtx;
    public:
        WindowCallbackHandler* instance(size_t stream_id = 0);
        WindowCallbackHandlerManager();
        void Init(bool firstInit);
        void Serialize(ISimpleSerializer* pSerializer);
    };
}