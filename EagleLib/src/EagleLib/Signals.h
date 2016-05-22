#pragma once
#include <boost/preprocessor.hpp>
#include "EagleLib/Defs.hpp"
#include "signals/signal_manager.h"
#include "signals/signal.h"
#include <EagleLib/rcc/SystemTable.hpp>
#include <ObjectInterfacePerModule.h>
namespace EagleLib
{
    // Basically reimplements some of the stuff from Signals::signal_manager, but with
    // access restrictions and stream segregation
    class EAGLE_EXPORTS SignalManager: public Signals::signal_manager
    {
    public:

		static	SignalManager* get_instance();
    };


}
