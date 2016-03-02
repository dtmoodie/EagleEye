#pragma once
#include <EagleLib/Defs.hpp>
class EAGLE_EXPORTS IObjectNotifiable
{
public:
    IObjectNotifiable();
    virtual ~IObjectNotifiable();    
    virtual void notify_swap();
protected:
    bool object_swapped;
};