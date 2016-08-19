#pragma once
#include "EagleLib/Detail/Export.hpp"
#include <MetaObject/IMetaObject.hpp>
#include <shared_ptr.hpp>


namespace EagleLib
{
    class EAGLE_EXPORTS Algorithm : public TInterface<IID_Algorithm, mo::IMetaObject>
    {
    public:
        Algorithm();
        virtual ~Algorithm();
        
        virtual void       Process();
        
        double             GetAverageProcessingTime() const;

        virtual void       SetEnabled(bool value);
        bool               IsEnabled() const;

        virtual bool       CheckInputs();

        virtual long long  GetTimestamp();

        void SetSyncInput(mo::InputParameter* input);
    protected:
        virtual void ProcessImpl() = 0;
        void Clock(int line_number);
        bool _enabled;
    private:
        
        struct impl;
        impl* _pimpl;
    };
}
