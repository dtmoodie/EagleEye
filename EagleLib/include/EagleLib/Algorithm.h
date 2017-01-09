#pragma once
#include "EagleLib/Detail/Export.hpp"
#include <MetaObject/IMetaObject.hpp>
#include <shared_ptr.hpp>

namespace EagleLib
{
    class EAGLE_EXPORTS Algorithm :
            public TInterface<ctcrc32("EagleLib::Algorithm"), mo::IMetaObject>
    {
    public:
        enum SyncMethod
        {
            SyncEvery = 0, // Require every timestamp to be processed
            SyncNewest     // Process data according to the newest timestamp
        };
        Algorithm();
        virtual ~Algorithm();
        
        virtual bool       Process();
        
        double             GetAverageProcessingTime() const;

        virtual void       SetEnabled(bool value);
        bool               IsEnabled() const;

        virtual long long  GetTimestamp();

        void               SetSyncInput(const std::string& name);
        void               SetSyncMethod(SyncMethod method);
        virtual void       PostSerializeInit();
        std::vector<mo::IParameter*> GetParameters(const std::string& filter = "") const;
    protected:
        virtual bool CheckInputs();
        virtual bool ProcessImpl() = 0;
        void Clock(int line_number);

        void onParameterUpdate(mo::Context* ctx, mo::IParameter* param);
        bool _enabled;
        struct impl;
        impl* _pimpl;
        unsigned int _rmt_hash = 0;
        unsigned int _rmt_cuda_hash = 0;
        std::vector<rcc::shared_ptr<Algorithm>> _algorithm_components;
    private:
        
    };
}
