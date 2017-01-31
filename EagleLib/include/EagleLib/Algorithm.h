#pragma once
#include "EagleLib/Detail/Export.hpp"
#include <MetaObject/MetaObject.hpp>
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
        //std::vector<mo::IParameter*> GetParameters(const std::string& filter = "") const;
        std::vector<mo::IParameter*> GetComponentParameters(const std::string& filter = "") const;
        std::vector<mo::IParameter*> GetAllParameters(const std::string& filter = "") const;
        mo::IParameter* GetOutput(const std::string& name) const;
        template<class T> 
        mo::ITypedParameter<T>* GetOutput(const std::string& name) const
        {
            return mo::IMetaObject::GetOutput<T>(name);
        }
        void  SetContext(mo::Context* ctx, bool overwrite = false);
        const std::vector<rcc::weak_ptr<Algorithm>>& GetComponents() const
        {
            return _algorithm_components;
        }
        void  Serialize(ISimpleSerializer *pSerializer);
        void AddComponent(rcc::weak_ptr<Algorithm> component);
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
        std::vector<rcc::weak_ptr<Algorithm>> _algorithm_components;

    private:
        
    };
}
