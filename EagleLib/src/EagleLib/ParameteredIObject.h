#pragma once
#include "EagleLib/Defs.hpp"
#include "IObject.h"
#include "IObjectInfo.h"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
namespace cv
{
    class FileNode;
}

namespace EagleLib
{
    class EAGLE_EXPORTS ParameteredIObject : public mo::IMetaObject
    {
    public:
        class EAGLE_EXPORTS ParameteredIObjectInfo: public IObjectInfo
        {
            
        };
        ParameteredIObject();
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(const cv::FileNode& configNode);
        virtual void Init(bool firstInit);
        virtual void SerializeAllParams(ISimpleSerializer* pSerializer);

        MO_BEGIN(ParameteredIObject, mo::IMetaObject)
            MO_SIGNAL(void, object_recompiled, ParameteredIObject*)
        MO_END;
    protected:
    };
}
