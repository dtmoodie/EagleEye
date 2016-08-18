#pragma once


#include <MetaObject/IMetaObject.hpp>

#include <shared_ptr.hpp>

namespace EagleLib
{
    class EAGLE_EXPORTS Algorithm : public TInterface<IID_Algorithm, mo::IMetaObject>
    {
        std::vector<rcc::shared_ptr<Algorithm>> child_algorithms;
    public:
        virtual std::vector<std::shared_ptr<mo::IParameter>> GetParameters() = 0;
    };
}
