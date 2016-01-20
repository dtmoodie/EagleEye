#pragma once

#include "ParameteredObject.h"

#include <opencv2/core/cuda.hpp>
#include <EagleLib/shared_ptr.hpp>
namespace EagleLib
{
    class EAGLE_EXPORTS Algorithm : public TInterface<IID_Algorithm, EagleLib::ParameteredObject>
    {
        std::vector<shared_ptr<Algorithm>> child_algorithms;
    public:
        virtual std::vector<Parameters::Parameter::Ptr> GetParameters() = 0;
    };
}
