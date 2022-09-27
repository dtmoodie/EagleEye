#pragma once
#include <ct/types/opencv.hpp>

#include <aqcore_export.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <aqcore/OpenCVCudaNode.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aqcore
{

    class MinMax : public OpenCVCudaNode
    {
      public:
        MO_DERIVE(MinMax, Node)
            INPUT(aq::SyncedImage, input)

            OUTPUT(double, min_value, 0.0)
            OUTPUT(double, max_value, 0.0)
        MO_END;

      protected:
        bool processImpl(aq::CVStream& stream);
        bool processImpl(mo::IAsyncStream& stream);
        using OpenCVCudaNode::processImpl;
        // bool processImpl(mo::IDeviceStream& stream);
    };

    class Threshold : public OpenCVCudaNode
    {
      public:
        MO_DERIVE(Threshold, OpenCVCudaNode)
            INPUT(aq::SyncedImage, input)
            OPTIONAL_INPUT(double, input_max)
            OPTIONAL_INPUT(double, input_min)

            PARAM(double, replace_value, 255.0)
            PARAM(double, max, 1.0)
            PARAM(double, min, 0.0)
            PARAM(bool, two_sided, false)
            PARAM(bool, truncate, false)
            PARAM(bool, inverse, false)
            PARAM(bool, source_value, true)
            PARAM(double, input_percent, 0.9)

            OUTPUT(aq::SyncedImage, output)

        MO_END;

      protected:
        bool processImpl(aq::CVStream& stream);
        bool processImpl(mo::IAsyncStream& stream);
        using OpenCVCudaNode::processImpl;
        // bool processImpl(mo::IDeviceStream& stream);
    };

    class NonMaxSuppression : public OpenCVCudaNode
    {
      public:
        MO_DERIVE(NonMaxSuppression, OpenCVCudaNode)
            PARAM(int, size, 5)
            INPUT(aq::SyncedImage, input)
            INPUT(aq::SyncedImage, mask)
            OUTPUT(aq::SyncedImage, output)
        MO_END;
    };

} // namespace aqcore
