#pragma once

#include <Aquila/types/SyncedImage.hpp>

#include <MetaObject/params/ParamMacros.hpp>

#include <Aquila/nodes/Node.hpp>

#include <aqcore/aqcore_export.hpp>
#include <opencv2/imgproc.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aqcore
{
    class aqcore_EXPORT ConvertToGrey : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ConvertToGrey, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    class aqcore_EXPORT ConvertToHSV : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ConvertToHSV, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            OUTPUT(aq::SyncedImage, output)
        MO_END;

        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    class ConvertToLab : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ConvertToLab, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    class ConvertTo : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ConvertTo, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)

            ENUM_PARAM(datatype, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
            PARAM(float, alpha, 1.0)
            PARAM(float, beta, 0.0)

            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    class aqcore_EXPORT ConvertColorspace : public aq::nodes::Node
    {

      public:
        MO_DERIVE(ConvertColorspace, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)

            ENUM_PARAM(conversion_code, cv::COLOR_BGR2HSV)

            OUTPUT(aq::SyncedImage, output)
        MO_END;

        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    class aqcore_EXPORT Magnitude : public aq::nodes::Node
    {
      public:
        MO_DERIVE(Magnitude, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    // TODO SyncedTensor
    /*class aqcore_EXPORT SplitChannels : public aq::nodes::Node
    {
      public:
        MO_DERIVE(SplitChannels, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };*/

    class aqcore_EXPORT ConvertDataType : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ConvertDataType, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)

            PARAM(double, alpha, 255.0)
            PARAM(double, beta, 0.0)
            PARAM(bool, continuous, false)

            mo::TControlParam<mo::EnumParam*> data_type_param;
            mo::EnumParam data_type =
                mo::EnumParam({"CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S", "CV_32F", "CV_64F"},
                              {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F});
            PUBLIC_ACCESS(data_type);
            PUBLIC_ACCESS(data_type_param);

            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    class aqcore_EXPORT MergeChannels : public aq::nodes::Node
    {
      public:
        MO_DERIVE(MergeChannels, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

    class aqcore_EXPORT Reshape : public aq::nodes::Node
    {
      public:
        MO_DERIVE(Reshape, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)

            PARAM(int, channels, 0)
            PARAM(int, rows, 0)

            OUTPUT(aq::SyncedImage, output)
        MO_END;
        template <class StreamType>
        bool processImpl(StreamType& stream);

      protected:
        bool processImpl() override;
    };

} // namespace aqcore
