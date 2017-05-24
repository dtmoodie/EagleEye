#pragma once
#include "src/precompiled.hpp"
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace Nodes
    {
        class ConvertToGrey: public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertToGrey, ::aq::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, grey_image, SyncedMemory());
            MO_END;
        protected:
            bool processImpl();
        };

        class ConvertToHSV: public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertToHSV, ::aq::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, hsv_image, SyncedMemory());
            MO_END;
        protected:
            bool processImpl();
        };
        class ConvertToLab : public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertToLab, ::aq::Nodes::Node)
            INPUT(SyncedMemory, input_image, nullptr)
            OUTPUT(SyncedMemory, lab_image, SyncedMemory())
            MO_END
        protected:
            bool processImpl();
        };

        class ConvertTo: public aq::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertTo, Node)
                INPUT(SyncedMemory, input, nullptr)
                OUTPUT(SyncedMemory, output, SyncedMemory())
                ENUM_PARAM(datatype, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
                PARAM(float, alpha, 1.0)
                PARAM(float, beta, 0.0)
            MO_END
        protected:
            bool processImpl();
        };

        class ConvertColorspace : public Node
        {

        public:
            MO_DERIVE(ConvertColorspace, ::aq::Nodes::Node)
                INPUT(SyncedMemory, input_image, nullptr)
                ENUM_PARAM(conversion_code, cv::COLOR_BGR2HSV)
                OUTPUT(SyncedMemory, output_image, SyncedMemory())
            MO_END
        protected:
            bool processImpl();
        };
        class Magnitude : public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(Magnitude, ::aq::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, output_magnitude, SyncedMemory());
            MO_END;
        protected:
            bool processImpl();
        };
        class SplitChannels: public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(SplitChannels, ::aq::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, channels, SyncedMemory());
            MO_END;
        protected:
            bool processImpl();
        };
        class ConvertDataType: public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertDataType, ::aq::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, output_image, SyncedMemory());
            ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F);
            PARAM(double, alpha, 255.0);
            PARAM(double, beta, 0.0);
            PARAM(bool, continuous, false);
            MO_END;
        protected:
            bool processImpl();
        };
        class MergeChannels: public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(MergeChannels, ::aq::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, merged_image, SyncedMemory());
            MO_END;
        protected:
            bool processImpl();
        };
        class Reshape: public ::aq::Nodes::Node
        {
        public:
            MO_DERIVE(Reshape, ::aq::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, reshaped_image, SyncedMemory());
                PARAM(int, channels, 0);
                PARAM(int, rows, 0);
            MO_END;
        protected:
            bool processImpl();

        };
    }
}
