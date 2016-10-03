#pragma once
#include "src/precompiled.hpp"

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace EagleLib
{
    namespace Nodes
    {
        class ConvertToGrey: public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertToGrey, ::EagleLib::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, grey_image, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };

        class ConvertToHSV: public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertToHSV, ::EagleLib::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, hsv_image, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class ConvertToLab : public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertToLab, ::EagleLib::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, lab_image, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class ConvertColorspace : public ::EagleLib::Nodes::Node
        {
        
        public:
            MO_DERIVE(ConvertColorspace, ::EagleLib::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, output_image, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class Magnitude : public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(Magnitude, ::EagleLib::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, output_magnitude, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class SplitChannels: public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(SplitChannels, ::EagleLib::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, channels, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class ConvertDataType: public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(ConvertDataType, ::EagleLib::Nodes::Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, output_image, SyncedMemory());
            ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F);
            PARAM(double, alpha, 255.0);
            PARAM(double, beta, 0.0);
            PARAM(bool, continuous, false);
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class MergeChannels: public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(MergeChannels, ::EagleLib::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, merged_image, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class Reshape: public ::EagleLib::Nodes::Node
        {
        public:
            MO_DERIVE(Reshape, ::EagleLib::Nodes::Node);
                INPUT(SyncedMemory, input_image, nullptr);
                OUTPUT(SyncedMemory, reshaped_image, SyncedMemory());
                PARAM(int, channels, 0);
                PARAM(int, rows, 0);
            MO_END;
        protected:
            bool ProcessImpl();
        
        };
    }
}
