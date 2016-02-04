#pragma once
#include <EagleLib/rcc/external_includes/cv_superres.hpp>
#include <EagleLib/nodes/Node.h>

namespace EagleLib
{
    namespace Nodes
    {
        class my_frame_source: public cv::superres::FrameSource
        {
            TS<SyncedMemory>* current_source;
            cv::cuda::Stream* current_stream;
        public:
            my_frame_source();
            virtual void nextFrame(cv::OutputArray frame);
            virtual void reset();
            virtual void input_frame(TS<SyncedMemory>& image, cv::cuda::Stream& stream);
        };

        class super_resolution: public Node
        {
            cv::Ptr<cv::superres::SuperResolution> super_res;
            cv::Ptr<my_frame_source> frame_source;
            

        public:
            super_resolution();
            virtual void Init(bool firstInit);
            virtual void doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
        };
    }
}