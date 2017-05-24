#pragma once
#include "precompiled.hpp"
#include <Aquila/rcc/external_includes/cv_superres.hpp>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
namespace aq
{
    namespace Nodes
    {
        class my_frame_source: public cv::superres::FrameSource
        {
            SyncedMemory* current_source;
            cv::cuda::Stream* current_stream;
        public:
            my_frame_source();
            virtual void nextFrame(cv::OutputArray frame);
            virtual void reset();
            virtual void input_frame(SyncedMemory& image, cv::cuda::Stream& stream);
        };

        class super_resolution: public Node
        {
            cv::Ptr<cv::superres::SuperResolution> super_res;
            cv::Ptr<my_frame_source> frame_source;
        public:
            MO_DERIVE(super_resolution, Node)
                INPUT(SyncedMemory, input, nullptr);
                PARAM(int, scale, 2);
                PARAM(int, iterations, 50);
                PARAM(double, tau, 0.0);
                PARAM(double, lambda, 0.0);
                PARAM(double, alpha, 1.0);
                PARAM(int, kernel_size, 5);
                PARAM(int, blur_size, 5);
                PARAM(double, blur_sigma, 1.0);
                PARAM(int, temporal_radius, 1);
                OUTPUT(SyncedMemory, output, SyncedMemory());
            MO_END;
        protected:
            bool processImpl();
        };
    }
}
