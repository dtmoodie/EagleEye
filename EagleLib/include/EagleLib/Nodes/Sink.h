#pragma once

#include "Node.h"

namespace EagleLib
{
    namespace Nodes
    {
        class EAGLE_EXPORTS Sink: public Node
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream) = 0;
        };
        class EAGLE_EXPORTS CpuSink: public Sink
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual void doProcess(const cv::Mat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream) = 0;
        };
        class EAGLE_EXPORTS GpuSink : public Sink
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual void doProcess(const cv::cuda::GpuMat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream) = 0;
        };
    } // namespace Nodes    
} // namespace EagleLib