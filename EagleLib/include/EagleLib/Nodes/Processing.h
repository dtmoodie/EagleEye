#pragma once
#include "Node.h"
namespace EagleLib
{
    namespace Nodes
    {
        class EAGLE_EXPORTS Processing : public Node
        {
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream) = 0;
        };
        class EAGLE_EXPORTS CpuProcessing : public Node
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual cv::Mat doProcess(cv::Mat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream) = 0;
        };

        class EAGLE_EXPORTS GpuProcessing : public Node
        {
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream) = 0;
        };
    }
    
}