#include "Sink.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;



void CpuSink::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    doProcess(input.GetMat(stream), input.timestamp, input.frame_number, stream);
}

void GpuSink::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    doProcess(input.GetGpuMat(stream), input.timestamp, input.frame_number, stream);
}

