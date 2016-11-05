#include "EagleLib/Nodes/Sink.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;



TS<SyncedMemory> CpuSink::process(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    doProcess(input.GetMat(stream), input.timestamp, input.frame_number, stream);
    return input;
}

TS<SyncedMemory> GpuSink::process(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    doProcess(input.GetGpuMat(stream), input.timestamp, input.frame_number, stream);
    return input;
}

