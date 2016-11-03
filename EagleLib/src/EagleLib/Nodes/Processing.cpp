#include "EagleLib/Nodes/Processing.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;

TS<SyncedMemory> CpuProcessing::process(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    input.GetMatMutable(stream, 0) = doProcess(input.GetMatMutable(stream), input.timestamp, input.frame_number, stream);
    return input;
}
TS<SyncedMemory> GpuProcessing::process(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    input.GetGpuMatMutable(stream,0) = doProcess(input.GetGpuMatMutable(stream, 0), input.timestamp, input.frame_number, stream);
    return input;
}
