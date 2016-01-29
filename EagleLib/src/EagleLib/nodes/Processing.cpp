#include "Processing.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;

void CpuProcessing::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    doProcess(input.GetMatMutable(stream),input.timestamp, input.frame_number, stream);
}
void GpuProcessing::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    doProcess(input.GetGpuMatMutable(stream, 0), input.timestamp, input.frame_number, stream);
}