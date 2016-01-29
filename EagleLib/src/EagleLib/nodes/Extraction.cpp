#include "Extraction.h"
using namespace EagleLib;
using namespace EagleLib::Nodes;

void CpuExtraction::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    process(input.GetMat(stream), input.timestamp, input.frame_number, stream);
}
void GpuExtraction::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    process(input.GetGpuMat(stream), input.timestamp, input.frame_number, stream);
}