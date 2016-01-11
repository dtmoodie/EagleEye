#include "Processing.h"

using namespace EagleLib;

void CpuProcessing::process(SyncedMemory& input, cv::cuda::Stream& stream)
{
	processImpl(input.GetMatMutable(stream, 0), stream);
}
void GpuProcessing::process(SyncedMemory& input, cv::cuda::Stream& stream)
{
	processImpl(input.GetGpuMatMutable(stream, 0), stream);
}