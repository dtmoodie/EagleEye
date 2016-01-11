#include "Extraction.h"
using namespace EagleLib;

void CpuExtraction::process(SyncedMemory& input, cv::cuda::Stream& stream)
{
	processImpl(input.GetMat(stream), stream);
}
void GpuExtraction::process(SyncedMemory& input, cv::cuda::Stream& stream)
{
	processImpl(input.GetGpuMat(stream), stream);
}