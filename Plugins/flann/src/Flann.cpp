#include "Flann.h"


using namespace EagleLib;
void PtCloud_backgroundSubtract_flann::Init(bool firstInit)
{
	if (firstInit)
	{
		addInputParameter<cv::cuda::GpuMat>("Input point cloud");
		updateParameter<float>("Radius", 5.0);
		//updateParameter<int>("K", 1);
	}
}
cv::cuda::GpuMat PtCloud_backgroundSubtract_flann::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	cv::cuda::GpuMat* input = nullptr;
	getParameter<cv::cuda::GpuMat>(0)->Data();
	if (!input)
		input = &img;
	if (input->depth() != CV_32F)
	{
		NODE_LOG(error) << "Input must be floating point";
		return img;
	}
	if (input->channels() != 4)
	{
		if (input->channels() != 3)
		{
			NODE_LOG(error) << "Input needs to either be 3 channel XYZ or 4 channels XYZ with unused 4th channel";
			return img;
		}
		// Image is 3 channel, padding magic time

	}
	// Input is 4 channels, awesome no copying needed
	if (!input->isContinuous())
	{
		auto buffer = inputBuffer.getFront();
	}
	// Input is 4 channel and continuous... woot
	*input = input->reshape(4, input->rows*input->cols);
	// Input is now a tensor row major matrix
	flann::Matrix<float> queries_gpu((float*)input->data, input->rows, input->cols, input->step);
	auto idxBuffer_ = idxBuffer.getFront();
	auto distBuffer_ = distBuffer.getFront();
	
	idxBuffer_->data.create(1, input->rows*input->cols, CV_32S);
	distBuffer_->data.create(1, input->rows*input->cols, CV_32F);

	flann::Matrix<int> d_idx((int*)idxBuffer_->data.data, input->rows*input->cols, 1, sizeof(int));
	flann::Matrix<float> d_dist((float*)distBuffer_->data.data, input->rows*input->cols, 1, sizeof(int));

	flann::SearchParams searchParams;
	searchParams.matrices_in_gpu_ram = true;
	nnIndex->radiusSearch(queries_gpu, d_idx, d_dist, *getParameter<float>(1)->Data(), searchParams);

	updateParameter("Neighbor index", idxBuffer_->data);
	updateParameter("Neighbor dist", distBuffer_->data);

	return img;
}
NODE_DEFAULT_CONSTRUCTOR_IMPL(PtCloud_backgroundSubtract_flann);