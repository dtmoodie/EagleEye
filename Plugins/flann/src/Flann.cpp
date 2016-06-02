#include "Flann.h"
#include "thrust/transform.h"
#include "thrust/transform_reduce.h"
#include "thrust/count.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_imgproc.hpp>
#include <parameters/ParameteredObjectImpl.hpp>
#include "EagleLib/rcc/ObjectManager.h"
#include "RuntimeSourceDependency.h"
SETUP_PROJECT_IMPL


using namespace EagleLib;
using namespace EagleLib::Nodes;
void ForegroundEstimate::NodeInit(bool firstInit)
{
	_build_model = false;
	if (firstInit)
	{
		addInputParameter<cv::cuda::GpuMat>("Input point cloud");
		updateParameter<float>("Radius", 5.0);
	}
	updateParameter<float>("Epsilon", 1.0);
	updateParameter<int>("Checks", -1);
	updateParameter<std::function<void(void)>>("Build index", std::function<void(void)>([this](){_build_model = true;}));
}


void ForegroundEstimate::BuildModel(cv::cuda::GpuMat& tensor, cv::cuda::Stream& stream)
{
	if (tensor.cols && tensor.rows)
	{
		flann::KDTreeCuda3dIndexParams params;
		params["input_is_gpu_float4"] = true;
		flann::Matrix<float> input_ = flann::Matrix<float>((float*)tensor.data, tensor.rows, 3, tensor.step);
		nnIndex.reset(new flann::GpuIndex<flann::L2<float>>(input_, params));
		nnIndex->buildIndex();
	}
}


cv::cuda::GpuMat ForegroundEstimate::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	cv::cuda::GpuMat tensor, cv32f;
	cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat>(0)->Data();
	if(!input)
		input = &img;

	if(input->depth() != CV_32F)
	{
		cv::cuda::createContinuous(input->size(), CV_32F, cv32f);
		input->convertTo(cv32f, CV_32F, stream);
	}else
	{
		if(input->isContinuous())
			cv32f = *input;
		else
		{
			cv::cuda::createContinuous(input->size(), CV_MAKE_TYPE(CV_32F, input->channels()), cv32f);
			input->copyTo(cv32f, stream);
		}
	}
	tensor = cv32f.reshape(1, cv32f.rows*cv32f.cols);
	cv::Mat dbgTensor(cv32f);
	if(tensor.cols != 4)
	{
		cv::cuda::GpuMat tmp;
		cv::cuda::createContinuous(tensor.rows, 4, CV_32F, tmp);
		tensor.copyTo(tmp.colRange(0,3), stream);
		tensor = tmp;
	}
	CV_Assert(tensor.cols == 4);
	if(_build_model)
	{
		BuildModel(tensor, stream);
		updateParameter("Background PointCloud", tensor, &stream)->type = Parameters::Parameter::Output;
		_build_model = false;	
		return img;
	}


	if (nnIndex) 
	{
		int elements = input->size().area();
		cv::cuda::GpuMat index, distance;
		index.create(1, elements, CV_32S);
		distance.create(1, elements, CV_32F);

		flann::Matrix<int> d_idx((int*)index.data, elements, 1, sizeof(int));
		flann::Matrix<float> d_dist((float*)distance.data, elements, 1, sizeof(float));

		flann::SearchParams searchParams;
		searchParams.matrices_in_gpu_ram = true;
		searchParams.eps = *getParameter<float>("Epsilon")->Data();
		searchParams.checks = *getParameter<int>("Checks")->Data();
		flann::Matrix<float> input_ = flann::Matrix<float>((float*)tensor.data, tensor.rows, 3, tensor.step);
		nnIndex->radiusSearch(input_, d_idx, d_dist, *getParameter<float>(1)->Data(), searchParams, cv::cuda::StreamAccessor::getStream(stream));
		index = index.reshape(1, input->rows);
		distance = distance.reshape(1, input->rows);
		cv::Mat dbgIdx(index);
		cv::Mat dbgDist(distance);
		updateParameter("NN Index", index, &stream)->type = Parameters::Parameter::Output;
		updateParameter("NN Distance", distance, &stream)->type = Parameters::Parameter::Output;
		cv::cuda::GpuMat point_mask;
		cv::cuda::threshold(index, point_mask, -1, 255, cv::THRESH_BINARY_INV, stream);
		cv::cuda::GpuMat tmp;
		point_mask.convertTo(tmp, CV_8U, stream);
		updateParameter("Point Mask", tmp, &stream)->type = Parameters::Parameter::Output;
	}
	else
	{
		//NODE_LOG(info) << "Background model not build yet";
	}
	return img;
}

//RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("flann_knl", ".cu")
NODE_DEFAULT_CONSTRUCTOR_IMPL(ForegroundEstimate, PtCloud, Extractor);