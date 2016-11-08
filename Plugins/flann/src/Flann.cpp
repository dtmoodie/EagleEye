#include "Flann.h"
#include "thrust/transform.h"
#include "thrust/transform_reduce.h"
#include "thrust/count.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_imgproc.hpp>
#include <EagleLib/Nodes/NodeInfo.hpp>

#include "RuntimeSourceDependency.h"
SETUP_PROJECT_IMPL


using namespace EagleLib;
using namespace EagleLib::Nodes;



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

bool ForegroundEstimate::ProcessImpl()
{
    const cv::cuda::GpuMat& input = input_point_cloud->GetGpuMat(Stream());
    cv::cuda::GpuMat cv32f;
    if (input.depth() != CV_32F)
    {
        cv::cuda::createContinuous(input.size(), CV_32F, cv32f);
        input.convertTo(cv32f, CV_32F, Stream());
    }
    else
    {
        if (input.isContinuous())
            cv32f = input;
        else
        {
            cv::cuda::createContinuous(input.size(), CV_MAKE_TYPE(CV_32F, input.channels()), cv32f);
            input.copyTo(cv32f, Stream());
        }
    }
    cv::cuda::GpuMat tensor = cv32f.reshape(1, cv32f.rows*cv32f.cols);
    
    if (tensor.cols != 4)
    {
        cv::cuda::GpuMat tmp;
        cv::cuda::createContinuous(tensor.rows, 4, CV_32F, tmp);
        tensor.copyTo(tmp.colRange(0, 3), Stream());
        tensor = tmp;
    }
    CV_Assert(tensor.cols == 4);
    if (build_model)
    {
        BuildModel(tensor, Stream());
        background_model_param.UpdateData(tensor, input_point_cloud_param.GetTimestamp(), _ctx);
        build_model = false;
        return true;
    }


    if (nnIndex)
    {
        int elements = input.size().area();
        cv::cuda::GpuMat index, distance;
        index.create(1, elements, CV_32S);
        distance.create(1, elements, CV_32F);

        flann::Matrix<int> d_idx((int*)index.data, elements, 1, sizeof(int));
        flann::Matrix<float> d_dist((float*)distance.data, elements, 1, sizeof(float));

        flann::SearchParams searchParams;
        searchParams.matrices_in_gpu_ram = true;
        searchParams.eps = epsilon;
        searchParams.checks = checks;
        flann::Matrix<float> input_ = flann::Matrix<float>((float*)tensor.data, tensor.rows, 3, tensor.step);
        nnIndex->radiusSearch(input_, d_idx, d_dist, radius, searchParams, cv::cuda::StreamAccessor::getStream(Stream()));
        index = index.reshape(1, input.rows);
        distance = distance.reshape(1, input.rows);
        index_param.UpdateData(index, input_point_cloud_param.GetTimestamp(), _ctx);
        distance_param.UpdateData(distance, input_point_cloud_param.GetTimestamp(), _ctx);
        cv::cuda::GpuMat point_mask;
        cv::cuda::threshold(index, point_mask, -1, 255, cv::THRESH_BINARY_INV, Stream());
        cv::cuda::GpuMat tmp;
        point_mask.convertTo(tmp, CV_8U, Stream());
        point_mask_param.UpdateData(point_mask, input_point_cloud_param.GetTimestamp(), _ctx);
    }
    else
    {
        //NODE_LOG(info) << "Background model not build yet";
    }
}



MO_REGISTER_CLASS(ForegroundEstimate);