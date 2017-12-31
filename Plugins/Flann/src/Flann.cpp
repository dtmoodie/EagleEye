#include "Flann.h"
#include "thrust/count.h"
#include "thrust/transform.h"
#include "thrust/transform_reduce.h"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>
#include <MetaObject/Logging/Profiling.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

using namespace aq;
using namespace aq::nodes;

void ForegroundEstimate::BuildModel(cv::cuda::GpuMat& tensor, cv::cuda::Stream& stream)
{
    SCOPED_PROFILE_NODE
    if (tensor.cols && tensor.rows)
    {
        flann::KDTreeCuda3dIndexParams params;
        params["input_is_gpu_float4"] = true;
        flann::Matrix<float> input_ = flann::Matrix<float>((float*)tensor.data, tensor.rows, 3, tensor.step);
        nnIndex.reset(new flann::GpuIndex<flann::L2<float>>(input_, params));
        nnIndex->buildIndex();
    }
}

bool ForegroundEstimate::processImpl()
{
    SCOPED_PROFILE_NODE
    cv::cuda::GpuMat input = input_point_cloud->getGpuMat(stream());
    cv::cuda::GpuMat cv32f;
    if (input.depth() != CV_32F)
    {
        cv::cuda::createContinuous(input.size(), CV_32F, cv32f);
        input.convertTo(cv32f, CV_32F, stream());
    }
    else
    {
        if (input.isContinuous())
            cv32f = input;
        else
        {
            cv::cuda::createContinuous(input.size(), CV_MAKE_TYPE(CV_32F, input.channels()), cv32f);
            input.copyTo(cv32f, stream());
        }
    }
    cv::cuda::GpuMat tensor = cv32f.reshape(1, cv32f.rows * cv32f.cols);

    if (tensor.cols != 4)
    {
        cv::cuda::GpuMat tmp;
        cv::cuda::createContinuous(tensor.rows, 4, CV_32F, tmp);
        tensor.copyTo(tmp.colRange(0, 3), stream());
        tensor = tmp;
    }
    CV_Assert(tensor.cols == 4);
    if (build_model)
    {
        BuildModel(tensor, stream());
        background_model_param.updateData(
            input, mo::tag::_param = input_point_cloud_param, mo::tag::_context = _ctx.get());
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
        nnIndex->radiusSearch(input_, d_idx, d_dist, radius, searchParams, cudaStream());
        index = index.reshape(1, input.rows);
        distance = distance.reshape(1, input.rows);
        index_param.updateData(index, mo::tag::_param = input_point_cloud_param, _ctx.get());
        distance_param.updateData(distance, mo::tag::_param = input_point_cloud_param, _ctx.get());
        cv::cuda::GpuMat point_mask;
        cv::cuda::threshold(index, point_mask, -1, 255, cv::THRESH_BINARY_INV, stream());
        cv::cuda::GpuMat tmp;
        point_mask.convertTo(tmp, CV_8U, stream());
        point_mask_param.updateData(tmp, mo::tag::_param = input_point_cloud_param, _ctx.get());
        if (foreground_param.hasSubscriptions())
        {
            cv::Mat mask = this->point_mask.getMat(stream());
            cv::Mat point_cloud = input_point_cloud->getMat(stream());
            stream().waitForCompletion();
            int points = cv::countNonZero(mask);
            cv::Mat foreground(1, points, CV_32FC3);
            int count = 0;
            for (int i = 0; i < mask.rows; ++i)
            {
                for (int j = 0; j < mask.cols; ++j)
                {
                    if (mask.at<uchar>(i, j))
                    {
                        foreground.at<cv::Vec3f>(count) = point_cloud.at<cv::Vec3f>(i, j);
                    }
                }
            }
            this->foreground_param.updateData(foreground, mo::tag::_param = input_point_cloud_param, _ctx.get());
        }
        return true;
    }
    else
    {
        return false;
    }
}

MO_REGISTER_CLASS(ForegroundEstimate);