#include "ElementWise.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
using namespace aq;
using namespace aq::Nodes;

bool Equal::processImpl()
{
    if(input->getSyncState() < input->DEVICE_UPDATED)
    {
        const cv::Mat& in = input->getMat(stream());
        cv::Mat out;
        cv::compare(in, cv::Scalar(value), out, cv::CMP_EQ);
        output_param.updateData(out, input_param.getTimestamp(), _ctx);
    }else
    {
        const cv::cuda::GpuMat& in = input->getGpuMat(stream());
        cv::cuda::GpuMat out;
        cv::cuda::compare(in, cv::Scalar(value), out, cv::CMP_EQ, stream());
        output_param.updateData(out, input_param.getTimestamp(), _ctx);
    }

    return true;
}

MO_REGISTER_CLASS(Equal)

bool AddBinary::processImpl()
{
    if(in1->getSyncState() < in1->DEVICE_UPDATED &&
       in2->getSyncState() < in1->DEVICE_UPDATED)
    {
        const cv::Mat& in1_mat = in1->getMat(stream());
        const cv::Mat& in2_mat = in2->getMat(stream());
        cv::Mat out;
        if(weight1 != 1.0 || weight2 != 1.0)
        {
            cv::addWeighted(in1_mat, weight1, in2_mat, weight2, 1.0, out);
        }else
        {
            out = in1_mat + in2_mat;
        }
        output_param.updateData(out, in1_param.getTimestamp(), _ctx);
    }else
    {
        const cv::cuda::GpuMat& in1_mat = in1->getGpuMat(stream());
        const cv::cuda::GpuMat& in2_mat = in2->getGpuMat(stream());
        cv::cuda::GpuMat out;
        if(weight1 != 1.0 || weight2 != 1.0)
        {
            cv::cuda::addWeighted(in1_mat, weight1, in2_mat, weight2, 1.0, out,
                                  -1, stream());
        }else
        {
            //out = in1_mat + in2_mat;
            cv::cuda::add(in1_mat, in2_mat, out, cv::noArray(), -1, stream());
        }
        output_param.updateData(out, in1_param.getTimestamp(), _ctx);
    }
    return true;
}

MO_REGISTER_CLASS(AddBinary)
