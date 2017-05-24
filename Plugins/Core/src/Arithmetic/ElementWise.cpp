#include "ElementWise.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
using namespace aq;
using namespace aq::Nodes;

bool Equal::ProcessImpl()
{
    if(input->getSyncState() < input->DEVICE_UPDATED)
    {
        const cv::Mat& in = input->getMat(Stream());
        cv::Mat out;
        cv::compare(in, cv::Scalar(value), out, cv::CMP_EQ);
        output_param.UpdateData(out, input_param.GetTimestamp(), _ctx);
    }else
    {
        const cv::cuda::GpuMat& in = input->getGpuMat(Stream());
        cv::cuda::GpuMat out;
        cv::cuda::compare(in, cv::Scalar(value), out, cv::CMP_EQ, Stream());
        output_param.UpdateData(out, input_param.GetTimestamp(), _ctx);
    }

    return true;
}

MO_REGISTER_CLASS(Equal)

bool AddBinary::ProcessImpl()
{
    if(in1->getSyncState() < in1->DEVICE_UPDATED &&
       in2->getSyncState() < in1->DEVICE_UPDATED)
    {
        const cv::Mat& in1_mat = in1->getMat(Stream());
        const cv::Mat& in2_mat = in2->getMat(Stream());
        cv::Mat out;
        if(weight1 != 1.0 || weight2 != 1.0)
        {
            cv::addWeighted(in1_mat, weight1, in2_mat, weight2, 1.0, out);
        }else
        {
            out = in1_mat + in2_mat;
        }
        output_param.UpdateData(out, in1_param.GetTimestamp(), _ctx);
    }else
    {
        const cv::cuda::GpuMat& in1_mat = in1->getGpuMat(Stream());
        const cv::cuda::GpuMat& in2_mat = in2->getGpuMat(Stream());
        cv::cuda::GpuMat out;
        if(weight1 != 1.0 || weight2 != 1.0)
        {
            cv::cuda::addWeighted(in1_mat, weight1, in2_mat, weight2, 1.0, out,
                                  -1, Stream());
        }else
        {
            //out = in1_mat + in2_mat;
            cv::cuda::add(in1_mat, in2_mat, out, cv::noArray(), -1, Stream());
        }
        output_param.UpdateData(out, in1_param.GetTimestamp(), _ctx);
    }
    return true;
}

MO_REGISTER_CLASS(AddBinary)
