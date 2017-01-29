#include "LabelDisplay.hpp"
#include <opencv2/imgproc.hpp>
#include <EagleLib/Nodes/NodeInfo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
using namespace EagleLib::Nodes;

bool LabelDisplay::ProcessImpl()
{
    if(num_classes_param.modified || d_lut.empty())
    {
        h_lut.create(1, num_classes, CV_8UC3);
        for(int i = 0; i < num_classes; ++i)
            h_lut.at<cv::Vec3b>(i) = cv::Vec3b(i*180 / num_classes, 200, 255);
        cv::cvtColor(h_lut, h_lut, cv::COLOR_HSV2BGR);
        if(ignore_class != -1 && ignore_class < num_classes)
            h_lut.at<cv::Vec3b>(ignore_class) = cv::Vec3b(0,0,0);

        d_lut.upload(h_lut, Stream());
        num_classes_param.modified = false;
    }
    cv::cuda::GpuMat output;
    EagleLib::applyColormap(label->GetGpuMat(Stream()), output, d_lut, Stream());

    if(original_image == nullptr)
    {
        colorized_param.UpdateData(output,label_param.GetTimestamp(), _ctx);
        return true;
    }else
    {
        cv::cuda::GpuMat input = original_image->GetGpuMat(Stream());
        cv::cuda::GpuMat resized;
        if(output.size() != input.size())
        {
            cv::cuda::resize(output, resized, input.size(),0, 0, cv::INTER_LINEAR, Stream());
        }else
        {
            resized = output;
        }

        cv::cuda::GpuMat combined;
        cv::cuda::addWeighted(input, 1.0 - label_weight, resized, label_weight, 0.0, combined, -1, Stream());
        colorized_param.UpdateData(combined, original_image_param.GetTimestamp(), _ctx);
        return true;
    }
    return false;

}

MO_REGISTER_CLASS(LabelDisplay)
