#include "LabelDisplay.hpp"
#include <Aquila/Nodes/NodeInfo.hpp>
#include <Aquila/ObjectDetection.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>


#include <fstream>
#include <boost/filesystem.hpp>
using namespace aq::Nodes;

bool LabelDisplay::ProcessImpl()
{
    if(d_lut.empty() ||
       (display_legend && d_legend.size() != label->GetSize()))
    {

        CreateColormap(h_lut, labels->size(), ignore_class);

        d_lut.upload(h_lut, Stream());
        if(display_legend)
        {
            cv::Mat legend;//(label->GetSize(), CV_8UC3);
            if(original_image)
            {
                legend.create(original_image->GetSize(), CV_8UC3);
                legend.setTo(0);
                legend_width = 100;
                int max_width = 0;
                legend_width += max_width * 15;

                cv::Rect legend_outline(3,3,legend_width , labels->size() * 20 + 15);
                cv::rectangle(legend, legend_outline, cv::Scalar(0,255,0));

                for(int i = 0; i < labels->size(); ++i)
                {
                    cv::Vec3b color = h_lut.at<cv::Vec3b>(i);
                    //cv::rectangle(legend, cv::Rect(8, 5 + 10 * i, 20, 20), color, 1, -1);
                    legend(cv::Rect(8, 5 + 20 * i, 50, 20)).setTo(color);
                    cv::putText(legend, (*labels)[i], cv::Point(65, 25 + 20 * i),
                                cv::FONT_HERSHEY_COMPLEX, 0.7,
                                cv::Scalar(color[0], color[1], color[2]));
                }
                d_legend.upload(legend, Stream());
            }
        }
    }

    cv::cuda::GpuMat input;
    if(dilate != 0)
    {
        if(!_dilate_filter || dilate_param.modified)
        {
            _dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,label->GetType(),
                                                              cv::getStructuringElement(cv::MORPH_CROSS, {dilate, dilate}));
            dilate_param.modified = false;
        }
        _dilate_filter->apply(label->GetGpuMat(Stream()), input, Stream());
    }else
    {
        input = label->GetGpuMat(Stream());
    }
    cv::cuda::GpuMat output;
    aq::applyColormap(input, output, d_lut, Stream());

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
        if(display_legend && d_legend.size() == combined.size())
        {
            combined(cv::Rect(3,3,legend_width , labels->size()*20 + 15)).setTo(cv::Scalar::all(0), Stream());
            cv::cuda::add(combined, d_legend, combined, cv::noArray(), -1, Stream());
        }
        colorized_param.UpdateData(combined, original_image_param.GetTimestamp(), _ctx);
        return true;
    }
    return false;

}

MO_REGISTER_CLASS(LabelDisplay)
