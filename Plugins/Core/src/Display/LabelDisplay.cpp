#include "LabelDisplay.hpp"
#include <EagleLib/Nodes/NodeInfo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>


#include <fstream>
#include <boost/filesystem.hpp>
using namespace EagleLib::Nodes;

bool LabelDisplay::ProcessImpl()
{
    if(label_file_param.modified || (labels.empty() && !label_file.empty()))
    {
        if(boost::filesystem::exists(label_file))
        {
            labels.clear();
            std::ifstream ifs(label_file.string());
            std::string label;

            while(std::getline(ifs, label))
                labels.push_back(label);
            num_classes_param.UpdateData(labels.size());
        }
        label_file_param.modified = false;
    }
    if(num_classes_param.modified || d_lut.empty() ||
       (display_legend && d_legend.size() != label->GetSize()))
    {
        h_lut.create(1, num_classes, CV_8UC3);
        for(int i = 0; i < num_classes; ++i)
            h_lut.at<cv::Vec3b>(i) = cv::Vec3b(i*180 / num_classes, 200, 255);
        cv::cvtColor(h_lut, h_lut, cv::COLOR_HSV2BGR);
        if(ignore_class != -1 && ignore_class < num_classes)
            h_lut.at<cv::Vec3b>(ignore_class) = cv::Vec3b(0,0,0);

        d_lut.upload(h_lut, Stream());
        num_classes_param.modified = false;
        if(display_legend)
        {
            cv::Mat legend;//(label->GetSize(), CV_8UC3);
            if(original_image)
            {
                if(num_classes != labels.size())
                {
                    labels.clear();
                    std::stringstream ss;
                    for(int i = 0; i < num_classes; ++i)
                    {
                        ss << i;
                        labels.push_back(ss.str());
                        ss.str(std::string());
                    }
                }

                legend.create(original_image->GetSize(), CV_8UC3);
                legend.setTo(0);
                legend_width = 100;
                int max_width = 0;
                for(auto& label : labels)
                {
                    max_width = std::max<int>(max_width, label.size());
                }
                legend_width += max_width * 15;

                cv::Rect legend_outline(3,3,legend_width , num_classes * 20 + 15);
                cv::rectangle(legend, legend_outline, cv::Scalar(0,255,0));

                for(int i = 0; i < num_classes; ++i)
                {
                    cv::Vec3b color = h_lut.at<cv::Vec3b>(i);
                    //cv::rectangle(legend, cv::Rect(8, 5 + 10 * i, 20, 20), color, 1, -1);
                    legend(cv::Rect(8, 5 + 20 * i, 50, 20)).setTo(color);
                    cv::putText(legend, labels[i], cv::Point(65, 25 + 20 * i),
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
    EagleLib::applyColormap(input, output, d_lut, Stream());

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
            combined(cv::Rect(3,3,legend_width , num_classes*20 + 15)).setTo(cv::Scalar::all(0), Stream());
            cv::cuda::add(combined, d_legend, combined, cv::noArray(), -1, Stream());
        }
        colorized_param.UpdateData(combined, original_image_param.GetTimestamp(), _ctx);
        return true;
    }
    return false;

}

MO_REGISTER_CLASS(LabelDisplay)
