#include "SaveAnnotations.hpp"
#include <fstream>
#include <EagleLib/Nodes/NodeInfo.hpp>
#include "EagleLib/utilities/UiCallbackHandlers.h"
#include "EagleLib/utilities/CudaCallbacks.hpp"
#include <opencv2/imgproc.hpp>
#include <boost/lexical_cast.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
using namespace EagleLib;
using namespace EagleLib::Nodes;
SaveAnnotations::SaveAnnotations()
{

}

void SaveAnnotations::on_class_change(int new_class)
{
    current_class_param.UpdateData(new_class);
    draw();
}

void SaveAnnotations::select_rect(std::string window_name, cv::Rect rect, int flags, cv::Mat img)
{
    if(window_name != "legend")
    {
        if(labels && current_class >= 0 && current_class < labels->size())
        {
            DetectedObject obj;
            obj.boundingBox = cv::Rect2f(float(rect.x) / _original_image.cols, float(rect.y) / _original_image.rows,
                                         float(rect.width) / _original_image.cols, float(rect.height) / _original_image.rows);
            obj.detections.emplace_back((*labels)[current_class], 1.0, current_class);
            _annotations.push_back(obj);
            draw();
        }
    }
}

void SaveAnnotations::on_key(int key)
{
    if(key == 'c')
    {
        _annotations.clear();
        draw();
    }
    if(key == 13)
    {
        std::ofstream ofs;
        std::stringstream ss;
        ss << output_directory.string() << "/" << annotation_stem <<
              std::setw(6) << std::setfill('0') << save_count << ".json";
        ofs.open(ss.str());

        ss.str("");
        ss << output_directory.string() << "/" << image_stem
           << std::setw(6) << std::setfill('0') << save_count << ".png";

        std::string image_path = ss.str();
        cereal::JSONOutputArchive ar(ofs);
        ar(cereal::make_nvp("ImageFile", image_path));
        ar(cereal::make_nvp("Timestamp", input_param.GetTimestamp()));
        ar(cereal::make_nvp("Annotations", _annotations));
        if(detections && detections->size() > 0)
        {
            std::vector<DetectedObject> objs;
            for(const auto& detection : *detections)
            {
                if(object_class != -1)
                {
                    if(detection.detections.size() && detection.detections[0].classNumber == object_class)
                    {
                        objs.push_back(detection);
                    }
                }else
                {
                    objs.push_back(detection);
                }
            }
            ar(cereal::make_nvp("detections", objs));
        }
        cv::imwrite(image_path, _original_image);
        ++save_count;
        _annotations.clear();
    }
}

void rectangle(cv::cuda::GpuMat& img, const cv::Rect& rect, cv::Scalar color, int thickness)
{
    cv::Rect top(rect.tl(), cv::Size(rect.width, thickness));
    img(top).setTo(color);

    top.y += rect.height - thickness;
    img(top).setTo(color);

    top.width = thickness;
    top.height = rect.height;
    top.x = rect.x;
    top.y = rect.y;
    img(top).setTo(color);

    top.x = rect.x + rect.width - thickness;
    img(top).setTo(color);
}

void SaveAnnotations::draw()
{
    _draw_image = _original_image.clone();
    cv::Mat draw_image = _draw_image;
    for(int i = 0; i < _annotations.size(); ++i)
    {
        auto bb = _annotations[i].boundingBox;
        cv::rectangle(draw_image, cv::Rect(bb.x * draw_image.cols, bb.y * draw_image.rows,
                                           bb.width * draw_image.cols, bb.height * draw_image.rows),
                      h_lut.at<cv::Vec3b>(_annotations[i].detections[0].classNumber), 5);
    }
    if(detections)
    {
        for(const auto& detection : *detections)
        {
            auto bb = detection.boundingBox;
            if(detection.detections.size())
            cv::rectangle(draw_image, cv::Rect(bb.x * draw_image.cols, bb.y * draw_image.rows,
                                               bb.width * draw_image.cols, bb.height * draw_image.rows),
                          h_lut.at<cv::Vec3b>(detection.detections[0].classNumber), 5);
        }
    }
    if( labels && current_class != -1 && current_class < labels->size())
    {
        //cv::putText(draw_image, (*labels)[current_class], cv::Point(15, 25), cv::FONT_HERSHEY_COMPLEX, 0.7, h_lut.at<cv::Vec3b>(current_class));
    }

    size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    mo::ThreadSpecificQueue::Push([draw_image, this]()
    {
        GetDataStream()->GetWindowCallbackManager()->imshow("original", draw_image);
        //GetDataStream()->GetWindowCallbackManager()->imshow("legend", h_legend);
    }, gui_thread_id, this);
}

bool SaveAnnotations::ProcessImpl()
{
    /*if(label_file_param.modified)
    {
        labels.clear();
        std::ifstream ifs;
        ifs.open(label_file.string());
        if(ifs.is_open())
        {
            std::string label;
            while(std::getline(ifs, label, '\n'))
            {
                labels.push_back(label);
            }
        }
        LOG(info) << "Loaded " << labels.size() << " class labels from " << label_file.string();

        h_lut.create(1, labels.size(), CV_8UC3);
        for(int i = 0; i < labels.size(); ++i)
            h_lut.at<cv::Vec3b>(i) = cv::Vec3b(i*180 / labels.size(), 200, 255);
        cv::cvtColor(h_lut, h_lut, cv::COLOR_HSV2BGR);

        int legend_width = 100;
        int max_width = 0;
        for(auto& label : labels)
        {
            max_width = std::max<int>(max_width, label.size());
        }
        legend_width += max_width * 15;

        h_legend.create(labels.size() * 20 + 15, legend_width, CV_8UC3);
        h_legend.setTo(0);
        for(int i = 0; i < labels.size(); ++i)
        {
            cv::Vec3b color = h_lut.at<cv::Vec3b>(i);
            h_legend(cv::Rect(8, 5 + 20 * i, 50, 20)).setTo(color);
            cv::putText(h_legend, boost::lexical_cast<std::string>(i) + " " + labels[i],
                        cv::Point(65, 25 + 20 * i),
                        cv::FONT_HERSHEY_COMPLEX, 0.7,
                        cv::Scalar(color[0], color[1], color[2]));
        }

        label_file_param.modified = false;
        GetDataStream()->GetWindowCallbackManager()->imshow("legend", h_legend);
    }*/
    if(h_lut.cols != labels->size())
    {
        h_lut.create(1, labels->size(), CV_8UC3);
        for(int i = 0; i < labels->size(); ++i)
            h_lut.at<cv::Vec3b>(i) = cv::Vec3b(i*180 / labels->size(), 200, 255);
        cv::cvtColor(h_lut, h_lut, cv::COLOR_HSV2BGR);
    }
    _annotations.clear();
    size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    if(input->GetSyncState() == EagleLib::SyncedMemory::DEVICE_UPDATED)
    {
        cv::Mat img = input->GetMat(Stream());
        //cv::Mat img = input->GetMat(Stream());

        EagleLib::cuda::enqueue_callback_async([img, this]()
        {
            GetDataStream()->GetWindowCallbackManager()->imshow("original", img);
        }, gui_thread_id ,Stream());
        _original_image = img;
    }else
    {
        //cv::Mat img = input->GetMat(Stream());
        cv::Mat img = input->GetMat(Stream());

        mo::ThreadSpecificQueue::Push([img, this]()
        {
            GetDataStream()->GetWindowCallbackManager()->imshow("original", img);
        }, gui_thread_id, this);
        _original_image = img;
    }
    return true;
}

MO_REGISTER_CLASS(SaveAnnotations)
