#include "SaveAnnotations.hpp"

#include <Aquila/core/IGraph.hpp>
#include <Aquila/gui/UiCallbackHandlers.h>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetectionSerialization.hpp>

#include <MetaObject/serialization/JSONPrinter.hpp>

#include <boost/lexical_cast.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include <fstream>
#include <iomanip>

#include <ct/reflect/cerealize.hpp>
#include <opencv2/imgproc.hpp>

using namespace aq;
using namespace aq::nodes;

SaveAnnotations::SaveAnnotations() {}

void SaveAnnotations::on_class_change(int new_class)
{
    current_class = new_class;
    draw();
}

void SaveAnnotations::select_rect(std::string window_name, cv::Rect rect, int flags, cv::Mat img)
{
    if (window_name != "legend")
    {
        auto cats = detections->getCatSet();
        if (current_class >= 0 && current_class < cats->size())
        {
            DetectedObject obj;
            obj.bounding_box = cv::Rect2f(float(rect.x) / _original_image.cols,
                                          float(rect.y) / _original_image.rows,
                                          float(rect.width) / _original_image.cols,
                                          float(rect.height) / _original_image.rows);

            obj.classifications[0] = (*cats)[current_class](1.0);
            _annotations.push_back(obj);
            draw();
        }
    }
}

void SaveAnnotations::on_key(int key)
{
    if (key == 'c')
    {
        _annotations.clear();
        draw();
    }
    if (key == 13)
    {
        std::ofstream ofs;
        std::stringstream ss;
        ss << output_directory.string() << "/" << annotation_stem << std::setw(6) << std::setfill('0') << save_count
           << ".json";
        ofs.open(ss.str());

        ss.str("");
        ss << output_directory.string() << "/" << image_stem << std::setw(6) << std::setfill('0') << save_count
           << ".png";

        std::string image_path = ss.str();

        mo::JSONSaver ar(ofs);
        // cereal::JSONOutputArchive ar(ofs);
        const mo::OptionalTime timestamp = input_param.getNewestTimestamp();
        ar(&image_path, "ImageFile");
        ar(&timestamp, "Timestamp");
        ar(&_annotations, "Annotations");
        if (detections)
        {
            // TODO filter by object_class?
            ar(detections, "detections");
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
    const auto num_entities = _annotations.getNumEntities();
    if (num_entities > 0)
    {
        mt::Tensor<const aq::detection::BoundingBox2d::DType, 1> bounding_boxes =
            _annotations.getComponent<aq::detection::BoundingBox2d>();
        mt::Tensor<const aq::detection::Classifications, 1> classifications =
            _annotations.getComponent<aq::detection::Classifications>();
        for (size_t i = 0; i < num_entities; ++i)
        {
            cv::Rect2f bb = bounding_boxes[i];
            boundingBoxToPixels(bb, draw_image.size());
            const cv::Scalar color = classifications[i][0].cat->color;
            cv::rectangle(draw_image, cv::Rect(bb.x, bb.y, bb.width, bb.height), color, 5);
        }
    }

    if (detections)
    {
        mt::Tensor<const aq::detection::BoundingBox2d::DType, 1> bounding_boxes =
            detections->getComponent<aq::detection::BoundingBox2d>();

        mt::Tensor<const aq::detection::Classifications, 1> classifications =
            detections->getComponent<aq::detection::Classifications>();

        const uint32_t num_entities = detections->getNumEntities();

        for (uint32_t i = 0; i < num_entities; ++i)
        {
            aq::detection::BoundingBox2d::DType bb = bounding_boxes[i];
            boundingBoxToPixels(bb, draw_image.size());
            const cv::Scalar color = classifications[i][0].cat->color;

            cv::rectangle(draw_image,
                          cv::Rect(bb.x * draw_image.cols,
                                   bb.y * draw_image.rows,
                                   bb.width * draw_image.cols,
                                   bb.height * draw_image.rows),
                          color,
                          5);
        }
    }
    if (current_class != -1 && current_class < _cats->size())
    {
        // cv::putText(draw_image, (*labels)[current_class], cv::Point(15, 25), cv::FONT_HERSHEY_COMPLEX, 0.7,
        // h_lut.at<cv::Vec3b>(current_class));
    }
    this->getGraph()->getObject<aq::WindowCallbackHandler>()->imshow("original", draw_image);
}

bool SaveAnnotations::processImpl()
{
    /*if(label_file_param.modified())
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
        MO_LOG(info) << "Loaded " << labels.size() << " class labels from " << label_file.string();

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

        label_file_param.modified(false);
        getGraph()->getWindowCallbackManager()->imshow("legend", h_legend);
    }*/
    _cats = detections->getCatSet();

    _annotations.clear();

    auto window_manager = this->getGraph()->getObject<aq::WindowCallbackHandler>();
    auto stream = this->getStream();
    bool sync = false;
    cv::Mat img = input->getMat(stream.get(), &sync);
    if (sync)
    {
        stream->pushWork([img, window_manager]() { window_manager->imshow("original", img); });
        _original_image = img;
    }
    else
    {
        window_manager->imshow("original", img);
        _original_image = img;
    }
    return true;
}

MO_REGISTER_CLASS(SaveAnnotations)
