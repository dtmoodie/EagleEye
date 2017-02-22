#include "CaffeSSDHandler.hpp"
#include "CaffeNetHandlerInfo.hpp"
#include "helpers.hpp"
using namespace EagleLib::Caffe;


std::map<int, int> SSDHandler::CanHandleNetwork(const caffe::Net<float>& net)
{
    const std::vector<int>& out_idx = net.output_blob_indices();
    auto layer_names = net.layer_names();
    auto layers = net.layers();
    std::map<int, int> output;
    for(int i = 0; i < layer_names.size(); ++i)
    {
        std::vector<int> top_ids = net.top_ids(i);
        for(auto id : top_ids)
        {
            if(std::find(out_idx.begin(), out_idx.end(), id) != out_idx.end())
            {
                // Layer(i) outputs from network
                std::string type = layers[i]->type();
                if(type == "DetectionOutput")
                {
                    output[id] = 10;
                }
            }
        }
    }
    return output;
}

void SSDHandler::HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size)
{
    auto output_blob= net.blob_by_name(output_blob_name);
    if(!output_blob)
        return;
    float* begin = output_blob->mutable_cpu_data();
    std::vector<DetectedObject> objects;

    const int num_detections = output_blob->height();
    cv::Mat all(num_detections, 7, CV_32F, begin);
    cv::Mat_<float> roi_num(num_detections, 1, begin, sizeof(float)*7);
    cv::Mat_<float> labels(num_detections, 1, begin + 1, sizeof(float)*7);
    cv::Mat_<float> confidence(num_detections, 1, begin + 2, sizeof(float)*7);
    cv::Mat_<float> xmin(num_detections, 1, begin + 3, sizeof(float) * 7);
    cv::Mat_<float> ymin(num_detections, 1, begin + 4, sizeof(float) * 7);
    cv::Mat_<float> xmax(num_detections, 1, begin + 5, sizeof(float) * 7);
    cv::Mat_<float> ymax(num_detections, 1, begin + 6, sizeof(float) * 7);

    for(int i = 0; i < num_detections; ++i)
    {

        if((detection_threshold.size() == 1 && confidence[i][0] > detection_threshold[0]) ||
            (labels[i][0] < detection_threshold.size() && confidence[i][0] > detection_threshold[int(labels[i][0])]))
        {
            int num = roi_num[i][0];
            DetectedObject obj;
            obj.boundingBox.x = xmin[i][0] * bounding_boxes[num].width + bounding_boxes[num].x;
            obj.boundingBox.y = ymin[i][0] * bounding_boxes[num].height + bounding_boxes[num].y;
            obj.boundingBox.width = (xmax[i][0] - xmin[i][0]) * bounding_boxes[num].width;
            obj.boundingBox.height = (ymax[i][0] - ymin[i][0]) * bounding_boxes[num].height;
            obj.timestamp = timestamp;
            // Check all current objects iou value
            bool append = true;

            if (this->labels && labels[i][0] < this->labels->size())
                obj.detections.emplace_back((*this->labels)[int(labels[i][0])], confidence[i][0], int(labels[i][0]));
            else
                obj.detections.emplace_back("", confidence[i][0], int(labels[i][0]));

            for(auto itr = objects.begin(); itr != objects.end(); ++itr)
            {
                float iou_val = iou(obj.boundingBox, itr->boundingBox);
                if(iou_val > 0.2)
                {
                    if(obj.detections[0].confidence > itr->detections[0].confidence)
                    {
                        // Current object has higher prediction, replace
                        *itr = obj;
                    }
                    append = false;
                }
            }
            if(append)
                objects.push_back(obj);
        }
    }
    begin += output_blob->width() * output_blob->height() * num_detections;
    if(objects.size())
    {
        LOG(trace) << "Detected " << objects.size() << " objets in frame " << timestamp;
    }
    num_detections_param.UpdateData(objects.size(), timestamp, _ctx);

    detections_param.UpdateData(objects, timestamp, _ctx);
}

MO_REGISTER_CLASS(SSDHandler)
