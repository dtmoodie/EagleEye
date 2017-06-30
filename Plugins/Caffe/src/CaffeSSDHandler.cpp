#include "CaffeSSDHandler.hpp"
#include "CaffeNetHandlerInfo.hpp"
#include "helpers.hpp"
using namespace aq::Caffe;

std::map<int, int> SSDHandler::CanHandleNetwork(const caffe::Net<float>& net){
    const std::vector<int>& out_idx = net.output_blob_indices();
    auto layer_names = net.layer_names();
    auto layers = net.layers();
    std::map<int, int> output;
    for(size_t i = 0; i < layer_names.size(); ++i){
        std::vector<int> top_ids = net.top_ids(static_cast<int>(i));
        for(auto id : top_ids){
            if(std::find(out_idx.begin(), out_idx.end(), id) != out_idx.end()){
                // Layer(i) outputs from network
                std::string type = layers[i]->type();
                if(type == "DetectionOutput"){
                    output[id] = 10;
                }
            }
        }
    }
    return output;
}


void SSDHandler::handleOutput(const caffe::Net<float>& net, const std::vector<cv::Rect>& bounding_boxes, mo::ITParam<aq::SyncedMemory>& input_param, const std::vector<aq::DetectedObject2d>& objs){
    (void)objs;
    auto output_blob= net.blob_by_name(output_blob_name);
    if(!output_blob)
        return;
    float* begin = output_blob->mutable_cpu_data();
    std::vector<DetectedObject> objects;

    const int num_detections = output_blob->height();
    cv::Mat_<float> roi_num(num_detections, 1, begin, sizeof(float)*7);
    cv::Mat_<float> labels(num_detections, 1, begin + 1, sizeof(float)*7);
    cv::Mat_<float> confidence(num_detections, 1, begin + 2, sizeof(float)*7);
    cv::Mat_<float> xmin(num_detections, 1, begin + 3, sizeof(float) * 7);
    cv::Mat_<float> ymin(num_detections, 1, begin + 4, sizeof(float) * 7);
    cv::Mat_<float> xmax(num_detections, 1, begin + 5, sizeof(float) * 7);
    cv::Mat_<float> ymax(num_detections, 1, begin + 6, sizeof(float) * 7);

    for(size_t i = 0; i < static_cast<size_t>(num_detections); ++i){
        if((detection_threshold.size() == 1 && confidence[static_cast<int>(i)][0] > detection_threshold[0]) ||
            (labels[static_cast<int>(i)][0] < detection_threshold.size() && 
             confidence[static_cast<int>(i)][0] > detection_threshold[static_cast<size_t>(labels[static_cast<int>(i)][0])]))
        {
            size_t num = static_cast<size_t>(roi_num[static_cast<int>(i)][0]);
            DetectedObject obj;
            obj.boundingBox.x = xmin[static_cast<int>(i)][0] * bounding_boxes[num].width + bounding_boxes[num].x;
            obj.boundingBox.y = ymin[static_cast<int>(i)][0] * bounding_boxes[num].height + bounding_boxes[num].y;
            obj.boundingBox.width = (xmax[static_cast<int>(i)][0] - xmin[static_cast<int>(i)][0]) * bounding_boxes[num].width;
            obj.boundingBox.height = (ymax[static_cast<int>(i)][0] - ymin[static_cast<int>(i)][0]) * bounding_boxes[num].height;
            obj.timestamp = input_param.getTimestamp();
            obj.framenumber = input_param.getFrameNumber();
            obj.id = current_id++;
            // Check all current objects iou value
            bool append = true;

            if (this->labels && labels[static_cast<int>(i)][0] < this->labels->size())
                obj.classification = Classification((*this->labels)[size_t(labels[static_cast<int>(i)][0])], confidence[static_cast<int>(i)][0], int(labels[static_cast<int>(i)][0]));
            else
                obj.classification = Classification("", confidence[static_cast<int>(i)][0], int(labels[static_cast<int>(i)][0]));

            for(auto itr = objects.begin(); itr != objects.end(); ++itr)
            {
                float iou_val = iou(obj.boundingBox, itr->boundingBox);
                if(iou_val > overlap_threshold)
                {
                    if(obj.classification.confidence > itr->classification.confidence)
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
        LOG(trace) << "Detected " << objects.size() << " objets in frame " << input_param.getFrameNumber();
    }
    num_detections_param.updateData(static_cast<int>(objects.size()), input_param.getTimestamp(), input_param.getFrameNumber(), _ctx);

    detections_param.updateData(objects, input_param.getTimestamp(), input_param.getFrameNumber(), _ctx);
}

MO_REGISTER_CLASS(SSDHandler)
