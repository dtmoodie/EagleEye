#include "INeuralNet.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

void aq::nodes::INeuralNet::preBatch(int batch_size){
}

bool aq::nodes::INeuralNet::processImpl(){
    return forwardAll();
}
bool aq::nodes::INeuralNet::forwardAll(){
    std::vector<cv::Rect2f> defaultROI;
    auto input_image_shape = input->getShape();
    defaultROI.push_back(cv::Rect2f(0, 0, 1.0, 1.0));
    if (bounding_boxes == nullptr){
        bounding_boxes = &defaultROI;
    }

    if (input_detections != nullptr && bounding_boxes == &defaultROI){
        defaultROI.clear();
        for (const auto& itr : *input_detections){
            defaultROI.emplace_back(
                itr.boundingBox.x / input_image_shape[2],
                itr.boundingBox.y / input_image_shape[1],
                itr.boundingBox.width / input_image_shape[2],
                itr.boundingBox.height / input_image_shape[1]);
        }
        if (defaultROI.size() == 0)
            return false;
    }
    std::vector<cv::Rect> pixel_bounding_boxes;
    for (int i = 0; i < bounding_boxes->size(); ++i){
        cv::Rect bb;
        bb.x = static_cast<int>((*bounding_boxes)[i].x * input_image_shape[2]);
        bb.y = static_cast<int>((*bounding_boxes)[i].y * input_image_shape[1]);
        bb.width = static_cast<int>((*bounding_boxes)[i].width * input_image_shape[2]);
        bb.height = static_cast<int>((*bounding_boxes)[i].height * input_image_shape[1]);
        if (bb.x + bb.width >= input_image_shape[2]){
            bb.x -= input_image_shape[2] - bb.width;
        }
        if (bb.y + bb.height >= input_image_shape[1]){
            bb.y -= input_image_shape[1] - bb.height;
        }
        bb.x = std::max(0, bb.x);
        bb.y = std::max(0, bb.y);
        pixel_bounding_boxes.push_back(bb);
    }
    if(image_scale > 0){
        reshapeNetwork(bounding_boxes->size(), input_image_shape[3], static_cast<int>(input_image_shape[1] * image_scale), static_cast<int>(input_image_shape[2] * image_scale));
    }
    cv::cuda::GpuMat float_image;
    if(input->getDepth() != CV_32F){
        input->getGpuMat(stream()).convertTo(float_image, CV_32F, stream());
    }else{
        input->clone(float_image, stream());
    }
    cv::cuda::subtract(float_image, channel_mean, float_image, cv::noArray(), -1, stream());
    if(pixel_scale != 1.0f){
        cv::cuda::multiply(float_image, cv::Scalar::all(pixel_scale), float_image, 1.0, -1, stream());
    }
    int batch_size = 1;
    preBatch(batch_size);
    cv::cuda::GpuMat resized;
    auto net_input = getNetImageInput();
    MO_ASSERT(net_input.size());
    cv::Size net_input_size = net_input[0][0].size();
    for (int i = 0; i < pixel_bounding_boxes.size();){ // for each roi
        int start = i, end = 0;
        for (int j = 0; j < net_input.size() && i < pixel_bounding_boxes.size(); ++j, ++i){ // for each image in the mini batch
            if (pixel_bounding_boxes[i].size() != net_input_size) {
                cv::cuda::resize(float_image(pixel_bounding_boxes[i]), resized, net_input_size, 0, 0, cv::INTER_LINEAR, stream());
            }else {
                resized = float_image(pixel_bounding_boxes[i]);
            }
            cv::cuda::split(resized, net_input[j], stream());
            end = start + j + 1;
        }
        if(forwardMinibatch()){
            std::vector<cv::Rect> batch_bounding_boxes;
            std::vector<DetectedObject2d> batch_detections;
            for (int j = start; j < end; ++j) {
                batch_bounding_boxes.push_back(pixel_bounding_boxes[j]);
            }
            if (input_detections != nullptr && bounding_boxes == &defaultROI) {
                for (int j = start; j < end; ++j)
                    batch_detections.push_back((*input_detections)[j]);
            }
            postBatch(batch_bounding_boxes, batch_detections);
        }
    }
    if (bounding_boxes == &defaultROI){
        bounding_boxes = nullptr;
    }
    return true;
}
