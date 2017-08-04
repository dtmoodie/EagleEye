#include "INeuralNet.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#ifndef NDEBUG
#include <opencv2/imgproc.hpp>
#endif

void aq::nodes::INeuralNet::on_weight_file_modified(mo::IParam*, mo::Context*, mo::OptionalTime_t, size_t,
                                                    const std::shared_ptr<mo::ICoordinateSystem>&, mo::UpdateFlags){
    initNetwork();
}

void aq::nodes::INeuralNet::preBatch(int batch_size) {
    (void)batch_size;
}

void aq::nodes::INeuralNet::postBatch() {
}

bool aq::nodes::INeuralNet::processImpl() {
    if (initNetwork()) {
        return forwardAll();
    }
    return false;
}

bool aq::nodes::INeuralNet::forwardAll() {
    std::vector<cv::Rect2f> defaultROI;
    auto                    input_image_shape = input->getShape();
    defaultROI.push_back(cv::Rect2f(0, 0, 1.0, 1.0));
    if (bounding_boxes == nullptr) {
        bounding_boxes = &defaultROI;
    }

    if (input_detections != nullptr && bounding_boxes == &defaultROI) {
        defaultROI.clear();
        for (const auto& itr : *input_detections) {
            defaultROI.emplace_back(
                itr.bounding_box.x / input_image_shape[2],
                itr.bounding_box.y / input_image_shape[1],
                itr.bounding_box.width / input_image_shape[2],
                itr.bounding_box.height / input_image_shape[1]);
        }
        if (defaultROI.size() == 0) {
            bounding_boxes = nullptr;
            preBatch(0);
            postBatch();
            return false;
        }

    }

#ifndef NDEBUG
    cv::Mat dbg_img;
    input->clone(dbg_img, stream());
    stream().waitForCompletion();
#endif

    std::vector<cv::Rect> pixel_bounding_boxes;
    for (size_t i = 0; i < bounding_boxes->size(); ++i) {
        cv::Rect bb;
        bb.x      = static_cast<int>((*bounding_boxes)[i].x * input_image_shape[2]);
        bb.y      = static_cast<int>((*bounding_boxes)[i].y * input_image_shape[1]);
        bb.width  = static_cast<int>((*bounding_boxes)[i].width * input_image_shape[2]);
        bb.height = static_cast<int>((*bounding_boxes)[i].height * input_image_shape[1]);
        if (bb.x + bb.width >= input_image_shape[2]) {
            bb.x -= input_image_shape[2] - bb.width;
        }
        if (bb.y + bb.height >= input_image_shape[1]) {
            bb.y -= input_image_shape[1] - bb.height;
        }
        bb.x = std::max(0, bb.x);
        bb.y = std::max(0, bb.y);
        pixel_bounding_boxes.push_back(bb);
#ifndef NDEBUG
        cv::rectangle(dbg_img, bb, cv::Scalar(0,255,0), 2);
#endif
    }

    cv::Scalar_<unsigned int> network_input_shape = getNetworkShape();
    if (image_scale > 0) {
        reshapeNetwork(static_cast<unsigned int>(bounding_boxes->size()),
            static_cast<unsigned int>(input_image_shape[3]),
            static_cast<unsigned int>(input_image_shape[1] * image_scale),
            static_cast<unsigned int>(input_image_shape[2] * image_scale));
    }
    if (pixel_bounding_boxes.size() != network_input_shape[0] && input_detections == nullptr) {
        reshapeNetwork(static_cast<unsigned int>(bounding_boxes->size()),
            network_input_shape[1],
            network_input_shape[2],
            network_input_shape[3]);
    }

    cv::cuda::GpuMat float_image;
    if (input->getDepth() != CV_32F) {
        input->getGpuMat(stream()).convertTo(float_image, CV_32F, stream());
    } else {
        input->clone(float_image, stream());
    }
    if (channel_mean[0] != 0.0 || channel_mean[1] != 0.0 || channel_mean[2] != 0.0)
        cv::cuda::subtract(float_image, channel_mean, float_image, cv::noArray(), -1, stream());
    if (pixel_scale != 1.0f) {
        cv::cuda::multiply(float_image, cv::Scalar::all(static_cast<double>(pixel_scale)), float_image, 1.0, -1, stream());
    }

    preBatch(static_cast<int>(pixel_bounding_boxes.size()));
    cv::cuda::GpuMat resized;
    auto             net_input = getNetImageInput();
    MO_ASSERT(net_input.size());
    MO_ASSERT(net_input[0].size() == static_cast<size_t>(input->getChannels()));
    cv::Size net_input_size = net_input[0][0].size();
    for (size_t i = 0; i < pixel_bounding_boxes.size();) { // for each roi
        size_t start = i, end = 0;
        for (size_t j = 0; j < net_input.size() && i < pixel_bounding_boxes.size(); ++j, ++i) { // for each image in the mini batch
            if (pixel_bounding_boxes[i].size() != net_input_size) {
                cv::cuda::resize(float_image(pixel_bounding_boxes[i]), resized, net_input_size, 0, 0, cv::INTER_LINEAR, stream());
            } else {
                resized = float_image(pixel_bounding_boxes[i]);
            }
            cv::cuda::split(resized, net_input[j], stream());
            end = start + j + 1;
        }
        if (forwardMinibatch()) {
            std::vector<cv::Rect>         batch_bounding_boxes;
            std::vector<DetectedObject2d> batch_detections;
            for (size_t j = start; j < end; ++j) {
                batch_bounding_boxes.push_back(pixel_bounding_boxes[j]);
            }
            if (input_detections != nullptr && bounding_boxes == &defaultROI) {
                for (size_t j = start; j < end; ++j)
                    batch_detections.push_back((*input_detections)[j]);
            }
            postMiniBatch(batch_bounding_boxes, batch_detections);
        }
    }
    postBatch();
    if (bounding_boxes == &defaultROI) {
        bounding_boxes = nullptr;
    }
    return true;
}
