#include "CaffeFCNHandler.hpp"
#include "helpers.hpp"
#include "CaffeNetHandlerInfo.hpp"
#include "Caffe.h"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace EagleLib::Caffe;

std::map<int, int> FCNHandler::CanHandleNetwork(const caffe::Net<float>& net)
{
    const std::vector<caffe::Blob<float>*>& blobs = net.output_blobs();
    const std::vector<int>& out_idx = net.output_blob_indices();
    const std::vector<std::string>& names = net.blob_names();
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
                if(type == "Softmax" || type == "Convolution" || type == "Crop")
                {
                    output[id] = 10;
                }
            }
        }
    }
    return output;
}

void FCNHandler::HandleOutput(const caffe::Net<float>& net, long long timestamp,  const std::vector<cv::Rect>& bounding_boxes)
{
    auto blob = net.blob_by_name(output_blob_name);
    if(!blob)
        return;
    cv::Mat label, confidence;
    EagleLib::Caffe::MaxSegmentation(blob.get(), label, confidence);
    label.setTo(0, confidence < min_confidence);
    label_param.UpdateData(label, timestamp, _ctx);
    confidence_param.UpdateData(confidence, timestamp, _ctx);
}

MO_REGISTER_CLASS(FCNHandler)

std::map<int, int> FCNSingleClassHandler::CanHandleNetwork(const caffe::Net<float>& net)
{
    const std::vector<caffe::Blob<float>*>& blobs = net.output_blobs();
    const std::vector<int>& out_idx = net.output_blob_indices();
    const std::vector<std::string>& names = net.blob_names();
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
                if(type == "Softmax" || type == "Convolution" || type == "Crop")
                {
                    output[id] = 10;
                }
            }
        }
    }
    return output;
}

void FCNSingleClassHandler::HandleOutput(const caffe::Net<float>& net, long long timestamp,  const std::vector<cv::Rect>& bounding_boxes)
{
    auto blob = net.blob_by_name(output_blob_name);
    if(blob)
    {
        blob->gpu_data();
        blob->cpu_data();
        auto wrapped = EagleLib::Nodes::CaffeBase::WrapBlob(*blob);
        if(class_index < blob->channels() && blob->num())
        {
            const cv::cuda::GpuMat& confidence = wrapped[0].GetGpuMat(_ctx->GetStream(), class_index);
            cv::Mat dbg = wrapped[0].GetMat(_ctx->GetStream(), class_index);
            cv::cuda::GpuMat confidence_out;
            cv::cuda::threshold(confidence, confidence_out, min_confidence, 255, cv::THRESH_BINARY, _ctx->GetStream());
            cv::cuda::GpuMat mask_out;
            confidence_out.convertTo(mask_out, CV_8UC1, _ctx->GetStream());
            label_param.UpdateData(mask_out, timestamp, _ctx);
            confidence_param.UpdateData(confidence, timestamp, _ctx);
        }
    }
}

MO_REGISTER_CLASS(FCNSingleClassHandler)

std::map<int, int> LaneHandler::CanHandleNetwork(const caffe::Net<float>& net)
{
    const std::vector<caffe::Blob<float>*>& blobs = net.output_blobs();
    const std::vector<int>& out_idx = net.output_blob_indices();
    const std::vector<std::string>& names = net.blob_names();
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
                if((type == "Softmax" || type == "Convolution" || type == "Crop") &&
                        (names[id] == "line_eq" || names[id] == "line_conf"))
                {
                    output[id] = 11;
                }
            }
        }
    }
    return output;
}
template<class T> T rail(T val, T min, T max)
{
    val = std::max(min, val);
    val = std::min(max, val);
    return val;
}

void LaneHandler::HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes)
{
    auto line_offset_blob = net.blob_by_name("line_eq");
    auto line_conf_blob = net.blob_by_name("softmax");
    auto line_conf_raw = net.blob_by_name("line_conf");
    auto input_blob = net.blob_by_name("data");
    if(line_offset_blob && line_conf_blob)
    {
        auto wrapped_conf = EagleLib::Nodes::CaffeBase::WrapBlob(*line_conf_raw);
        auto wrapped_eq = EagleLib::Nodes::CaffeBase::WrapBlob(*line_offset_blob);
        const auto& h_conf = wrapped_conf[0].GetMatVec(_ctx->GetStream());
        cv::Mat_<uchar> output_class(size_y, size_x);
        cv::Mat_<float> output_conf(size_y, size_x);
        for(int y = 0; y < size_y; ++y)
        {
            for(int x = 0; x < size_x; ++x)
            {
                int y_ = y + pad_y;
                int x_ = x + pad_x;
                float max_val = 0;
                uchar max_class = 0;
                for(int c = 0; c < h_conf.size(); ++c)
                {
                    float val = h_conf[c].at<float>(y_, x_);
                    if(val > min_confidence && val > max_val)
                    {
                        max_class = c;
                        max_val = val;
                    }
                }
                output_class(y,x) = max_class;
                output_conf(y,x) = max_val;
            }
        }

        int input_height = input_blob->height();
        int input_width = input_blob->width();
        cv::Mat_<float> h_eq = wrapped_eq[0].GetMat(_ctx->GetStream());
        h_eq = h_eq(cv::Rect(pad_x, pad_y, size_x, size_y));
        cv::Mat_<float> line_eq(size_y, size_x);
        line_eq.setTo(0);
        for(int y = 0; y < size_y; ++y)
        {
            for(int x = 0; x < size_x; ++x)
            {
                if(output_conf(y,x) > min_confidence)
                {
                    int search_x = x+1;
                    // Find a grouping of adjacent neurons that
                    // share a high confidence output
                    while(search_x < size_x && output_conf(y, search_x) > min_confidence)
                    {
                        ++search_x;
                    }
                    float offset = 0.0f;
                    float count = 0.0f;
                    int rf_center_y = (float(y) / float(size_y)) * input_height;
                    for(int i = x; i < search_x; ++i)
                    {
                        // Calculate center of receptive field wrt input to network
                        int rf_center_x = (float(i) / float(size_x)) * input_width;

                        // Offset is wrt the receptive field of the neuron
                        float Ox = h_eq(y, i);
                        offset += Ox + i - x;
                        count += 1.0f;
                    }
                    offset /= count;
                    x = rail<int>(x + offset, 0, size_x - 1);
                    line_eq(y,x) = 1;
                    x = search_x + 1;
                }
            }
        }

        lane_param.UpdateData(output_class, timestamp, _ctx);
        confidence_param.UpdateData(output_conf, timestamp, _ctx);
    }
}
MO_REGISTER_CLASS(LaneHandler)
