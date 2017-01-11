#include "CaffeFCNHandler.hpp"
#include "helpers.hpp"
#include "CaffeNetHandlerInfo.hpp"
using namespace EagleLib::Caffe;

std::vector<int> FCNHandler::CanHandleNetwork(const caffe::Net<float>& net)
{
    const std::vector<caffe::Blob<float>*>& blobs = net.output_blobs();
    std::vector<int> output;
    for(int i = 0; i < blobs.size(); ++i)
    {
        const std::vector<int>& shape = blobs[i]->shape();
        if(shape.size() == 4)
        {
            if(shape[2] > 1 && shape[3] > 1)
                output.push_back(i);
        }
    }
    return output;
}

void FCNHandler::HandleOutput(const caffe::Net<float>& net, long long timestamp,  const std::vector<cv::Rect>& bounding_boxes)
{
    auto blob = net.blob_by_name(output_blob_name);
    cv::Mat label, confidence;
    EagleLib::Caffe::MaxSegmentation(blob.get(), label, confidence);
    label.setTo(0, confidence < min_confidence);
    label_param.UpdateData(label, timestamp, _ctx);
    confidence_param.UpdateData(confidence, timestamp, _ctx);
}


MO_REGISTER_CLASS(FCNHandler)
