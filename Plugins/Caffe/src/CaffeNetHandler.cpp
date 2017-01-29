#include <MetaObject/Parameters/IO/CerealPolicy.hpp>
#include <MetaObject/Parameters/IO/CerealMemory.hpp>
#include "CaffeNetHandler.hpp"
#include "EagleLib/IO/JsonArchive.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

INSTANTIATE_META_PARAMETER(std::vector<rcc::shared_ptr<EagleLib::Caffe::NetHandler>>)

std::vector<boost::shared_ptr<caffe::Layer<float>>>
EagleLib::Caffe::NetHandler::GetOutputLayers(const caffe::Net<float>& net)
{
    const std::vector<caffe::Blob<float>*>& blobs = net.output_blobs();
    const std::vector<int>& out_idx = net.output_blob_indices();
    const std::vector<std::string>& names = net.blob_names();
    auto layer_names = net.layer_names();
    auto layers = net.layers();
    std::vector<boost::shared_ptr<caffe::Layer<float>>> output;
    for(int i = 0; i < layer_names.size(); ++i)
    {
        std::vector<int> top_ids = net.top_ids(i);
        for(auto id : top_ids)
        {
            if(std::find(out_idx.begin(), out_idx.end(), id) != out_idx.end())
            {
                output.push_back(layers[i]);
            }
        }
    }
    return output;
}

void EagleLib::Caffe::NetHandler::SetOutputBlob(const caffe::Net<float> &net, int output_blob_index)
{
    output_blob_name = net.blob_names()[output_blob_index];
}
