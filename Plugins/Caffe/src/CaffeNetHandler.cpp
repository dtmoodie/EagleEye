#include <MetaObject/serialization/CerealPolicy.hpp>
#include <MetaObject/serialization/CerealMemory.hpp>
#include "CaffeNetHandler.hpp"
#include "Aquila/serialization/cereal/JsonArchive.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include "MetaObject/params/detail/MetaParamImpl.hpp"
INSTANTIATE_META_PARAM(std::vector<rcc::shared_ptr<aq::Caffe::NetHandler>>);

std::vector<boost::shared_ptr<caffe::Layer<float>>>
aq::Caffe::NetHandler::getOutputLayers(const caffe::Net<float>& net){
    const std::vector<int>& out_idx = net.output_blob_indices();
    auto layer_names = net.layer_names();
    auto layers = net.layers();
    std::vector<boost::shared_ptr<caffe::Layer<float>>> output;
    for(int i = 0; i < layer_names.size(); ++i){
        std::vector<int> top_ids = net.top_ids(i);
        for(auto id : top_ids){
            if(std::find(out_idx.begin(), out_idx.end(), id) != out_idx.end()){
                output.push_back(layers[i]);
            }
        }
    }
    return output;
}

void aq::Caffe::NetHandler::setOutputBlob(const caffe::Net<float> &net, int output_blob_index){
    output_blob_name = net.blob_names()[output_blob_index];
}
