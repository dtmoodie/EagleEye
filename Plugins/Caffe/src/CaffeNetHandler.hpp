#pragma once
#include <MetaObject/MetaObject.hpp>
#include <MetaObject/IMetaObjectInfo.hpp>
#include <EagleLib/SyncedMemory.h>
#include <EagleLib/Algorithm.h>
#include <caffe/net.hpp>
namespace EagleLib
{
    namespace Caffe
    {
        class NetHandlerInfo: public mo::IMetaObjectInfo
        {
        public:
            // Return a map of blob index, priority of this handler
            virtual std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net) const = 0;
        };
        class NetHandler:
              public TInterface<ctcrc32("EagleLib::Caffe::NetHandler"), Algorithm>
        {
        public:
            static std::vector<boost::shared_ptr<caffe::Layer<float>>> GetOutputLayers(const caffe::Net<float>& net);
            typedef std::vector<SyncedMemory> WrappedBlob_t;
            typedef std::map<std::string, WrappedBlob_t> BlobMap_t;
            typedef NetHandlerInfo InterfaceInfo;
            typedef NetHandler Interface;
            MO_BEGIN(NetHandler)
                PARAM(std::string, output_blob_name, "score")
            MO_END
            virtual void SetOutputBlob(const caffe::Net<float>& net, int output_blob_index);
            virtual void StartBatch(){}
            virtual void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size) = 0;
            virtual void EndBatch(long long timestamp){}
            void SetLabels(std::vector<std::string>* labels){this->labels = labels;}
        protected:
            bool ProcessImpl() { return true;}
            std::vector<std::string>* labels = nullptr;
        };
    }
}
