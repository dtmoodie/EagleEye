#pragma once
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/IMetaObjectInfo.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/core/Algorithm.hpp>
#include "Aquila/types/ObjectDetection.hpp"
#include <caffe/net.hpp>

namespace aq
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
              public TInterface<NetHandler, Algorithm>
        {
        public:
            static std::vector<boost::shared_ptr<caffe::Layer<float>>> getOutputLayers(const caffe::Net<float>& net);
            typedef std::vector<SyncedMemory> WrappedBlob_t;
            typedef std::map<std::string, WrappedBlob_t> BlobMap_t;
            typedef NetHandlerInfo InterfaceInfo;
            typedef NetHandler Interface;
            MO_BEGIN(NetHandler)
                PARAM(std::string, output_blob_name, "score")
            MO_END
            virtual void setOutputBlob(const caffe::Net<float>& net, int output_blob_index);
            virtual void startBatch(){}
            virtual void handleOutput(const caffe::Net<float>& net, const std::vector<cv::Rect>& bounding_boxes,
                                      mo::ITParam<aq::SyncedMemory>& input_param, const std::vector<DetectedObject2d>& objs) = 0;
            virtual void endBatch(boost::optional<mo::Time_t> timestamp){}
            void setLabels(std::vector<std::string>* labels){this->labels = labels;}
        protected:
            bool processImpl() { return true;}
            std::vector<std::string>* labels = nullptr;
        };
    }
}
