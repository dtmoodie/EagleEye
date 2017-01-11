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
          virtual std::vector<int> CanHandleNetwork(const caffe::Net<float>& net) const = 0;

        };
        class NetHandler:
              public TInterface<ctcrc32("EagleLib::Caffe::NetHandler"), Algorithm>
        {
        public:
            typedef std::vector<SyncedMemory> WrappedBlob_t;
            typedef std::map<std::string, WrappedBlob_t> BlobMap_t;
            typedef NetHandlerInfo InterfaceInfo;
            typedef NetHandler Interface;
            MO_BEGIN(NetHandler)
            MO_END
            virtual void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes) = 0;
        protected:
            bool ProcessImpl() { return true;}
        };
    }
}
