#pragma once

#include "EagleLib/Detail/PluginExport.hpp"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/ObjectDetection.hpp"
#include "MetaObject/MetaObject.hpp"
#include "MetaObject/Parameters/Types.hpp"
#include "RuntimeLinkLibrary.h"
SETUP_PROJECT_DEF
#include "caffe/blob.hpp"
#include "caffe/net.hpp"

namespace EagleLib
{
    namespace Nodes
    {
        class CaffeBase : public Node
        {
        public:
            typedef std::vector<SyncedMemory> WrappedBlob_t;
            typedef std::map<std::string, WrappedBlob_t> BlobMap_t;
            MO_DERIVE(CaffeBase, Node)
                PROPERTY(boost::shared_ptr<caffe::Net<float>>, NN, boost::shared_ptr<caffe::Net<float>>());
                PROPERTY(boost::shared_ptr< std::vector< std::string > >, labels, boost::shared_ptr< std::vector< std::string > >());
                PROPERTY(BlobMap_t, wrapped_outputs, BlobMap_t());
                PROPERTY(BlobMap_t, wrapped_inputs, BlobMap_t());
                PROPERTY(cv::Scalar, channel_mean, cv::Scalar());
                PROPERTY(bool, weightsLoaded, false);
                PARAM(mo::ReadFile, nn_model_file, mo::ReadFile());
                PARAM(mo::ReadFile, nn_weight_file, mo::ReadFile());
                PARAM(mo::ReadFile, label_file, mo::ReadFile());
                PARAM(mo::ReadFile, mean_file, mo::ReadFile());
                PARAM(float, scale, 0.00390625f);
                TOOLTIP(scale, "Scale factor to multiply the image by, after mean subtraction");
                PARAM(int, num_classifications, 5);
                OPTIONAL_INPUT(std::vector<cv::Rect>, bounding_boxes, nullptr);
                INPUT(SyncedMemory, input, nullptr);
            MO_END;
            virtual void NodeInit(bool firstInit);
            static std::vector<SyncedMemory> WrapBlob(caffe::Blob<float>& blob);
            static std::vector<SyncedMemory> WrapBlob(caffe::Blob<double>& blob);
        protected:
            bool InitNetwork();
            void WrapInput();
            void WrapOutput();
            std::vector<caffe::Blob<float>*> input_blobs;
            std::vector<caffe::Blob<float>*> output_blobs;
        };

        class CaffeImageClassifier : public CaffeBase
        {
        public:
            MO_DERIVE(CaffeImageClassifier, CaffeBase)
                PARAM(int, num_classifications, 5);
                OUTPUT(std::vector<DetectedObject>, detections, std::vector<DetectedObject>());
            MO_END;
            
        protected:
            bool ProcessImpl();
        };
        
        class CaffeDetector: public CaffeBase
        {
        public:

        protected:

        };
    }


}