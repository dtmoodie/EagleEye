#pragma once
#define COMPACT_GOOGLE_LOG_debug COMPACT_GOOGLE_LOG_DEBUG
#ifndef USE_CUDNN
#define USE_CUDNN
#endif
#include "Aquila/Detail/PluginExport.hpp"
#include "Aquila/Nodes/Node.h"
#include "Aquila/ObjectDetection.hpp"
#include "Aquila/rcc/external_includes/cv_calib3d.hpp"
#include "Aquila/ObjectDetection.hpp"
#include "CaffeNetHandler.hpp"
#include "MetaObject/MetaObject.hpp"
#include "MetaObject/Parameters/Types.hpp"
#include "RuntimeLinkLibrary.h"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#ifdef _MSC_VER // Windows
  #ifdef _DEBUG
    RUNTIME_COMPILER_LINKLIBRARY("libcaffe.lib")
  #else
    RUNTIME_COMPILER_LINKLIBRARY("opencv_calib3d" CV_VERSION_ ".lib")
  #endif
#else // Linux
  RUNTIME_COMPILER_LINKLIBRARY("-lcaffe")
#endif

namespace aq
{
    namespace Nodes
    {
        class CaffeBase : public Node
        {
        public:
            typedef std::vector<SyncedMemory> WrappedBlob_t;
            typedef std::map<std::string, WrappedBlob_t> BlobMap_t;
            MO_DERIVE(CaffeBase, Node)
                PROPERTY(boost::shared_ptr<caffe::Net<float>>, NN, boost::shared_ptr<caffe::Net<float>>())
                PROPERTY(BlobMap_t, wrapped_outputs, BlobMap_t())
                PROPERTY(BlobMap_t, wrapped_inputs, BlobMap_t())
                PARAM(cv::Scalar, channel_mean, cv::Scalar(104, 117, 123))
                PARAM(bool, bgr_swap, false)
                PARAM(bool, debug_dump, false)
                PROPERTY(bool, weightsLoaded, false)
                PARAM(mo::ReadFile, nn_model_file, mo::ReadFile())
                PARAM(mo::ReadFile, nn_weight_file, mo::ReadFile())
                PARAM(mo::ReadFile, label_file, mo::ReadFile())
                PARAM(mo::ReadFile, mean_file, mo::ReadFile())
                PARAM(float, pixel_scale, 0.00390625f)
                TOOLTIP(pixel_scale, "Scale factor to multiply the image by, after mean subtraction")
                PARAM(float, image_scale, 1.0)
                TOOLTIP(image_scale, "Scale factor for reducing image size to fit into network")
                PARAM(int, num_classifications, 5)
                OPTIONAL_INPUT(std::vector<cv::Rect2f>, bounding_boxes, nullptr)
                OPTIONAL_INPUT(std::vector<DetectedObject>, input_detections, nullptr)
                PARAM(int, detection_class, -1)
                TOOLTIP(detection_class, "When given an input_detections, decide which class of input detections should be used to select regions of interest for this classifier")
                INPUT(SyncedMemory, input, nullptr)
                OUTPUT(std::vector<std::string>, labels, {})
            MO_END
            virtual void NodeInit(bool firstInit);
            static std::vector<SyncedMemory> WrapBlob(caffe::Blob<float>& blob, bool bgr_swap = false);
            static std::vector<SyncedMemory> WrapBlob(caffe::Blob<double>& blob, bool bgr_swap = false);
        protected:
            bool InitNetwork();
            void WrapInput();
            bool CheckInput();
            void WrapOutput();
            void ReshapeInput(int num, int channels, int height, int width);
            std::vector<caffe::Blob<float>*> input_blobs;
            std::vector<caffe::Blob<float>*> output_blobs;
            /*
            0 = classifier
            1 = detector
            2 = fcn
            */
            enum NetworkType
            {
                Classifier_e = 0,
                Detector_e = 1,
                FCN_e = 1 << 1
            };
            NetworkType _network_type;
        };

        class CaffeImageClassifier : public CaffeBase
        {
        public:
            MO_DERIVE(CaffeImageClassifier, CaffeBase)
                PARAM(int, num_classifications, 5)
            MO_END
            void       PostSerializeInit();
        protected:
            bool ProcessImpl();
            std::vector<rcc::shared_ptr<Caffe::NetHandler>> net_handlers;
        };
        
        class CaffeDetector: public CaffeBase
        {
        public:

        protected:

        };
    }
}
