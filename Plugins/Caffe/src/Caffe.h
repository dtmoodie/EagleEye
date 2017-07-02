#pragma once
#define COMPACT_GOOGLE_LOG_debug COMPACT_GOOGLE_LOG_DEBUG
#ifndef USE_CUDNN
#define USE_CUDNN
#endif
#include "Aquila/nodes/Node.hpp"
#include "Aquila/types/ObjectDetection.hpp"
#include "Aquila/rcc/external_includes/cv_calib3d.hpp"
#include "Aquila/types/ObjectDetection.hpp"
#include "CaffeNetHandler.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/params/Types.hpp"
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include "CaffeExport.hpp"
#include "Aquila/rcc/external_includes/Caffe_link_libs.hpp"
#ifdef _MSC_VER // Windows
  #ifdef _DEBUG
    RUNTIME_COMPILER_LINKLIBRARY("libcaffe.lib")
  #else
    RUNTIME_COMPILER_LINKLIBRARY("opencv_calib3d" CV_VERSION_ ".lib")
  #endif
#else // Linux
//  RUNTIME_COMPILER_LINKLIBRARY("-lcaffe")
#endif
extern "C" Caffe_EXPORT void InitModule();
namespace aq
{
    namespace nodes
    {
        class Caffe_EXPORT CaffeBase : public Node
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
                INPUT(SyncedMemory, input, nullptr)
                OUTPUT(std::vector<std::string>, labels, {})
                APPEND_FLAGS(labels, mo::Unstamped_e)
            MO_END
            virtual void nodeInit(bool firstInit);
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
        };

        class Caffe_EXPORT CaffeImageClassifier : public CaffeBase
        {
        public:
            MO_DERIVE(CaffeImageClassifier, CaffeBase)
                PARAM(int, num_classifications, 5)
            MO_END
            void postSerializeInit();
        protected:
            bool processImpl();
            std::vector<rcc::shared_ptr<Caffe::NetHandler>> net_handlers;
        };

        class Caffe_EXPORT CaffeDetector: public CaffeBase
        {
        public:

        protected:

        };
    }
}
