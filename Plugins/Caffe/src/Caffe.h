#pragma once
#define COMPACT_GOOGLE_LOG_debug COMPACT_GOOGLE_LOG_DEBUG
#ifndef USE_CUDNN
#define USE_CUDNN
#endif

#include "Aquila/rcc/external_includes/Caffe_link_libs.hpp"
#include "CaffeExport.hpp"
#include "CaffeNetHandler.hpp"

#include <INeuralNet.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/rcc/external_includes/cv_calib3d.hpp>
#include <Aquila/types/ObjectDetection.hpp>

#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/Types.hpp>
#include <RuntimeObjectSystem/RuntimeLinkLibrary.h>

#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/net.hpp>

extern "C" Caffe_EXPORT void InitModule();

namespace aq {
namespace nodes {
    class Caffe_EXPORT CaffeBase : public INeuralNet {
    public:
        typedef std::vector<SyncedMemory> WrappedBlob_t;
        typedef std::map<std::string, WrappedBlob_t> BlobMap_t;
        MO_DERIVE(CaffeBase, INeuralNet)
        PROPERTY(boost::shared_ptr<caffe::Net<float> >, NN, boost::shared_ptr<caffe::Net<float> >())
        PROPERTY(BlobMap_t, wrapped_outputs, BlobMap_t())
        PROPERTY(BlobMap_t, wrapped_inputs, BlobMap_t())
        PROPERTY(bool, weightsLoaded, false)

        MO_END
        virtual void nodeInit(bool firstInit);
        static std::vector<SyncedMemory> WrapBlob(caffe::Blob<float>& blob, bool bgr_swap = false);
        static std::vector<SyncedMemory> WrapBlob(caffe::Blob<double>& blob, bool bgr_swap = false);

    protected:
        virtual bool initNetwork();
        virtual bool reshapeNetwork(unsigned int num, unsigned int channels, unsigned int height, unsigned int width);
        virtual cv::Scalar_<unsigned int>                   getNetworkShape() const;
        virtual std::vector<std::vector<cv::cuda::GpuMat> > getNetImageInput(int requested_batch_size = 1);
        virtual void preBatch(int batch_size);
        virtual void postMiniBatch(const std::vector<cv::Rect>& batch_bb = std::vector<cv::Rect>(),
            const std::vector<DetectedObject2d>&                dets     = std::vector<DetectedObject2d>());
        virtual void postBatch();
        virtual bool forwardMinibatch();

        void WrapInput();
        bool CheckInput();
        void WrapOutput();

        std::vector<caffe::Blob<float>*>                 input_blobs;
        std::vector<caffe::Blob<float>*>                 output_blobs;
        std::vector<rcc::shared_ptr<Caffe::NetHandler> > net_handlers;
    };

    class Caffe_EXPORT CaffeImageClassifier : public CaffeBase {
    public:
        MO_DERIVE(CaffeImageClassifier, CaffeBase)
        PARAM(int, num_classifications, 5)
        MO_END
        void postSerializeInit();
    };

    class Caffe_EXPORT CaffeDetector : public CaffeBase {
    public:
    protected:
    };
}
}
