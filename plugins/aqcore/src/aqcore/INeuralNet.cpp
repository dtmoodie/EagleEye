#include <MetaObject/core/metaobject_config.hpp>

#include "INeuralNet.hpp"
#include <MetaObject/logging/profiling.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/imgproc.hpp>

#include "OpenCVCudaNode.hpp"
namespace aqcore
{
    rcc::shared_ptr<INeuralNet> INeuralNet::create(const std::string& model, const std::string& weight)
    {
        std::shared_ptr<mo::MetaObjectFactory> factory = mo::MetaObjectFactory::instance();
        std::vector<IObjectConstructor*> ctrs = factory->getConstructors(getHash());
        for (IObjectConstructor* ctr : ctrs)
        {
            const INeuralNetInfo* info = dynamic_cast<const INeuralNetInfo*>(ctr->GetObjectInfo());
            if (info)
            {
                if (info->canLoad(model, weight))
                {
                    auto obj = ctr->Construct();
                    if (obj)
                    {
                        rcc::shared_ptr<INeuralNet> tobj(obj);
                        tobj->model_file = model;
                        tobj->weight_file = weight;
                        return tobj;
                    }
                }
            }
        }
        return {};
    }

    void INeuralNet::on_weight_file_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream*)
    {
        initNetwork();
    }

    void INeuralNet::preBatch(int batch_size) { (void)batch_size; }

    void INeuralNet::postBatch() {}

    bool INeuralNet::processImpl(mo::IAsyncStream& stream)
    {
        mo::IDeviceStream* dev_stream = stream.getDeviceStream();
        if (dev_stream)
        {
            return this->processImpl(*dev_stream);
        }
        return false;
    }

    bool INeuralNet::processImpl(mo::IDeviceStream& stream)
    {
        if (initNetwork())
        {
            return forwardAll(stream);
        }
        return false;
    }

    void INeuralNet::setStream(const mo::IAsyncStreamPtr_t& stream)
    {
        m_cv_stream = getCVStream(stream);
        IClassifier::setStream(stream);
    }

    std::vector<cv::Rect> INeuralNet::getRegions() const
    {
        mo::vector<cv::Rect2f> default_roi;

        if (regions_of_interest)
        {
            default_roi = *regions_of_interest;
        }
        else
        {
            default_roi.push_back(cv::Rect2f(0, 0, 1.0, 1.0));
        }

        auto input_image_shape = input->size();
        for (size_t i = 0; i < default_roi.size(); ++i)
        {
            aq::boundingBoxToPixels(default_roi[i], input_image_shape);
        }

        if (input_detections != nullptr)
        {
            default_roi.clear();
            mt::Tensor<const cv::Rect2f, 1> bbs = input_detections->getComponent<aq::detection::BoundingBox2d>();
            const uint32_t num_detections = input_detections->getNumEntities();
            for (uint32_t i = 0; i < num_detections; ++i)
            {
                cv::Rect2f bb = bbs[i];
                default_roi.push_back(bb);
            }
        }
        const uint32_t image_width = input_image_shape(1);
        const uint32_t image_height = input_image_shape(0);
        std::vector<cv::Rect> pixel_bounding_boxes;
        for (size_t i = 0; i < default_roi.size(); ++i)
        {
            cv::Rect bb;
            bb.x = static_cast<int>(default_roi[i].x);
            bb.y = static_cast<int>(default_roi[i].y);
            bb.width = static_cast<int>(default_roi[i].width);
            bb.height = static_cast<int>(default_roi[i].height);
            if (bb.x + bb.width >= image_width)
            {
                bb.x -= image_width - bb.width;
            }
            if (bb.y + bb.height >= image_height)
            {
                bb.y -= image_height - bb.height;
            }
            bb.x = std::max(0, bb.x);
            bb.y = std::max(0, bb.y);
            pixel_bounding_boxes.push_back(bb);
        }
        return pixel_bounding_boxes;
    }

    bool INeuralNet::forwardAll(mo::IDeviceStream& stream)
    {
        cv::Scalar_<unsigned int> network_input_shape = getNetworkShape();
        std::vector<cv::Rect> pixel_bounding_boxes = getRegions();
        if (pixel_bounding_boxes.size() == 0)
        {
            preBatch(0);
            postBatch();
            return false;
        }
        const auto input_image_shape = input->shape();
        const uint32_t batch_size = pixel_bounding_boxes.size();
        const uint32_t height = input_image_shape(0);
        const uint32_t width = input_image_shape(1);
        const uint32_t channels = input_image_shape(2);

        if (image_scale > 0)
        {
            this->getLogger().trace("Reshaping network");
            if (reshapeNetwork(batch_size, channels, height * image_scale, width * image_scale))
            {
                this->getLogger().trace("Reshaping complete");
            }
            else
            {
                this->getLogger().trace("unable to reshape network");
            }
        }
        // Request a larger batch size
        if (pixel_bounding_boxes.size() != network_input_shape[0] && input_detections == nullptr)
        {
            this->getLogger().trace("Resizing batch size");

            reshapeNetwork(static_cast<unsigned int>(pixel_bounding_boxes.size()),
                           network_input_shape[1],
                           network_input_shape[2],
                           network_input_shape[3]);

            this->getLogger().trace("Batch resizing complete");
        }
        this->getLogger().trace("Preprocessing begin");

        cv::cuda::GpuMat float_image;
        cv::cuda::Stream& cvstream = *m_cv_stream;
        cv::cuda::GpuMat input = this->input->getGpuMat(&stream);

        {
            mo::ScopedProfile preprocessing("INeuralNet::preprocessing");
            if (input.depth() != CV_32F)
            {
                input.convertTo(float_image, CV_32F, cvstream);
            }
            else
            {
                float_image = input;
            }
            if (channel_mean[0] != 0.0 || channel_mean[1] != 0.0 || channel_mean[2] != 0.0)
            {
                cv::cuda::subtract(float_image, channel_mean, float_image, cv::noArray(), -1, cvstream);
            }
            if (pixel_scale != 1.0f)
            {
                const cv::Scalar scale = cv::Scalar::all(static_cast<double>(pixel_scale));
                cv::cuda::multiply(float_image, scale, float_image, 1.0, -1, cvstream);
            }
            this->getLogger().trace("Preprocessing complete");
        }

        preBatch(static_cast<int>(pixel_bounding_boxes.size()));

        cv::cuda::GpuMat resized;
        auto net_input = getNetImageInput();
        MO_ASSERT(net_input.size());
        MO_ASSERT(net_input[0].size() == static_cast<size_t>(input.channels()));
        cv::Size net_input_size = net_input[0][0].size();

        for (size_t i = 0; i < pixel_bounding_boxes.size();)
        {
            // for each roi
            size_t start = i, end = 0;
            for (size_t j = 0; j < net_input.size() && i < pixel_bounding_boxes.size(); ++j, ++i)
            { // for each image in the mini batch
                if (pixel_bounding_boxes[i].size() != net_input_size)
                {
                    cv::cuda::resize(float_image(pixel_bounding_boxes[i]),
                                     resized,
                                     net_input_size,
                                     0,
                                     0,
                                     cv::INTER_LINEAR,
                                     cvstream);
                }
                else
                {
                    resized = float_image(pixel_bounding_boxes[i]);
                }
                cv::cuda::split(resized, net_input[j], cvstream);
                end = start + j + 1;
            }

            this->getLogger().trace("forward mini batch");
            if (forwardMinibatch(stream))
            {
                this->getLogger().trace("Forward mini batch complete");
                std::vector<cv::Rect> batch_bounding_boxes;
                for (size_t j = start; j < end; ++j)
                {
                    batch_bounding_boxes.push_back(pixel_bounding_boxes[j]);
                }

                postMiniBatch(stream, batch_bounding_boxes, input_detections);

                this->getLogger().trace("Post mini batch complete");
            }
        }
        postBatch();
        this->getLogger().trace("Post batch complete");

        return true;
    }

} // namespace aqcore
