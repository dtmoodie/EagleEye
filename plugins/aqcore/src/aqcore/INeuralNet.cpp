#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "INeuralNet.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#ifndef NDEBUG
#include <opencv2/imgproc.hpp>
#endif

namespace aq
{
    namespace nodes
    {
        rcc::shared_ptr<INeuralNet> INeuralNet::create(const std::string& model, const std::string& weight)
        {
            auto ctrs = mo::MetaObjectFactory::instance().getConstructors(getHash());
            for (auto ctr : ctrs)
            {
                INeuralNetInfo* info = dynamic_cast<INeuralNetInfo*>(ctr->GetObjectInfo());
                if (info)
                {
                    if (info->canLoad(model, weight))
                    {
                        auto obj = ctr->Construct();
                        if (obj)
                        {
                            auto tobj = dynamic_cast<INeuralNet*>(obj);
                            if (!tobj)
                            {
                                delete obj;
                                return {};
                            }
                            tobj->model_file = model;
                            tobj->weight_file = weight;
                            return {tobj};
                        }
                    }
                }
            }
            return {};
        }

        void aq::nodes::INeuralNet::on_weight_file_modified(mo::IParam*,
                                                            mo::Context*,
                                                            mo::OptionalTime_t,
                                                            size_t,
                                                            const std::shared_ptr<mo::ICoordinateSystem>&,
                                                            mo::UpdateFlags)
        {
            initNetwork();
        }

        void aq::nodes::INeuralNet::preBatch(int batch_size) { (void)batch_size; }

        void aq::nodes::INeuralNet::postBatch() {}

        bool aq::nodes::INeuralNet::processImpl()
        {
            if (initNetwork())
            {
                return forwardAll();
            }
            return false;
        }

        std::vector<cv::Rect> aq::nodes::INeuralNet::getRegions() const
        {
            std::vector<cv::Rect2f> defaultROI;

            if (bounding_boxes)
            {
                defaultROI = *bounding_boxes;
            }
            else
            {
                defaultROI.push_back(cv::Rect2f(0, 0, 1.0, 1.0));
            }

            auto input_image_shape = input->getShape();
            if (input_detections != nullptr)
            {
                defaultROI.clear();
                for (const auto& itr : *input_detections)
                {
                    defaultROI.emplace_back(itr.bounding_box.x / input_image_shape[2],
                                            itr.bounding_box.y / input_image_shape[1],
                                            itr.bounding_box.width / input_image_shape[2],
                                            itr.bounding_box.height / input_image_shape[1]);
                }
            }
            std::vector<cv::Rect> pixel_bounding_boxes;
            for (size_t i = 0; i < defaultROI.size(); ++i)
            {
                cv::Rect bb;
                bb.x = static_cast<int>(defaultROI[i].x * input_image_shape[2]);
                bb.y = static_cast<int>(defaultROI[i].y * input_image_shape[1]);
                bb.width = static_cast<int>(defaultROI[i].width * input_image_shape[2]);
                bb.height = static_cast<int>(defaultROI[i].height * input_image_shape[1]);
                if (bb.x + bb.width >= input_image_shape[2])
                {
                    bb.x -= input_image_shape[2] - bb.width;
                }
                if (bb.y + bb.height >= input_image_shape[1])
                {
                    bb.y -= input_image_shape[1] - bb.height;
                }
                bb.x = std::max(0, bb.x);
                bb.y = std::max(0, bb.y);
                pixel_bounding_boxes.push_back(bb);
            }
            return pixel_bounding_boxes;
        }

        bool aq::nodes::INeuralNet::forwardAll()
        {
            cv::Scalar_<unsigned int> network_input_shape = getNetworkShape();
            std::vector<cv::Rect> pixel_bounding_boxes = getRegions();
            if (pixel_bounding_boxes.size() == 0)
            {
                preBatch(0);
                postBatch();
                return false;
            }
            auto input_image_shape = input->getShape();

            if (image_scale > 0)
            {
                MO_LOG(trace) << this->getTreeName() << " reshaping network";
                reshapeNetwork(static_cast<unsigned int>(pixel_bounding_boxes.size()),
                               static_cast<unsigned int>(input_image_shape[3]),
                               static_cast<unsigned int>(input_image_shape[1] * image_scale),
                               static_cast<unsigned int>(input_image_shape[2] * image_scale));
                MO_LOG(trace) << this->getTreeName() << " reshaping complete";
            }
            // Request a larger batch size
            if (pixel_bounding_boxes.size() != network_input_shape[0] && input_detections == nullptr)
            {
                MO_LOG(trace) << this->getTreeName() << " resizing batch size";
                reshapeNetwork(static_cast<unsigned int>(bounding_boxes->size()),
                               network_input_shape[1],
                               network_input_shape[2],
                               network_input_shape[3]);
                MO_LOG(trace) << this->getTreeName() << " batch resize complete";
            }

            MO_LOG(trace) << this->getTreeName() << " preprocessing";
            cv::cuda::GpuMat float_image;
            if (input->getDepth() != CV_32F)
            {
                input->getGpuMat(stream()).convertTo(float_image, CV_32F, stream());
            }
            else
            {
                input->clone(float_image, stream());
            }
            if (channel_mean[0] != 0.0 || channel_mean[1] != 0.0 || channel_mean[2] != 0.0)
            {
                cv::cuda::subtract(float_image, channel_mean, float_image, cv::noArray(), -1, stream());
            }
            if (pixel_scale != 1.0f)
            {
                cv::cuda::multiply(
                    float_image, cv::Scalar::all(static_cast<double>(pixel_scale)), float_image, 1.0, -1, stream());
            }
            MO_LOG(trace) << this->getTreeName() << " preprocessing complete";

            preBatch(static_cast<int>(pixel_bounding_boxes.size()));

            cv::cuda::GpuMat resized;
            auto net_input = getNetImageInput();
            MO_ASSERT(net_input.size());
            MO_ASSERT(net_input[0].size() == static_cast<size_t>(input->getChannels()));
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
                                         stream());
                    }
                    else
                    {
                        resized = float_image(pixel_bounding_boxes[i]);
                    }
                    cv::cuda::split(resized, net_input[j], stream());
                    end = start + j + 1;
                }
                MO_LOG(trace) << this->getTreeName() << " forward mini batch";
                if (forwardMinibatch())
                {
                    MO_LOG(trace) << this->getTreeName() << " forward mini batch complete";
                    std::vector<cv::Rect> batch_bounding_boxes;
                    DetectedObjectSet batch_detections;
                    for (size_t j = start; j < end; ++j)
                    {
                        batch_bounding_boxes.push_back(pixel_bounding_boxes[j]);
                    }
                    if (input_detections != nullptr)
                    {
                        batch_detections.setCatSet(input_detections->getCatSet());
                        for (size_t j = start; j < end; ++j)
                        {
                            batch_detections.push_back((*input_detections)[j]);
                        }
                    }
                    postMiniBatch(batch_bounding_boxes, batch_detections);
                    MO_LOG(trace) << this->getTreeName() << " post mini batch complete";
                }
            }
            postBatch();
            MO_LOG(trace) << this->getTreeName() << " post batch complete";
            return true;
        }
    } // namespace nodes
} // namespace aq
#endif
