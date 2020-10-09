#include <ct/types/opencv.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <aqcore/INeuralNet.hpp>
#include <boost/filesystem.hpp>

#ifdef _MSC_VER

#define OPENCV
#include "yolo_v2_class.hpp"
class YOLO : virtual public aq::nodes::INeuralNet
{
  public:
    MO_DERIVE(YOLO, aq::nodes::INeuralNet)
    MO_END;

  protected:
    virtual std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int batch_size = 1)
    {
        if (!detector)
        {
            if (boost::filesystem::exists(model_file) && boost::filesystem::exists(weight_file))
            {
                detector = std::make_shared<Detector>(model_file.string(), weight_file.string());
            }
            else
            {
                LOG(warning) << "Model file '" << model_file << "' or weight file '" << weight_file
                             << "' does not exist";
            }
        }
        if (input_buffer.empty() && detector)
        {
            int width, height;
            if (detector->get_network_size(width, height))
            {
                workspace = cv::cuda::GpuMat(1, width * height * 3, CV_32F);
                float* ptr = workspace.ptr<float>();
                std::vector<cv::cuda::GpuMat> channels;
                for (int i = 0; i < 3; ++i)
                {
                    channels.emplace_back(height, width, CV_32F, static_cast<void*>(ptr), width);
                    ptr += height * width;
                }
                if (swap_bgr)
                {
                    std::swap(channels[0], channels[2]);
                }
                input_buffer.push_back(channels);
                input_ptr = ptr;
            }
        }
        return input_buffer;
    }
    virtual bool forwardMinibatch()
    {
        if (input_ptr)
        {
            current_detections = detector->detect(input_ptr);
            return true;
        }
        return false;
    }
    virtual void postBatch(const std::vector<cv::Rect>& batch_bb, const std::vector<aq::DetectedObject2d>& dets)
    {
        // do things with current_detections
    }
    virtual bool reshapeNetwork(int num, int channels, int height, int width) { return false; }

  private:
    std::shared_ptr<Detector> detector;
    std::vector<std::vector<cv::cuda::GpuMat>> input_buffer;
    cv::cuda::GpuMat workspace;
    float* input_ptr = nullptr;
    std::vector<bbox_t> current_detections;
};

#else
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cudnn.h"
#include "curand.h"
namespace darknet
{
    extern "C" {
#define CUDNN
#define GPU
#include "darknet.h"
#undef GPU
#undef CUDNN
    }
} // namespace darknet

class YOLO : virtual public aqcore::INeuralNet
{
  public:
    using OutputComponents_t = ct::VariadicTypedef<aq::detection::BoundingBox2d,
                                                   aq::detection::Classifications,
                                                   aq::detection::Confidence,
                                                   aq::detection::Id>;
    using Output_t = aq::TDetectedObjectSet<OutputComponents_t>;

    Output_t m_dets;

    MO_DERIVE(YOLO, aqcore::INeuralNet)
        PARAM(float, det_thresh, 0.5f)
        PARAM(float, cat_thresh, 0.5f)
        PARAM(float, nms_threshold, 0.45f)
        OUTPUT(Output_t, output)
    MO_END;

  protected:
    bool initNetwork()
    {
        if ((m_net == nullptr))
        {
            if (boost::filesystem::exists(model_file) && boost::filesystem::exists(weight_file))
            {
                darknet::network* net = darknet::load_network(
                    const_cast<char*>(model_file.string().c_str()), const_cast<char*>(weight_file.string().c_str()), 0);
                darknet::set_batch_network(net, 1);
                m_net = net;
                net->train = false;
                int width = m_net->w;
                int height = m_net->h;
                float* ptr = m_net->input_gpu;
                m_input = ptr;
                for (int i = 0; i < m_net->c; ++i)
                {
                    m_input_channels.emplace_back(height, width, CV_32F, ptr);
                    ptr += (width * height);
                }
                if (swap_bgr)
                {
                    std::swap(m_input_channels[0], m_input_channels[2]);
                }
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    bool reshapeNetwork(unsigned int num, unsigned int channels, unsigned int height, unsigned int width)
    {
        return false;
    }

    cv::Scalar_<unsigned int> getNetworkShape() const
    {
        return {1, 3, static_cast<unsigned int>(m_net->h), static_cast<unsigned int>(m_net->h)};
    }

    std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int requested_batch_size) { return {m_input_channels}; }

    bool forwardMinibatch()
    {
        m_net->input_gpu = m_input;
        this->getStream()->synchronize();
        int i;
        for (i = 0; i < m_net->n; ++i)
        {
            this->getLogger().trace("{} forward", i);
            m_net->index = i;
            darknet::layer l = m_net->layers[i];
            if (l.delta_gpu)
            {
                darknet::fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }
            l.forward_gpu(l, *m_net);
            m_net->input_gpu = l.output_gpu;
            m_net->input = l.output;
            if (l.truth)
            {
                m_net->truth_gpu = l.output_gpu;
                m_net->truth = l.output;
            }
        }
        return true;
    }

    Output_t m_output;

    void postMiniBatch(const std::vector<cv::Rect>& batch_bb, const aq::DetectedObjectSet* dets) override
    {
        darknet::layer l = darknet::get_network_output_layer(m_net);
        darknet::cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
        int nboxes = 0;
        auto det_ptr = darknet::get_network_boxes(
            m_net, m_input_channels[0].cols, m_input_channels[0].rows, det_thresh, cat_thresh, 0, 1, &nboxes);

        if (nms_threshold > 0.0f)
        {
            darknet::do_nms_sort(det_ptr, nboxes, l.classes, nms_threshold);
        }

        std::shared_ptr<darknet::detection> det_owner(
            det_ptr, [nboxes](darknet::detection* dets) { darknet::free_detections(dets, nboxes); });
        auto labels = this->getLabels();
        m_output.setCatSet(labels);
        for (int i = 0; i < nboxes; ++i)
        {
            if (det_ptr[i].objectness > det_thresh)
            {
                darknet::box box = det_ptr[i].bbox;
                cv::Rect2f rect(box.x, box.y, box.w, box.h);
                rect.x = rect.x - rect.width / 2.0f;
                rect.y = rect.y - rect.height / 2.0f;
                aq::clipNormalizedBoundingBox(rect);
                aq::boundingBoxToPixels(rect, batch_bb[0].size());
                aq::DetectedObject det(rect);
                det.confidence = det_ptr[i].objectness;
                det.id = i;
                std::vector<aq::Classification> cats;
                for (int j = 0; j < labels->size(); ++j)
                {
                    if (det_ptr[i].prob[j] > cat_thresh)
                    {
                        auto label = (*labels)[j](det_ptr[i].prob[j]);
                        cats.emplace_back(std::move(label));
                    }
                }
                if (!cats.empty())
                {
                    det.classifications = cats;
                    m_output.push_back(std::move(det));
                }
            }
        }
        return;
    }

    void postBatch()
    {
        // output_param.emitUpdate(input_param);
        this->output.publish(std::move(m_output), mo::tags::param = &input_param);
    }

  private:
    darknet::network* m_net;
    std::vector<cv::cuda::GpuMat> m_input_channels;
    cv::cuda::GpuMat m_input_buffer;
    float* m_input = nullptr;
};

#endif
MO_REGISTER_CLASS(YOLO)
