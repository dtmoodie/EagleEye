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

std::vector<cv::cuda::GpuMat> getInput(float* ptr, int width, int height, int c)
{
    std::vector<cv::cuda::GpuMat> output;
    for (int i = 0; i < c; ++i)
    {
        output.emplace_back(height, width, CV_32F, ptr);
        ptr += (width * height);
    }
    return output;
}

std::vector<cv::cuda::GpuMat> getInput(float** ptr, int width, int height, int c)
{
    std::vector<cv::cuda::GpuMat> output;
    for (int i = 0; i < c; ++i)
    {
        output.emplace_back(height, width, CV_32F, ptr);
        ptr += (width * height);
    }
    return output;
}

namespace darknet
{

    // create a compatibility layer depending on if this is the OG darknet from pjreddie or if this is the new darknet
#ifdef LIB_API
    struct Network
    {
        Network(const std::string& model_file, const std::string& weight_file, const uint32_t batch_size) : state{0}
        {
            char* mf = const_cast<char*>(model_file.c_str());
            char* wf = const_cast<char*>(weight_file.c_str());
            m_network = load_network_custom(mf, wf, 0, batch_size);

            // m_network = load_network(mf, wf, 0);
        }

        ~Network() {}

        cv::Scalar_<unsigned int> getInputShape() const
        {
            return {1, uint32_t(m_network->h), uint32_t(m_network->w), uint32_t(m_network->c)};
        }

        std::vector<cv::cuda::GpuMat> getInputBindings()
        {
            float** input = m_network->input_gpu;

            // TODO replace with some kind of managed tensor
            cv::cuda::createContinuous(m_network->h * m_network->c, m_network->w, CV_32F, m_buffer);
            float* ptr = ct::ptrCast<float>(m_buffer.data);
            input[0] = ptr;
            m_input = ptr;
            std::vector<cv::cuda::GpuMat> output;
            for (int i = 0; i < m_network->c; ++i)
            {
                output.emplace_back(m_network->h, m_network->w, CV_32F, ptr);
                ptr += (m_network->h * m_network->w);
            }
            m_network->input_state_gpu = ptr;

            return output;
        }

        void forward(mo::IDeviceStream&)
        {
            state.index = 0;
            state.net = *m_network;
            state.workspace = m_network->workspace;
            state.input = m_network->input_state_gpu;
            state.input = m_input;
            state.truth = 0;
            state.train = 0;
            state.delta = 0;

            for (int i = 0; i < m_network->n; ++i)
            {
                state.index = i;
                darknet::layer l = m_network->layers[i];

                l.forward_gpu(l, state);
                state.input = l.output_gpu;
            }
        }

        void getDetections(float original_height,
                           float original_width,
                           float thresh,
                           float hier,
                           float nms,
                           aq::DetectedObjectSet& output,
                           mo::IDeviceStream& stream)
        {
            cudaDeviceSynchronize();
            for (int i = 0; i < m_network->n; ++i)
            {
                darknet::layer l = m_network->layers[i];
                if (l.type == darknet::YOLO)
                {
                    float* output_cpu = l.output;
                    const float* output_gpu = l.output_gpu;
                    const size_t size = l.outputs * l.batch;
                    stream.deviceToHost({output_cpu, size}, {output_gpu, size});
                }
            }
            stream.synchronize();

            int* map = nullptr;
            int nboxes = 0;
            const int letter_box = m_network->letter_box;
            detection* dets = get_network_boxes(
                m_network, original_width, original_height, thresh, hier, map, 0, &nboxes, letter_box);

            layer l = m_network->layers[m_network->n - 1];
            do_nms_obj(dets, nboxes, l.classes, nms);

            // TODO need a reserve function
            output.resize(nboxes);
            auto ids = output.getComponentMutable<aq::detection::Id>();
            auto bbs = output.getComponentMutable<aq::detection::BoundingBox2d>();
            auto cls = output.getComponentMutable<aq::detection::Classifications>();
            auto conf = output.getComponentMutable<aq::detection::Confidence>();
            const auto cats = output.getCatSet();
            int count = 0;
            for (int i = 0; i < nboxes; ++i)
            {
                const float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
                const float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
                const auto confidence = dets[i].objectness;
                if (confidence > thresh)
                {
                    bbs[count].x = xmin;
                    bbs[count].y = ymin;
                    bbs[count].width = dets[i].bbox.w;
                    bbs[count].height = dets[i].bbox.h;

                    conf[count] = confidence;

                    ids[count] = count;

                    // TODO iterate over classes
                    const auto num_classes = dets[i].classes;
                    MO_ASSERT_EQ(num_classes, cats->size());
                    for (int j = 0; j < num_classes; ++j)
                    {
                        const float prob = dets[i].prob[j];

                        if (prob > 0.0)
                        {
                            const aq::Category& cat = (*cats)[j];
                            cls[count].append(cat(prob));
                        }
                    }
                    ++count;
                }
            }
            output.resize(count);
            free_detections(dets, nboxes);
        }

        network* m_network = nullptr;
        network_state state = {0};
        float* m_input = nullptr;
        cv::cuda::GpuMat m_buffer;
    };

#else
    // TODO compatibility layer for original darknet
#endif

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
    bool initNetwork() override
    {
        if ((m_net == nullptr))
        {
            if (boost::filesystem::exists(model_file) && boost::filesystem::exists(weight_file))
            {
                std::string model = model_file.string();
                std::string weights = weight_file.string();
                m_net.reset(new darknet::Network(model, weights, 1));

                m_input_channels = m_net->getInputBindings();
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

    bool reshapeNetwork(unsigned int , unsigned int , unsigned int , unsigned int ) override
    {
        return false;
    }

    cv::Scalar_<unsigned int> getNetworkShape() const override { return m_net->getInputShape(); }

    std::vector<std::vector<cv::cuda::GpuMat>> getNetImageInput(int ) override { return {m_input_channels}; }

    bool forwardMinibatch(mo::IDeviceStream& stream) override
    {
        stream.synchronize();

        m_net->forward(stream);
        return true;
    }

    void postMiniBatch(mo::IDeviceStream& stream,
                       const std::vector<cv::Rect>& ,
                       const aq::DetectedObjectSet* ) override
    {
        auto input_image_shape = this->input->size();
        m_dets.setCatSet(this->getLabels());
        m_net->getDetections(
            input_image_shape(0), input_image_shape(1), det_thresh, cat_thresh, nms_threshold, m_dets, stream);
        return;
    }

    void postBatch() override
    {
        this->output.publish(std::move(m_dets), mo::tags::param = &input_param);
        m_dets = Output_t();
    }

  private:
    std::unique_ptr<darknet::Network> m_net;
    std::vector<cv::cuda::GpuMat> m_input_channels;
    cv::cuda::GpuMat m_input_buffer;
};

#endif
MO_REGISTER_CLASS(YOLO)
