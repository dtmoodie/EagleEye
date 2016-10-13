#define PARAMTERS_GENERATE_PERSISTENCE

#include "Caffe.h"
#include "caffe_init.h"

#include "EagleLib/nodes/Node.h"
#include "EagleLib/Nodes/NodeInfo.hpp"
#include <EagleLib/ObjectDetection.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudawarping.hpp>
#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include "caffe_include.h"

#include <boost/tokenizer.hpp>

#include <string>

#undef LOG
#include "caffe/caffe.hpp"
#undef LOG


SETUP_PROJECT_IMPL;


using namespace EagleLib;
using namespace EagleLib::Nodes;

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


template <typename T>
std::vector<size_t> sort_indexes(const T* begin, size_t size) {

  // initialize original index locations
  std::vector<size_t> idx(size);
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), [&begin](size_t i1, size_t i2) {return begin[i1] < begin[i2];});

  return idx;
}
template <typename T>
std::vector<size_t> sort_indexes_ascending(const T* begin, size_t size) {

    // initialize original index locations
    std::vector<size_t> idx(size);
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&begin](size_t i1, size_t i2) {return begin[i1] > begin[i2]; });

    return idx;
}
template <typename T>
std::vector<size_t> sort_indexes(const T* begin, const T* end) {
    return sort_indexes<T>(begin, end - begin);
}


namespace EagleLib
{
    namespace Nodes
    {
        class CaffeImageClassifier : public Node
        {
            cv::Mat wrapped_output;
        public:
            
            virtual void NodeInit(bool firstInit);
            virtual void WrapInput();
            virtual void WrapOutput();

            MO_DERIVE(CaffeImageClassifier, Node)
                PROPERTY(caffe::Blob<float>*, input_layer, nullptr);
                PROPERTY(boost::shared_ptr<caffe::Net<float>>, NN, boost::shared_ptr<caffe::Net<float>>());
                PROPERTY(boost::shared_ptr< std::vector< std::string > >, labels, boost::shared_ptr< std::vector< std::string > >());
                PROPERTY(std::vector<std::vector<cv::cuda::GpuMat>>, wrappedInputs, std::vector<std::vector<cv::cuda::GpuMat>>());
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
                OUTPUT(std::vector<DetectedObject>, detections, std::vector<DetectedObject>());
            MO_END;
        protected:
            bool ProcessImpl();
        };
    }
}

void CaffeImageClassifier::WrapInput()
{
    if(NN == nullptr)
    {
        BOOST_LOG_TRIVIAL(error) << "Neural network not defined";
        return;
    }
    if(NN->num_inputs() == 0)
        return;

    input_layer = NN->input_blobs()[0];

    std::stringstream ss;
    ss << "Architecture loaded, num inputs: " << NN->num_inputs();
    ss << " num outputs: " << NN->num_outputs();
    input_layer = NN->input_blobs()[0];
    ss << " input batch size: " << input_layer->num();
    ss << " input channels: " << input_layer->channels();
    ss << " input size: (" << input_layer->width() << ", " << input_layer->height() << ")";
    float* input_data = input_layer->mutable_gpu_data();

    int width = input_layer->width();
    int height = input_layer->height();

    for(int j = 0; j < input_layer->num(); ++j)
    {
        std::vector<cv::cuda::GpuMat> wrappedChannels;
        for(int i = 0; i < input_layer->channels(); ++i)
        {
            cv::cuda::GpuMat channel(height, width, CV_32FC1, input_data);
            wrappedChannels.push_back(channel);
            input_data += height*width;
        }
        wrappedInputs.push_back(wrappedChannels);
    }
}
void CaffeImageClassifier::WrapOutput()
{
    if (NN == nullptr)
    {
        BOOST_LOG_TRIVIAL(error) << "Neural network not defined";
        return;
    }
    if (NN->num_inputs() == 0)
        return;

    caffe::Blob<float>* output_layer = NN->output_blobs()[0];
    float* begin = output_layer->mutable_cpu_data();
    float* end = begin + output_layer->channels()*output_layer->num();
    wrapped_output = cv::Mat(1, end - begin, CV_32F, begin);
}

void CaffeImageClassifier::NodeInit(bool firstInit)
{
    EagleLib::caffe_init_singleton::inst();
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}

bool CaffeImageClassifier::ProcessImpl()
{
    if (::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
    if(nn_model_file_param.modified)
    {
        if(boost::filesystem::exists(nn_model_file))
        {
            NN.reset(new caffe::Net<float>(nn_model_file.string(), caffe::TEST));
            WrapInput();
            nn_model_file_param.modified = false;
        }else
        {
            BOOST_LOG_TRIVIAL(debug) << "Architecture file does not exist";
        }
    }
    if(nn_weight_file_param.modified && NN)
    {
        if(boost::filesystem::exists(nn_weight_file))
        {
            NN->CopyTrainedLayersFrom(nn_weight_file.string());
            const std::vector<boost::shared_ptr<caffe::Layer<float>>>& layers = NN->layers();
            std::vector<std::string> layerNames;
            layerNames.reserve(layers.size());
            for (auto layer : layers)
            {
                layerNames.push_back(std::string(layer->type()));
            }
            BOOST_LOG_TRIVIAL(info) << "Weights loaded";
            weightsLoaded = true;
            UpdateParameter("Loaded layers", layerNames);
            nn_weight_file_param.modified = false;
        }else
        {
            BOOST_LOG_TRIVIAL(debug) << "Weight file does not exist";
        }
    }
    if(label_file_param.modified)
    {
        if(boost::filesystem::exists(label_file))
        {
            std::ifstream ifs(label_file.string().c_str());
            if (!ifs)
            {
                BOOST_LOG_TRIVIAL(error) << "Unable to load label file";
            }
            labels.reset(new std::vector<std::string>());
            std::string line;
            while (std::getline(ifs, line))
            {
                labels->push_back(line);
            }
            BOOST_LOG_TRIVIAL(info) << "Loaded " << labels->size() << " classes";
            label_file_param.modified = false;
        }
    }
    if(mean_file_param.modified)
    {
        if(boost::filesystem::exists(mean_file))
        {
            if (boost::filesystem::is_regular_file(mean_file))
            {
                caffe::BlobProto blob_proto;
                if (caffe::ReadProtoFromBinaryFile(mean_file.string().c_str(), &blob_proto))
                {
                    caffe::Blob<float> mean_blob;
                    mean_blob.FromProto(blob_proto);
                    if (input_layer == nullptr)
                    {
                        BOOST_LOG_TRIVIAL(error) << "Input layer not defined";
                        return false;
                    }
                    if (input_layer->channels() != mean_blob.channels())
                    {
                        BOOST_LOG_TRIVIAL(error) << "Number of channels of mean file doesn't match input layer.";
                        return false;
                    }

                    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
                    std::vector<cv::Mat> channels;
                    float* data = mean_blob.mutable_cpu_data();
                    for (int i = 0; i < input_layer->channels(); ++i)
                    {
                        /* Extract an individual channel. */
                        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
                        channels.push_back(channel);
                        data += mean_blob.height() * mean_blob.width();
                    }

                    /* Merge the separate channels into a single image. */
                    cv::Mat mean;
                    cv::merge(channels, mean);
                    channel_mean = cv::mean(mean);
                }
            }
        }
    }
    if(NN == nullptr || weightsLoaded == false)
    {
        BOOST_LOG_TRIVIAL(trace) << "Model not loaded";
        return false;
    }
    cv::cuda::GpuMat float_image;
    if (input->GetDepth() != CV_32F)
    {
        input->GetGpuMat(Stream()).convertTo(float_image, CV_32F, Stream());
    }
    else
    {
        float_image = input->GetGpuMat(Stream());
    }
    cv::cuda::multiply(float_image, cv::Scalar::all(scale), float_image, 1.0, -1, Stream());
    std::vector<cv::Rect> defaultROI;
    defaultROI.push_back(cv::Rect(cv::Point(), input->GetSize()));
    if (bounding_boxes == nullptr)
    {
        bounding_boxes = &defaultROI;
    }
    if (bounding_boxes->size() > wrappedInputs.size())
    {
        BOOST_LOG_TRIVIAL(debug) << "Too many input Regions of interest to handle in one pass, this network can only handle " << wrappedInputs.size() << " inputs at a time";
    }
    cv::Size input_size(input_layer->width(), input_layer->height());
    for (int i = 0; i < bounding_boxes->size() && i < wrappedInputs.size(); ++i)
    {
        cv::cuda::GpuMat resized;
        if ((*bounding_boxes)[i].size() != input_size)
        {
            cv::cuda::resize(float_image, resized, input_size, 0, 0, cv::INTER_LINEAR, Stream());
        }
        else
        {
            resized = float_image((*bounding_boxes)[i]);
        }
        cv::cuda::split(resized, wrappedInputs[i], Stream());
    }
    const float* blob_data = input_layer->gpu_data();
    int channels = input_layer->channels();
    for (int i = 0; i < wrappedInputs.size(); ++i)
    {
        for (int j = 0; j < wrappedInputs[i].size(); ++j)
        {
            if (reinterpret_cast<float*>(wrappedInputs[i][j].data) != blob_data + input_size.area() * j + i*input_size.area() * channels)
            {
                BOOST_LOG_TRIVIAL(debug) << "GPU mat not mapping input blob";
                WrapInput();
                break;
            }
        }
    }
    float loss;
    input_layer->mutable_gpu_data();
    NN->ForwardPrefilled(&loss);
    caffe::Blob<float>* output_layer = NN->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels() * output_layer->num();
    const size_t step = output_layer->channels();
    if (begin != (const float*)wrapped_output.data)
    {
        BOOST_LOG_TRIVIAL(debug) << "Output not wrapped to mat";
        WrapOutput();
    }
    
    std::vector<DetectedObject> objects(std::min(bounding_boxes->size(), wrappedInputs.size()));
    for (int i = 0; i < bounding_boxes->size() && i < wrappedInputs.size(); ++i)
    {
        auto idx = sort_indexes_ascending(begin + i * output_layer->channels(), (size_t)output_layer->channels());
        objects[i].detections.resize(num_classifications);
        for (int j = 0; j < num_classifications; ++j)
        {
            objects[i].detections[j].confidence = (begin + i * output_layer->channels())[idx[j]];
            objects[i].detections[j].classNumber = idx[j];
            if (labels && idx[j] < labels->size())
            {
                objects[i].detections[j].label = (*labels)[idx[j]];
            }
        }
        objects[i].boundingBox = (*bounding_boxes)[i];
    }
    detections_param.UpdateData(objects, input_param.GetTimestamp(), _ctx);

    return true;
}

MO_REGISTER_CLASS(CaffeImageClassifier)
