#include "Caffe.h"
#include "caffe_init.h"

#include "EagleLib/nodes/Node.h"
#include <EagleLib/ObjectDetection.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudawarping.hpp>
#include "EagleLib/rcc/ObjectManager.h"
#include <parameters/ParameteredObjectImpl.hpp>


#include <boost/tokenizer.hpp>

#include <string>

#undef LOG
#include "caffe/caffe.hpp"
#undef LOG
#include <signals/logging.hpp>
#ifdef _MSC_VER
  #ifdef _DEBUG
	RUNTIME_COMPILER_LINKLIBRARY("libcaffe_SHARED-d.lib");
	RUNTIME_COMPILER_LINKLIBRARY("libglog.lib")
  #else
    RUNTIME_COMPILER_LINKLIBRARY("libcaffe_SHARED.lib")
  #endif
#else
#endif

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
	class CaffeImageClassifier : public Node
	{
		caffe::Blob<float>* input_layer;
		boost::shared_ptr<caffe::Net<float>> NN;
		bool weightsLoaded;
		boost::shared_ptr< std::vector< std::string > > labels;
		std::vector<std::vector<cv::cuda::GpuMat>> wrappedInputs;
		cv::Mat wrapped_output;
		cv::Scalar channel_mean;
	public:
		CaffeImageClassifier();
		virtual void Serialize(ISimpleSerializer* pSerializer);
		virtual void NodeInit(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
		virtual void WrapInput();
		virtual void WrapOutput();
        };
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

void CaffeImageClassifier::Serialize(ISimpleSerializer* pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(NN);
    SERIALIZE(weightsLoaded);
    SERIALIZE(labels);
    SERIALIZE(input_layer);
}

void CaffeImageClassifier::NodeInit(bool firstInit)
{
	EagleLib::caffe_init_singleton::inst();
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    
    if(firstInit)
    {
        updateParameter("NN model file", Parameters::ReadFile());
        updateParameter("NN weights file", Parameters::ReadFile());
        updateParameter("Label file", Parameters::ReadFile());
        updateParameter("Mean file", Parameters::ReadFile());
		updateParameter("Scale", 0.00390625f)->SetTooltip("Scale factor to multiply the image by, after mean subtraction");
		updateParameter("Subtraction required", false);
        updateParameter("Num classifications", 5);
        addInputParameter<std::vector<cv::Rect>>("Bounding boxes");
		
        weightsLoaded = false;
        input_layer = nullptr;
    }
}

cv::cuda::GpuMat CaffeImageClassifier::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    if(_parameters[0]->changed)
    {
        Parameters::ReadFile* path = getParameter<Parameters::ReadFile>(0)->Data();
        if(boost::filesystem::exists(*path))
        {
            NN.reset(new caffe::Net<float>(path->string(), caffe::TEST));
            WrapInput();
            _parameters[0]->changed = false;
        }else
        {
            BOOST_LOG_TRIVIAL(debug) << "Architecture file does not exist";
        }
    }
    if(_parameters[1]->changed && NN)
    {
        Parameters::ReadFile* path = getParameter<Parameters::ReadFile>(1)->Data();
        if(boost::filesystem::exists(*path))
        {
            NN->CopyTrainedLayersFrom(path->string());
            const std::vector<boost::shared_ptr<caffe::Layer<float>>>& layers = NN->layers();
            std::vector<std::string> layerNames;
            layerNames.reserve(layers.size());
            for(auto layer: layers)
            {
                layerNames.push_back(std::string(layer->type()));
            }
            BOOST_LOG_TRIVIAL(info) << "Weights loaded";
            _parameters[1]->changed = false;
            weightsLoaded = true;
            updateParameter("Loaded layers", layerNames);
        }else
        {
			BOOST_LOG_TRIVIAL(debug) << "Weight file does not exist";
        }
    }
    if(_parameters[2]->changed)
    {
        // Handle loading of the label file
        Parameters::ReadFile* path = getParameter<Parameters::ReadFile>(2)->Data();
        if(boost::filesystem::exists(*path))
        {
            if(boost::filesystem::is_regular_file(*path))
            {
                std::ifstream ifs(path->string().c_str());
                if(!ifs)
                {
                    BOOST_LOG_TRIVIAL(error) << "Unable to load label file";
                }
                labels.reset(new std::vector<std::string>());
                std::string line;
                while(std::getline(ifs,line))
                {
                    labels->push_back(line);
                }
                BOOST_LOG_TRIVIAL(info) << "Loaded " << labels->size() <<" classes";
                _parameters[2]->changed = false;
            }
        }
    }
    if(_parameters[3]->changed)
    {
        Parameters::ReadFile* path = getParameter<Parameters::ReadFile>(3)->Data();
        if(boost::filesystem::exists(*path))
        {
            if(boost::filesystem::is_regular_file(*path))
            {
                caffe::BlobProto blob_proto;
                if(caffe::ReadProtoFromBinaryFile(path->string().c_str(), &blob_proto))
                {
                    caffe::Blob<float> mean_blob;
                    mean_blob.FromProto(blob_proto);
                    if(input_layer == nullptr)
                    {
                        BOOST_LOG_TRIVIAL(error) <<  "Input layer not defined";
                        return img;
                    }
                    if(input_layer->channels() != mean_blob.channels())
                    {
                        BOOST_LOG_TRIVIAL(error) << "Number of channels of mean file doesn't match input layer.";
                        return img;
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

                    /* Compute the global mean pixel value and create a mean image
                     * filled with this value. */
                    channel_mean = cv::mean(mean);
                    updateParameter("Required mean subtraction", channel_mean);
                    updateParameter("Subtraction required", false);
                    _parameters[3]->changed = false;
                }else
                {
                    BOOST_LOG_TRIVIAL(error) <<  "Unable to load mean file";
                }
            }
        }
    }
    if(NN == nullptr || weightsLoaded == false)
    {
		BOOST_LOG_TRIVIAL(trace) << "Model not loaded";
        return img;
    }
    /*if(img.size() != cv::Size(input_layer->width(), input_layer->height()))
    {
        cv::cuda::GpuMat resized;
        cv::cuda::resize(img,resized,cv::Size(input_layer->width(), input_layer->height()), 0, 0, cv::INTER_LINEAR, stream);
        img = resized;
        BOOST_LOG_TRIVIAL(info) <<  "Resize required";
    }*/
	cv::cuda::GpuMat float_image;
    if(img.depth() != CV_32F)
    {
        img.convertTo(float_image, CV_32F,stream);
	}
	else
	{
		float_image = img;
	}

    if(*getParameter<bool>("Subtraction required")->Data())
    {
        cv::cuda::subtract(float_image, channel_mean, float_image, cv::noArray(), -1, stream);
    }
	cv::cuda::multiply(float_image, cv::Scalar::all(*getParameter<float>("Scale")->Data()), float_image, 1.0, -1, stream);
	cv::Mat tmp(float_image);
    std::vector<cv::Rect> defaultROI;
    defaultROI.push_back(cv::Rect(cv::Point(), img.size()));
    std::vector<cv::Rect>* inputROIs = getParameter<std::vector<cv::Rect>>("Bounding boxes")->Data();
    if(inputROIs == nullptr)
    {
        inputROIs = &defaultROI;
    }

    if(inputROIs->size() > wrappedInputs.size())
    {
		BOOST_LOG_TRIVIAL(warning) <<  "Too many input Regions of interest to handle in one pass, this network can only handle " << wrappedInputs.size() <<" inputs at a time";
    }
    cv::Size input_size(input_layer->width(), input_layer->height());
    for(int i = 0; i < inputROIs->size() && i < wrappedInputs.size(); ++i)
    {
        cv::cuda::GpuMat resized;
        if((*inputROIs)[i].size() != input_size)
        {
            cv::cuda::resize(float_image, resized, input_size, 0, 0, cv::INTER_LINEAR, stream);
        }else
        {
            resized = float_image((*inputROIs)[i]);
        }
		cv::Mat tmp(resized);
        cv::cuda::split(resized, wrappedInputs[i], stream);
    }
    // Check if channels are still wrapping correctly
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

    stream.waitForCompletion();
    TIME

    float loss;
    input_layer->mutable_gpu_data();
    TIME
    NN->ForwardPrefilled(&loss);
    TIME
    caffe::Blob<float>* output_layer = NN->output_blobs()[0];

    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels() * output_layer->num();
    const size_t step = output_layer->channels();


    if(begin != (const float*)wrapped_output.data)
    {
		BOOST_LOG_TRIVIAL(debug) << "Output not wrapped to mat";
        WrapOutput();
    }
    int numClassifications = *getParameter<int>("Num classifications")->Data();
    std::vector<DetectedObject> objects(std::min(inputROIs->size(), wrappedInputs.size()));
    for(int i = 0; i < inputROIs->size() && i < wrappedInputs.size(); ++i)
    {
        auto idx = sort_indexes_ascending(begin + i * output_layer->channels(), (size_t)output_layer->channels());
        objects[i].detections.resize(numClassifications);
        for(int j = 0; j < numClassifications; ++j)
        {
            objects[i].detections[j].confidence = (begin + i * output_layer->channels())[idx[j]];
            objects[i].detections[j].classNumber = idx[j];
            if(labels && idx[j] < labels->size())
            {
                objects[i].detections[j].label = (*labels)[idx[j]];
            }
        }
        objects[i].boundingBox = (*inputROIs)[i];
    }
    updateParameter("Detections", objects)->type =  Parameters::Parameter::Output;
    updateParameter("Highest scoring class", objects[0].detections[0].classNumber);
    updateParameter("Highest score", objects[0].detections[0].confidence);
	updateParameter("Probability distribution", wrapped_output);
    if(labels)
    {
        updateParameter("Highest scoring label", objects[0].detections[0].label);
    }
    TIME
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(CaffeImageClassifier, Image, Extractor, ObjectClassification)
