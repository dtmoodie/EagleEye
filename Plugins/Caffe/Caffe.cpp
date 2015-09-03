#include "Caffe.h"

#include <external_includes/cv_cudaimgproc.hpp>
#include <external_includes/cv_cudaarithm.hpp>
#include <external_includes/cv_cudawarping.hpp>
#include <Manager.h>
#include "caffe/caffe.hpp"
#include <boost/tokenizer.hpp>
#include <ObjectDetection.hpp>
#include <string>
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("E:/libsrc/caffe/build_dev/lib/Debug/libcaffe-d.lib")
RUNTIME_COMPILER_LINKLIBRARY("E:/libsrc/caffe/build_dev/lib/Debug/proto-d.lib")
RUNTIME_COMPILER_LINKLIBRARY("E:/libsrc/protobuf/vsprojects/x64/Debug/libprotobufd.lib")
RUNTIME_COMPILER_LINKLIBRARY("C:/Repo/Raven/packages/ceres-glog.1.10.0/build/native/lib/x64/v120/libglog.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("E:/libsrc/caffe/build_dev/bin/Release/libcaffe.lib")
RUNTIME_COMPILER_LINKLIBRARY("E:/libsrc/caffe/build_dev/lib/Release/proto.lib")
RUNTIME_COMPILER_LINKLIBRARY("E:/libsrc/protobuf/vsprojects/x64/Release/libprotobuf.lib")
RUNTIME_COMPILER_LINKLIBRARY("C:/Repo/Raven/packages/ceres-glog.1.10.0/build/native/lib/x64/v120/libglog.lib")

#endif
#define CALL
  
#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lcaffe")
#define CALL
#endif


using namespace EagleLib;

IPerModuleInterface* CALL GetModule()
{
    return PerModuleInterface::GetInstance();
}
void CALL SetupIncludes()
{
#ifdef _MSC_VER
    EagleLib::NodeManager::getInstance().addIncludeDir("C:/code/EagleEye/EagleLib/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/libsrc/caffe/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/libs/OpenBLAS/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/libs/opencv/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/code/caffe/3rdparty/include/Snappy");
	EagleLib::NodeManager::getInstance().addIncludeDir("E:/code/caffe/3rdparty/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/libsrc/leveldb/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/code/caffe/3rdparty/include/lmdb");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/code/caffe/3rdparty/include/hdf5");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/libsrc/caffe/build/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("C:/Repo/Raven/packages/ceres-glog.1.10.0/build/native/include");
    EagleLib::NodeManager::getInstance().addIncludeDir("C:/libs/boost_1_57_0");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/libsrc/caffe/src");
    EagleLib::NodeManager::getInstance().addIncludeDir("E:/libsrc/protobuf/src/");
#else
#ifdef CAFFE_INCLUDES
    std::string text(CAFFE_INCLUDES);
    boost::char_separator<char> sep(",");
    boost::tokenizer< boost::char_separator<char> > tokens(text, sep);

    for(auto itr = tokens.begin();itr != tokens.end(); ++itr)
    {
        EagleLib::NodeManager::getInstance().addIncludeDir(*itr);
    }
#endif // CAFFE_INCLUDES

#endif

}
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
        cv::Scalar channel_mean;
	public:
		CaffeImageClassifier();
		virtual void Serialize(ISimpleSerializer* pSerializer);
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
        virtual void WrapInput();
        };
}

void CaffeImageClassifier::WrapInput()
{
    if(NN == nullptr)
    {
        NODE_LOG(error) << "Neural network not defined";
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

void CaffeImageClassifier::Serialize(ISimpleSerializer* pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(NN);
    SERIALIZE(weightsLoaded);
    SERIALIZE(labels);
    SERIALIZE(input_layer);
}

void CaffeImageClassifier::Init(bool firstInit)
{
    //std::cout << caffe::LayerRegistry<float>::LayerTypeList() << std::endl;
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    if(firstInit)
    {
        updateParameter("NN model file", boost::filesystem::path());
        updateParameter("NN weights file", boost::filesystem::path());
        updateParameter("Label file", boost::filesystem::path());
        updateParameter("Mean file", boost::filesystem::path());
		updateParameter("Subtraction required", false);
        updateParameter("Num classifications", 5);
        addInputParameter<std::vector<cv::Rect>>("Bounding boxes");
        weightsLoaded = false;
        input_layer = nullptr;
    }
}

cv::cuda::GpuMat CaffeImageClassifier::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed)
    {
        boost::filesystem::path& path = *getParameter<boost::filesystem::path>(0)->Data();
        if(boost::filesystem::exists(path))
        {
            NN.reset(new caffe::Net<float>(path.string(), caffe::TEST));
            WrapInput();
            parameters[0]->changed = false;
        }else
        {
            NODE_LOG(warning) << "Architecture file does not exist";
        }
    }
    if(parameters[1]->changed && NN)
    {
        boost::filesystem::path path = *getParameter<boost::filesystem::path>(1)->Data();
        if(boost::filesystem::exists(path))
        {
            NN->CopyTrainedLayersFrom(path.string());
            const std::vector<boost::shared_ptr<caffe::Layer<float>>>& layers = NN->layers();
            std::vector<std::string> layerNames;
            layerNames.reserve(layers.size());
            for(auto layer: layers)
            {
                layerNames.push_back(std::string(layer->type()));
            }
            NODE_LOG(info) << "Weights loaded";
            parameters[1]->changed = false;
            weightsLoaded = true;
            updateParameter("Loaded layers", layerNames);
        }else
        {
			NODE_LOG(warning) << "Weight file does not exist";
        }
    }
    if(parameters[2]->changed)
    {
        // Handle loading of the label file
        boost::filesystem::path& path = *getParameter<boost::filesystem::path>(2)->Data();
        if(boost::filesystem::exists(path))
        {
            if(boost::filesystem::is_regular_file(path))
            {
                std::ifstream ifs(path.string().c_str());
                if(!ifs)
                {
                    NODE_LOG(error) << "Unable to load label file";
                }
                labels.reset(new std::vector<std::string>());
                std::string line;
                while(std::getline(ifs,line))
                {
                    labels->push_back(line);
                }
                NODE_LOG(info) << "Loaded " << labels->size() <<" classes";
                parameters[2]->changed = false;
            }
        }
    }
    if(parameters[3]->changed)
    {
        boost::filesystem::path& path = *getParameter<boost::filesystem::path>(3)->Data();
        if(boost::filesystem::exists(path))
        {
            if(boost::filesystem::is_regular_file(path))
            {
                caffe::BlobProto blob_proto;
                if(caffe::ReadProtoFromBinaryFile(path.string().c_str(), &blob_proto))
                {
                    caffe::Blob<float> mean_blob;
                    mean_blob.FromProto(blob_proto);
                    if(input_layer == nullptr)
                    {
                        NODE_LOG(error) <<  "Input layer not defined";
                        return img;
                    }
                    if(input_layer->channels() != mean_blob.channels())
                    {
                        NODE_LOG(error) << "Number of channels of mean file doesn't match input layer.";
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
                    parameters[3]->changed = false;
                }else
                {
                    NODE_LOG(error) <<  "Unable to load mean file";
                }
            }
        }
    }
    if(NN == nullptr || weightsLoaded == false)
    {
        NODE_LOG(error) << "Model not loaded";
        return img;
    }
    if(img.size() != cv::Size(input_layer->width(), input_layer->height()))
    {
        cv::cuda::resize(img,img,cv::Size(input_layer->width(), input_layer->height()), 0, 0, cv::INTER_LINEAR, stream);
        NODE_LOG(warning) <<  "Resize required";
    }
    if(img.depth() != CV_32F)
    {
        img.convertTo(img, CV_32F,stream);
    }
    if(*getParameter<bool>("Subtraction required")->Data())
    {
        cv::cuda::subtract(img, channel_mean, img, cv::noArray(), -1, stream);
    }
    std::vector<cv::Rect> defaultROI;
    defaultROI.push_back(cv::Rect(cv::Point(), img.size()));
    std::vector<cv::Rect>* inputROIs = getParameter<std::vector<cv::Rect>>("Bounding boxes")->Data();
    if(inputROIs == nullptr)
    {
        inputROIs = &defaultROI;
    }

    if(inputROIs->size() > wrappedInputs.size())
    {
        NODE_LOG(warning) <<  "Too many input Regions of interest to handle in one pass, this network can only handle " << wrappedInputs.size() <<" inputs at a time";
    }
    for(int i = 0; i < inputROIs->size() && i < wrappedInputs.size(); ++i)
    {
        cv::cuda::split(img((*inputROIs)[i]), wrappedInputs[i], stream);
    }
    // Check if channels are still wrapping correctly
    if(input_layer->gpu_data() != reinterpret_cast<float*>(wrappedInputs[0][0].data))
    {
        NODE_LOG(error) << "Gpu mat not wrapping input blob!";
        WrapInput();
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
    const float* end = begin + output_layer->channels()*output_layer->num();
    int numClassifications = *getParameter<int>("Num classifications")->Data();
    std::vector<DetectedObject> objects(std::min(inputROIs->size(), wrappedInputs.size()));
    for(int i = 0; i < inputROIs->size() && i < wrappedInputs.size(); ++i)
    {
        auto idx = sort_indexes(begin + i * output_layer->channels(), (size_t)output_layer->channels());
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
    updateParameter("Detections", objects, Parameters::Parameter::Output);
    auto maxvalue = std::max_element(begin, end);
    TIME
    int idx = maxvalue - begin;
    float score = *maxvalue;
    updateParameter("Highest scoring class", idx);
    updateParameter("Highest score", score);
    if(labels)
    {
        if(idx < labels->size())
        {
            std::string label = (*labels)[idx];
            updateParameter("Highest scoring label", label);
        }
    }
    TIME
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(CaffeImageClassifier)
