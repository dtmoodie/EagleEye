#include "Caffe.h"
#include "caffe/caffe.hpp"
#include <external_includes/cv_cudaimgproc.hpp>
#include <external_includes/cv_cudaarithm.hpp>
#include <external_includes/cv_cudawarping.hpp>
RUNTIME_COMPILER_LINKLIBRARY("-lcaffe")

using namespace EagleLib;

IPerModuleInterface* CALL GetModule()
{
    return PerModuleInterface::GetInstance();
}
void CALL setupIncludes()
{

}

//#include "ros/ros.h"
//#include <iostream>
//#include <caffe/caffe.hpp>
//#include <vector>
//#include <sstream>
//using std::string;
//using caffe::Blob;
//using caffe::Caffe;
//using caffe::Datum;
//using caffe::Net;
//using caffe::shared_ptr;
//using caffe::vector;
//using caffe::MemoryDataLayer;

//void Log(std::string msg) {
//	std::cout << msg << std::endl;
//}

//int main(int argc, char **argv) {
//#ifdef CPU_ONLY:
//	std::cout<<"CPU_ONLY" << std::endl;
//    Caffe::set_mode(Caffe::CPU);
//#endif
//	ros::init(argc, argv, "ros_caffe_main");

//	std::string model_path = "/home/darrenl/workspace/eclipse_clusplus_workspace/TestCaffe/bvlc_reference_deploy_memorylayer.prototxt";
//	std::string weights_path = "/home/darrenl/workspace/eclipse_clusplus_workspace/TestCaffe/bvlc_reference_caffenet.caffemodel";
//	std::string image_path = "/home/darrenl/cat.jpg";
//	// Use CPU only.
//	// Initial
//	Net<float> *caffe_net;
//	caffe_net = new Net<float>(model_path, caffe::TEST);
//	caffe_net->CopyTrainedLayersFrom(weights_path);
//	// Assign datum
//	Datum datum;
//	if (!ReadImageToDatum(image_path, 1, 227, 227, &datum)) {
//		Log("Read Image fail");
//		return -1;
//	}
//	// Use MemoryDataLayer
//	const boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer =
//			boost::static_pointer_cast<MemoryDataLayer<float> >(
//					caffe_net->layer_by_name("data"));
//	std::vector<Datum> datum_vector;
//	datum_vector.push_back(datum);
//	memory_data_layer->AddDatumVector(datum_vector);
//	std::vector<Blob<float>*> dummy_bottom_vec;
//	float loss;
//	const std::vector<Blob<float>*>& result = caffe_net->Forward(
//			dummy_bottom_vec, &loss);
//	const std::vector<float> probs = std::vector<float>(result[1]->cpu_data(),
//			result[1]->cpu_data() + result[1]->count());
//	// Find the index with max prob
//	int max_index = -1;
//	float max_value;
//	for (int index = 0; index != probs.size(); index++) {
//		if (index == 0) {
//			max_index = 0;
//			max_value = probs[max_index];
//			continue;
//		}
//		// Compare
//		if (max_value < probs[index]) {
//			max_value = probs[index];
//			max_index = index;
//		}
//	}
//	/**
//	 * Result : Toilet is 861 Cat is 281, etc..
//	 **/
//	std::stringstream ss;
//	ss << "max index: " << max_index;
//	Log(ss.str());
//	return 0;
//}
namespace EagleLib
{
	class CaffeImageClassifier : public Node
	{
        caffe::Blob<float>* input_layer;
        boost::shared_ptr<caffe::Net<float>> NN;
        bool weightsLoaded;
        boost::shared_ptr< std::vector< std::string > > labels;
        std::vector<cv::cuda::GpuMat> wrappedChannels;
        cv::Scalar channel_mean;
	public:
		CaffeImageClassifier();
		virtual void Serialize(ISimpleSerializer* pSerializer);
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
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

    if(firstInit)
    {
        updateParameter("NN model file", boost::filesystem::path());
        updateParameter("NN weights file", boost::filesystem::path());
        updateParameter("Label file", boost::filesystem::path());
        updateParameter("Mean file", boost::filesystem::path());
        weightsLoaded = false;
        //labels.reset(new std::vector<std::string>>);
        input_layer = nullptr;
    }
}

cv::cuda::GpuMat CaffeImageClassifier::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed)
    {
        boost::filesystem::path& path = getParameter<boost::filesystem::path>(0)->data;
        if(boost::filesystem::exists(path))
        {
            NN.reset(new caffe::Net<float>(path.string(), caffe::TEST));
            std::stringstream ss;
            ss << "Architecture loaded, num inputs: " << NN->num_inputs();
            ss << " num outputs: " << NN->num_outputs();
            input_layer = NN->input_blobs()[0];
            ss << " input channels: " << input_layer->channels();
            ss << " input size: (" << input_layer->width() << ", " << input_layer->height() << ")";
            float* input_data = input_layer->mutable_gpu_data();
            int width = input_layer->width();
            int height = input_layer->height();
            for(int i = 0; i < input_layer->channels(); ++i)
            {
                cv::cuda::GpuMat channel(height, width, CV_32FC1, input_data);
                wrappedChannels.push_back(channel);
                input_data += height*width;
            }
            log(Status, ss.str());
            parameters[0]->changed = false;
        }else
        {
            log(Warning, "Architecture file does not exist");
        }
    }
    if(parameters[1]->changed && NN)
    {
        boost::filesystem::path path = getParameter<boost::filesystem::path>(1)->data;
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
            log(Status, "Weights loaded");
            parameters[1]->changed = false;
            weightsLoaded = true;
            updateParameter("Loaded layers", layerNames);
        }else
        {
            log(Warning, "Weight file does not exist");
        }
    }
    if(parameters[2]->changed)
    {
        // Handle loading of the label file
        boost::filesystem::path& path = getParameter<boost::filesystem::path>(2)->data;
        if(boost::filesystem::exists(path))
        {
            if(boost::filesystem::is_regular_file(path))
            {
                std::ifstream ifs(path.string().c_str());
                if(!ifs)
                {
                    log(Error, "Unable to load label file");
                }
                labels.reset(new std::vector<std::string>());
                std::string line;
                while(std::getline(ifs,line))
                {
                    labels->push_back(line);
                }
                log(Status, "Loaded " + boost::lexical_cast<std::string>(labels->size()) + " classes");
                parameters[2]->changed = false;
            }
        }
    }
    if(parameters[3]->changed)
    {
        boost::filesystem::path& path = getParameter<boost::filesystem::path>(3)->data;
        if(boost::filesystem::exists(path))
        {
            if(boost::filesystem::is_regular_file(path))
            {
                caffe::BlobProto blob_proto;
                if(caffe::ReadProtoFromBinaryFile(path.string().c_str(), &blob_proto))
                {
                    caffe::Blob<float> mean_blob;
                    mean_blob.FromProto(blob_proto);
                    if(input_layer->channels() != mean_blob.channels())
                    {
                        log(Error,"Number of channels of mean file doesn't match input layer.");
                        return img;
                    }

                    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
                    std::vector<cv::Mat> channels;
                    float* data = mean_blob.mutable_cpu_data();
                    for (int i = 0; i < input_layer->channels(); ++i) {
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
                }else
                {
                    log(Error, "Unable to load mean file");
                }
            }
        }
    }
    if(NN == nullptr || weightsLoaded == false)
    {
        log(Error, "Model not loaded");
        return img;
    }
    if(img.size() != cv::Size(input_layer->width(), input_layer->height()))
    {
        cv::cuda::resize(img,img,cv::Size(input_layer->width(), input_layer->height()), 0, 0, cv::INTER_LINEAR, stream);
    }
    if(img.depth() != CV_32F)
    {
        img.convertTo(img, CV_32F,stream);
    }
    if(getParameter<bool>("Subtraction required")->data)
    {
        cv::cuda::subtract(img, channel_mean, img, cv::noArray(), -1, stream);
    }
    cv::cuda::split(img,wrappedChannels,stream);
    // Check if channels are still wrapping correctly
    if(input_layer->gpu_data() != reinterpret_cast<float*>(wrappedChannels[0].data))
    {
        log(Error, "Gpu mat not wrapping input blob!");
        wrappedChannels.clear();
        float* input_data = input_layer->mutable_gpu_data();
        int width = input_layer->width();
        int height = input_layer->height();
        for(int i = 0; i < input_layer->channels(); ++i)
        {
            cv::cuda::GpuMat channel(height, width, CV_32FC1, input_data);
            wrappedChannels.push_back(channel);
            input_data += height*width;
        }

        return img;
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
    const float* end = begin + output_layer->channels();
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
