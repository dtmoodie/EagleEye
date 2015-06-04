#include "Caffe.h"
#include "caffe/caffe.hpp"

RUNTIME_COMPILER_LINKLIBRARY("-lcaffe")

using namespace EagleLib;

IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}
void setupIncludes()
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

class CaffeImageClassifier: public Node
{
    boost::shared_ptr<caffe::Net<float>> NN;
    boost::shared_ptr<caffe::MemoryDataLayer<float> > memory_data_layer;
public:
    CaffeImageClassifier();
    virtual void Serialize(ISimpleSerializer* pSerializer);
    virtual void Init(bool firstInit);
    virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
};

void CaffeImageClassifier::Serialize(ISimpleSerializer* pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(NN);
}

void CaffeImageClassifier::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("NN model file", boost::filesystem::path());
        updateParameter("NN weights file", boost::filesystem::path());

    }
}

cv::cuda::GpuMat CaffeImageClassifier::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    std::cout << "beyah" << std::endl;
    if(parameters[0]->changed)
    {
        boost::filesystem::path path = getParameter<boost::filesystem::path>(0)->data;
        if(boost::filesystem::exists(path))
        {
            NN.reset(new caffe::Net<float>(path.string(), caffe::TEST));
            log(Status, "Architecture loaded");
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
            const  std::vector<boost::shared_ptr<caffe::Layer<float>>>& layers = NN->layers();
            std::vector<std::string> layerNames;
            layerNames.reserve(layers.size());
            for(auto layer: layers)
            {
                layerNames.push_back(std::string(layer->type()));
            }
            log(Status, "Weights loaded");
            parameters[1]->changed = false;
            updateParameter("Loaded layers", layerNames);
        }else
        {
            log(Warning, "Weight file does not exist");
        }
    }
    if(NN == nullptr)
    {
        log(Error, "Model not loaded");
        return img;
    }
//    TIME
//    cv::Mat h_img;
//    img.download(h_img,stream);
//    caffe::Datum datum;
//    stream.waitForCompletion();
//    TIME
//    caffe::CVMatToDatum(h_img,&datum);
//    caffe::BlobProto blobProto;
//    blobProto.set_num(1);
//    blobProto.set_channels(h_img.channels());
//    blobProto.set_width(h_img.cols);
//    blobProto.set_height(h_img.rows);
//    const int datumSize = datum.channels() * datum.height() * datum.width();
//    const std::string& data = datum.data();
//    for(int i = 0; i < datumSize; ++i)
//    {
//        blobProto.add_data(uchar(data[i]));
//    }
//    caffe::Blob<float>* blob = new caffe::Blob<float>(1, datum.channels(), datum.height(), datum.width());
//    blob->FromProto(blobProto);
//    std::vector<caffe::Blob<float>*> bottom;
//    bottom.push_back(blob);


//    TIME
//    float loss;
//    const std::vector<caffe::Blob<float>*>& result = NN->Forward(bottom, &loss);
//    const std::vector<float> probs = std::vector<float>(result[1]->cpu_data(), result[1]->cpu_data() + result[1]->count());
//    TIME
//    updateParameter("Probabilities", probs);
//    auto maxvalue = std::max_element(probs.begin(), probs.end());
//    TIME
//    int idx = maxvalue - probs.begin();
//    float score = *maxvalue;
//    updateParameter("Highest scoring class", idx);
//    updateParameter("Highest score", score);
//    TIME
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(CaffeImageClassifier)
