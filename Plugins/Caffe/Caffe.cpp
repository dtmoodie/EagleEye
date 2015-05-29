#include "Caffe.h"
#include "caffe/caffe.hpp"

RUNTIME_COMPILER_LINKLIBRARY("-lcaffe")

using namespace EagleLib;

IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}



class CaffeImageClassifier: public Node
{
    boost::shared_ptr<caffe::Net<float>> NN;
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
        }
    }
    if(parameters[1]->changed && NN)
    {
        boost::filesystem::path path = getParameter<boost::filesystem::path>(1)->data;
        if(boost::filesystem::exists(path))
        {
            NN->CopyTrainedLayersFrom(path.string());
        }
    }
    if(NN == nullptr)
    {
        log(Error, "Model not loaded");
        return img;
    }
    cv::cuda::HostMem h_img;
    img.download(h_img,stream);
    caffe::Datum datum;
    stream.waitForCompletion();
    caffe::CVMatToDatum(h_img.createMatHeader(),&datum);

    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(CaffeImageClassifier)
