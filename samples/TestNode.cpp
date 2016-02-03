#include <EagleLib/nodes/Node.h>
#include <../RuntimeObjectSystem/ISimpleSerializer.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>


#if _WIN32
	#if _DEBUG
		RUNTIME_COMPILER_LINKLIBRARY("opencv_highgui300d.lib");
		RUNTIME_COMPILER_LINKLIBRARY("opencv_core300d.lib");
		RUNTIME_COMPILER_LINKLIBRARY("opencv_cuda300d.lib");
		RUNTIME_COMPILER_LINKLIBRARY("opencv_cudawarping300d.lib")
		RUNTIME_COMPILER_LINKLIBRARY("opencv_imgproc300d.lib");
		RUNTIME_COMPILER_LINKLIBRARY("opencv_videoio300d.lib");
		RUNTIME_COMPILER_LINKLIBRARY("opencv_imgcodecs300d.lib");

		RUNTIME_COMPILER_LINKLIBRARY("../Debug/RuntimeObjectSystem.lib");
		RUNTIME_COMPILER_LINKLIBRARY("../Debug/RuntimeCompiler.lib");
		RUNTIME_COMPILER_LINKLIBRARY("../Debug/EagleLib.lib");

	#else
		RUNTIME_COMPILER_LINKLIBRARY("opencv_highgui300.lib")
	#endif
#else
	RUNTIME_COMPILER_LINKLIBRARY("-lopencv_highgui")
#endif


namespace EagleLib
{
    namespace Nodes
    {
    
    class TestNode: public Node
    {
    public:
		TestNode();
        virtual ~TestNode()
        {
            std::cout << "Deleting  testNode " << std::endl;
        }
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img)
        {
            
            cv::Mat h_img = cv::imread("/home/dan/Dropbox/Photos/x0ml8.png");
            //cv::Mat h_img = cv::imread("E:/drive/metadata timeoffset.png");
			static int count = 0;
			//std::cout << getParameter<std::string>("OutputString")->data << std::endl;
			testVector.push_back(count);
			std::cout << "Test Vector size: " << testVector.size() << std::endl;
			++count;
            std::cout << h_img.size() << std::endl;
            return cv::cuda::GpuMat(h_img);
           // std::cout << "Test!" << std::endl;
            return img;
        }
        virtual void Init(bool firstInit)
        {
            Node::Init(firstInit);
			if (firstInit)
			{
                ParameteredObject::addParameter<std::string>("OutputString","Default!!!!!!!");
			}
			updateParameter("Output", &testVector)->SetTooltip("Test output vector")->type = Parameters::Parameter::Output;
                
        }
        virtual void Serialize(ISimpleSerializer *pSerializer)
        {
            std::cout << "Running TestNode Serializer" << std::endl;
			SERIALIZE(testVector);
            Node::Serialize(pSerializer);
        }

    private:
		std::vector<int> testVector;
    };

    class TestChildNode: public Node
    {
    public:
		TestChildNode();
        virtual ~TestChildNode()
        {
            std::cout << "Deleting TestChildNode" << std::endl;
        }
		virtual void Init(bool firstInit)
		{
			Node::Init(firstInit);
			if (firstInit)
			{
				addInputParameter<std::vector<int>>("Test input Vector")->SetTooltip("Test input Vector");
			}
		}
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img)
        {
			cv::cuda::resize(img, img, cv::Size(1000, 1000));
           // std::cout << getParameter<std::string>("Output")->data << std::endl;
			auto ptr = getParameter<std::vector<int>>("Test input Vector")->Data();
			if (ptr)
				std::cout << "Successfully accessing input vector of size: " << ptr->size() << std::endl;

			std::cout << img.size() << std::endl;
            std::stringstream ss;
            ss << "TestChildNodeDisplay: " << img.size();
            cv::imshow(ss.str(), cv::Mat(img));
			cv::waitKey(30);
            std::cout << "ChildNode!" << std::endl;
            return img;
        }

    private:

    };
    }

}
using namespace EagleLib;
using namespace EagleLib::Nodes;

NODE_DEFAULT_CONSTRUCTOR_IMPL(TestNode);
NODE_DEFAULT_CONSTRUCTOR_IMPL(TestChildNode);
