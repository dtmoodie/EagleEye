#include "nodes/Node.h"
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <vector>
/*  Parallel stack nodes execute all child nodes in parallel on separate threads
 *
 *
 *
 *
 *
*/



namespace EagleLib
{
    class CV_EXPORTS ParallelStack: public Node
    {
    public:
        ParallelStack();
        ~ParallelStack();
        virtual cv::cuda::GpuMat process(cv::cuda::GpuMat img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
//        virtual void registerDisplayCallback(boost::function<void(cv::Mat)>& f);
//        virtual void registerDisplayCallback(boost::function<void(cv::cuda::GpuMat)>& f);
//        virtual void spawnDisplay();
//        virtual void killDisplay();
//        virtual std::string getName();

//        virtual int addChild(boost::shared_ptr<Node> child);
//        virtual void removeChild(boost::shared_ptr<Node> child);
//        virtual void removeChild(int idx);
    private:
        std::vector< boost::shared_ptr<boost::thread> > threads;
    };


}
