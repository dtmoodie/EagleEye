#include "Blur.hpp"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/core/IAsyncStream.hpp>

#include <opencv2/imgproc.hpp>
namespace aq
{
    namespace nodes
    {
        bool GaussianBlur::processImpl(mo::IAsyncStream& stream)
        {
            cv::Mat blurred;
            cv::Mat in = input->getMat(&stream);
            cv::GaussianBlur(in, blurred, {kerenl_size, kerenl_size}, sigma, sigma);
            output.publish(blurred, mo::tags::param = &input_param);
            return true;
        }

        bool GaussianBlur::processImpl()
        {
            mo::IAsyncStreamPtr_t stream = this->getStream();
            nodeStreamSwitch(this, *stream);
            return true;
        }
    } // namespace nodes
} // namespace aq
using namespace aq::nodes;
MO_REGISTER_CLASS(GaussianBlur)
