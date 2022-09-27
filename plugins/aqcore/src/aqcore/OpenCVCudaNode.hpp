#ifndef AQCORE_OPENCV_CUDA_NODE_HPP
#define AQCORE_OPENCV_CUDA_NODE_HPP
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/CVStream.hpp>

#include <aqcore_export.hpp>
namespace cv
{
    namespace cuda
    {
        class Stream;
    }
} // namespace cv

namespace aqcore
{

    aqcore_EXPORT std::unique_ptr<cv::cuda::Stream> getCVStream(const mo::IAsyncStreamPtr_t& stream);

    struct aqcore_EXPORT OpenCVCudaNode : virtual aq::nodes::Node
    {
        MO_DERIVE(OpenCVCudaNode, aq::nodes::Node)
        MO_END;

        void setStream(const mo::IAsyncStreamPtr_t& stream) override;

        cv::cuda::Stream* getCVStream();

        /**
         * @brief processImpl empty implementation, override in your subclass
         * @param stream
         * @return
         */
        virtual bool processImpl(aq::CVStream& stream);

        bool processImpl() override;
        /**
         * @brief processImpl this override checks if the input stream is an aq::CVStream, if it is then it calls the
         * above override.  It does not try to wrap an opencv stream
         * @param stream
         * @return
         */
        bool processImpl(mo::IDeviceStream& stream) override;

        /**
         * @brief processImpl
         * @param stream
         * @return
         */
        bool processImpl(mo::IAsyncStream& stream) override;

      private:
        std::unique_ptr<cv::cuda::Stream> m_cv_stream;
        std::shared_ptr<aq::CVStream> m_stream;
    };
} // namespace aqcore

#endif // AQCORE_OPENCV_CUDA_NODE_HPP
