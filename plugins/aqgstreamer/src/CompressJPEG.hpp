#ifndef AQGSTREAMER_COMPRESS_JPEG_HPP
#define AQGSTREAMER_COMPRESS_JPEG_HPP
#include "sinks.hpp"

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aqgstreamer
{

    class CompressImage : virtual public GstreamerSinkBase, virtual public GstreamerSrcBase
    {
      public:
        static constexpr const aq::ImageEncoding jpg = aq::ImageEncoding::JPG;
        static constexpr const aq::ImageEncoding png = aq::ImageEncoding::PNG;

        CompressImage();
        ~CompressImage();

        MO_DERIVE(CompressImage, GstreamerSinkBase)
            INPUT(aq::SyncedImage, input)

            PARAM(int, quality, 90)
            PARAM(bool, use_hardware_accel, true)
            ENUM_PARAM(encoding, jpg, png)

            OUTPUT(aq::CompressedImage, output)

        MO_END;

        bool createPipeline(const std::string& pipeline_) final;

      protected:
        GstFlowReturn onPull() override;
        bool processImpl() override;

        mo::IAsyncStreamPtr_t m_gstreamer_stream;
    };
} // namespace aqgstreamer

#endif // AQGSTREAMER_COMPRESS_JPEG_HPP
