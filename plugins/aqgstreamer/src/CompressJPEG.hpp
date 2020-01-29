#pragma once
#include "sinks.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/SyncedMemory.hpp>
namespace aqgstreamer
{
class CompressImage : virtual public aq::nodes::gstreamer_sink_base, virtual public aq::gstreamer_src_base
{
  public:
    static constexpr const int32_t jpg = aq::types::CompressedImage::Encoding::jpg;
    static constexpr const int32_t png = aq::types::CompressedImage::Encoding::png;
    CompressImage();
    ~CompressImage();
    MO_DERIVE(CompressImage, aq::nodes::gstreamer_sink_base)
        INPUT(aq::SyncedMemory, input, nullptr)

        PARAM(int, quality, 90)
        PARAM(bool, use_hardware_accel, true)
        ENUM_PARAM(encoding, jpg, png)

        OUTPUT(aq::types::CompressedImage, output, {})

    MO_END

    virtual bool create_pipeline(const std::string& pipeline_) final;

  protected:
    virtual GstFlowReturn on_pull() override;
    virtual bool processImpl() override;

    std::shared_ptr<mo::Context> m_gstreamer_context;
};
}
