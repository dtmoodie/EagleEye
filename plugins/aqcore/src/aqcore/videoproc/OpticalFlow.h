#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/rcc/external_includes/cv_cudaoptflow.hpp>
#include <Aquila/rcc/external_includes/cv_video.hpp>
#include <Aquila/types/Stamped.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/utilities/cuda/CudaUtils.hpp>
#include <Aquila/nodes/NodeContextSwitch.hpp>

#include <MetaObject/params/TMultiInput.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
namespace nodes
{

class IPyrOpticalFlow : public Node
{
  public:
    IPyrOpticalFlow();
    MO_DERIVE(IPyrOpticalFlow, Node)
        INPUT(SyncedMemory, input, nullptr)
        OPTIONAL_INPUT(std::vector<cv::cuda::GpuMat>, image_pyramid, nullptr)
        PARAM(int, window_size, 13)
        PARAM(int, iterations, 30)
        PARAM(int, pyramid_levels, 3)
        PARAM(bool, use_initial_flow, false)
    MO_END

  protected:
    size_t PrepPyramid();
    void build_pyramid(std::vector<cv::cuda::GpuMat>& pyramid);
    TS<std::vector<cv::cuda::GpuMat>> prevGreyImg;
    std::vector<cv::cuda::GpuMat> greyImg;
};

class DensePyrLKOpticalFlow : public IPyrOpticalFlow
{
  public:
    MO_DERIVE(DensePyrLKOpticalFlow, IPyrOpticalFlow)
        OUTPUT(SyncedMemory, flow_field, SyncedMemory())
    MO_END
    bool processImpl();

  protected:
    cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> opt_flow;
};

class SparsePyrLKOpticalFlow : public IPyrOpticalFlow
{
  public:
    MO_DERIVE(SparsePyrLKOpticalFlow, IPyrOpticalFlow)
        INPUT(SyncedMemory, input_points, nullptr)
        APPEND_FLAGS(input_points, mo::ParamFlags::RequestBuffered_e)
        OUTPUT(SyncedMemory, tracked_points, SyncedMemory())
        OUTPUT(SyncedMemory, status, SyncedMemory())
        OUTPUT(SyncedMemory, error, SyncedMemory())
    MO_END

  protected:
    bool processImpl();
    cv::cuda::GpuMat prev_key_points;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optFlow;
};

class PyrLKLandmarkTracker: public Node
{
public:
    MO_DERIVE(PyrLKLandmarkTracker, Node)
        INPUT(SyncedMemory, input, nullptr)
        INPUT(LandmarkDetectionSet, detections, nullptr)
        APPEND_FLAGS(detections, mo::ParamFlags::RequestBuffered_e)

        PARAM(int, window_size, 13)
        PARAM(int, iterations, 30)
        PARAM(int, pyramid_levels, 3)

        OUTPUT(LandmarkDetectionSet, output, {})
    MO_END;

    template<class CTX>
    bool processImpl(CTX* ctx);
protected:
    virtual bool processImpl() override;

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_opt_flow;
    TS<SyncedMemory> m_prev_pyramid;

};

} // namespace aq::nodes
} // namespace aq
