#pragma once
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <aqcore/aqcore_export.hpp>
#include <opencv2/imgproc.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
namespace nodes
{
class aqcore_EXPORT ConvertToGrey : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(ConvertToGrey, ::aq::nodes::Node)
        INPUT(SyncedMemory, input, nullptr)
        OUTPUT(SyncedMemory, grey, SyncedMemory())
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class aqcore_EXPORT ConvertToHSV : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(ConvertToHSV, ::aq::nodes::Node)
        INPUT(SyncedMemory, input_image, nullptr)
        OUTPUT(SyncedMemory, hsv_image, SyncedMemory())
    MO_END

    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};
class ConvertToLab : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(ConvertToLab, ::aq::nodes::Node)
        INPUT(SyncedMemory, input_image, nullptr)
        OUTPUT(SyncedMemory, lab_image, SyncedMemory())
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class ConvertTo : public aq::nodes::Node
{
  public:
    MO_DERIVE(ConvertTo, Node)
        INPUT(SyncedMemory, input, nullptr)
        OUTPUT(SyncedMemory, output, SyncedMemory())
        ENUM_PARAM(datatype, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
        PARAM(float, alpha, 1.0)
        PARAM(float, beta, 0.0)
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class aqcore_EXPORT ConvertColorspace : public Node
{

  public:
    MO_DERIVE(ConvertColorspace, ::aq::nodes::Node)
        INPUT(SyncedMemory, input_image, nullptr)
        ENUM_PARAM(conversion_code, cv::COLOR_BGR2HSV)
        OUTPUT(SyncedMemory, output_image, SyncedMemory())
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class aqcore_EXPORT Magnitude : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(Magnitude, ::aq::nodes::Node)
        INPUT(SyncedMemory, input, nullptr)
        OUTPUT(SyncedMemory, output, SyncedMemory())
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class aqcore_EXPORT SplitChannels : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(SplitChannels, aq::nodes::Node)
        INPUT(SyncedMemory, input, nullptr)
        OUTPUT(SyncedMemory, output, SyncedMemory())
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class aqcore_EXPORT ConvertDataType : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(ConvertDataType, ::aq::nodes::Node)
        INPUT(SyncedMemory, input, nullptr)
        OUTPUT(SyncedMemory, output, SyncedMemory())
        ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
        PARAM(double, alpha, 255.0)
        PARAM(double, beta, 0.0)
        PARAM(bool, continuous, false)
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class aqcore_EXPORT MergeChannels : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(MergeChannels, ::aq::nodes::Node)
        INPUT(SyncedMemory, input_image, nullptr)
        OUTPUT(SyncedMemory, merged_image, SyncedMemory())
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

class aqcore_EXPORT Reshape : public ::aq::nodes::Node
{
  public:
    MO_DERIVE(Reshape, aq::nodes::Node)
        INPUT(SyncedMemory, input_image, nullptr)
        OUTPUT(SyncedMemory, reshaped_image, SyncedMemory())
        PARAM(int, channels, 0)
        PARAM(int, rows, 0)
    MO_END
    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    virtual bool processImpl() override;
};

} // namespace aq::nodes
} // namespace aq
