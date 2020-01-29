#pragma once
#include <aqcore/IDetector.hpp>
#include "HaarDetector.hpp"

namespace aq
{
namespace nodes
{
class FaceDetector : virtual public TInterface<FaceDetector, IImageDetector>
{
  public:
    template <class T>
    using InterfaceHelper = IImageDetector::InterfaceHelper<T>;

    MO_DERIVE(FaceDetector, IImageDetector)
        MO_STATIC_SLOT(Node::Ptr, create, std::string)
        MO_STATIC_SLOT(std::vector<std::string>, list)
    MO_END;

  protected:
    bool processImpl() override;
    void createLabels();
};
#if MO_OPENCV_HAVE_CUDA == 1
class HaarFaceDetector : virtual public HaarDetector, virtual public FaceDetector
{
  public:
    MO_DERIVE(HaarFaceDetector, HaarDetector)

    MO_END

  protected:
    bool processImpl() override;
};
#endif
}
}
