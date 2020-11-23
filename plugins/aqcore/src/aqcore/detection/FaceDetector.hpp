#pragma once
#include "HaarDetector.hpp"
#include <aqcore/IDetector.hpp>

namespace aqcore
{

    class FaceDetector : public TInterface<FaceDetector, IImageDetector>
    {
      public:
        template <class T>
        using InterfaceHelper = IImageDetector::InterfaceHelper<T>;

        MO_DERIVE(FaceDetector, IImageDetector)
            MO_STATIC_SLOT(aq::nodes::Node::Ptr, create, std::string)
            MO_STATIC_SLOT(std::vector<std::string>, list)
        MO_END;

      protected:
        void createLabels();
    };

    class HaarFaceDetector : virtual public HaarDetector
    {
      public:
        MO_DERIVE(HaarFaceDetector, HaarDetector)

        MO_END;

      protected:
        bool processImpl() override;
    };

} // namespace aqcore
