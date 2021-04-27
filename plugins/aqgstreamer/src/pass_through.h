#ifndef AQGSTREAMER_PASS_THROUGH_HPP
#define AQGSTREAMER_PASS_THROUGH_HPP
#include <MetaObject/object/MetaObject.hpp>
#include "gstreamer.hpp"

namespace aqgstreamer
{

    class aqgstreamer_EXPORT H264PassThrough : public GstreamerSinkBase
    {
      public:
        MO_DERIVE(H264PassThrough, GstreamerSinkBase)
            PARAM(std::string, pipeline, "");
            PARAM(bool, active, false);
        MO_END;

      protected:
        bool processImpl() override;

      private:
        GstElement* m_valve = nullptr;
        bool m_previously_active = false;
    };
} // namespace aqgstreamer

#endif // AQGSTREAMER_PASS_THROUGH_HPP
