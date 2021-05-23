#pragma once
#ifdef HAVE_GSTREAMER

#include "gstreamer.h"

namespace aq
{
namespace nodes
{

class aqframegrabbers_EXPORT GrabberHTML : public GrabberGstreamer
{
  public:
    MO_DERIVE(GrabberHTML, GrabberGstreamer)

    MO_END;
    virtual bool loadData(const std::string& file_path) override;
    static int canLoad(const std::string& document);
};
}
}
#endif // HAVE_GSTREAMER
