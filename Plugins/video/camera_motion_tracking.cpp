#include "camera_motion_tracking.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
using namespace aq;
using namespace aq::Nodes;


std::vector<std::vector<std::string>> track_camera_motion::GetParentalDependencies()
{
    std::vector<std::vector<std::string>> output;
    output.push_back(std::vector<std::string>({ "GoodFeaturesToTrack", "FastFeatureDetector", "ORBFeatureDetector" }));
    output.push_back(std::vector<std::string>({ "SparsePyrLKOpticalFlow" }));
    return output;
}

bool track_camera_motion::processImpl()
{
    return false;
}

MO_REGISTER_CLASS(track_camera_motion);
