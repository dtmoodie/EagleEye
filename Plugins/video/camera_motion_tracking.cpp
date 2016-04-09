#include "camera_motion_tracking.hpp"

using namespace EagleLib;
using namespace EagleLib::Nodes;

track_camera_motion::track_camera_motion_info::track_camera_motion_info() :
	NodeInfo("track_camera_motion", { "Video", "Extraction" })
{
}
std::vector<std::vector<std::string>> track_camera_motion::track_camera_motion_info::GetParentalDependencies() const
{
	std::vector<std::vector<std::string>> output;
	output.push_back(std::vector<std::string>({ "GoodFeaturesToTrack", "FastFeatureDetector", "ORBFeatureDetector" }));
	output.push_back(std::vector<std::string>({ "SparsePyrLKOpticalFlow" }));
	return output;
}

TS<SyncedMemory> track_camera_motion::doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
	return input;
}
static track_camera_motion::track_camera_motion_info info;
REGISTERCLASS(track_camera_motion, &info);