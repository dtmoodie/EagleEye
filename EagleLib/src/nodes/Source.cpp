#include "Source.h"
#include "SystemTable.hpp"
#include "Events.h"
using namespace EagleLib;


SourceNodeBase::SourceNodeBase()
{
	auto table = PerModuleInterface::GetInstance()->GetSystemTable();
	if (table)
	{
		auto signalHandler = table->GetSingleton<ISignalHandler>();
		auto signal = signalHandler->GetSignalSafe<boost::signals2::signal<void(PlaybackState)>>("SetPlaybackState");

		RegisterSignalConnection(signal->connect(boost::bind(&SourceNodeBase::on_playback_state_change, this, _1)));
	}
	current_state = PAUSED;
}

void SourceNodeBase::Init(bool firstInit)
{

}

cv::cuda::GpuMat SourceNodeBase::doProcess(cv::cuda::GpuMat& input, cv::cuda::Stream& stream)
{
	cv::cuda::GpuMat output_image;
	switch (current_state)
	{
	case(PLAYING) :
	{
		if (get_next_frame(output_image, stream))
		{
			onUpdate(&stream);
			return output_image;
		}
	}
	case(PAUSED) :
	{
		if (get_current_frame(output_image, stream))
		{
			return output_image;
		}
	}
	case(FAST_FORWARD) :
	{

	}
	case(FAST_BACKWARD) :
	{

	}
	case(STOP) :
	{
		
	}
	} // switch(current_state)

	return cv::cuda::GpuMat();
}
bool SourceNodeBase::SkipEmpty() const
{
	return false;
}
void SourceNodeBase::on_playback_state_change(PlaybackState val)
{
	current_state = val;
}