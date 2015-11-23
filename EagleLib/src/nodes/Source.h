#pragma once

#include "Node.h"


namespace EagleLib
{
	class EAGLE_EXPORTS SourceNodeBase: public Node
	{
	public:
		SourceNodeBase();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& input, cv::cuda::Stream& stream);
		virtual bool SkipEmpty() const;
	protected:
		virtual bool get_previous_frame(cv::cuda::GpuMat& next_frame_out, cv::cuda::Stream& stream) = 0;
		virtual bool get_next_frame(cv::cuda::GpuMat& next_frame_out, cv::cuda::Stream& stream) = 0;
		virtual bool get_current_frame(cv::cuda::GpuMat& current_frame_out, cv::cuda::Stream& stream) = 0;
		virtual double get_current_timestamp() = 0;
		virtual size_t get_current_frame_number() = 0;
		virtual bool get_frame(size_t index, cv::cuda::GpuMat& desired_frame_out, cv::cuda::Stream& stream) = 0;
		virtual size_t get_num_frames() = 0;

		virtual void on_playback_state_change(PlaybackState val);
	private:
		PlaybackState current_state;
	};
}