#pragma once
#include <EagleLib/nodes/Node.h>
#include <EagleLib/ParameteredIObjectImpl.hpp>
namespace EagleLib
{
	namespace Nodes
	{
		class PLUGIN_EXPORTS caffe_input_populator: public Node
		{
		public:
			caffe_input_populator();
			virtual void NodeInit(bool firstInit);
			virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream);
			virtual bool pre_check(const TS<SyncedMemory>& input);
			std::vector<std::pair<int,int>> sample_permutation;

			BEGIN_PARAMS(caffe_input_populator);
				PARAM(bool, shuffle, true);
				PARAM(int, sample_index, 0);
				PARAM(int, blob_index, 0);
				PARAM(std::string, blob_name, "");
			END_PARAMS;

			SIGNALS_BEGIN(caffe_input_populator, Node);
				SLOT_DEF(void, fill_blobs);
				REGISTER_SLOT(fill_blobs);
			SIGNALS_END;
		};
	}
}