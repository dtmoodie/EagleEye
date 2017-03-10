#include "MXnetExport.hpp"
#include <Aquila/Nodes/Node.h>
#include <MetaObject/Parameters/Types.hpp>
#define DMLC_USE_CXX11 1
#define MXNET_USE_CUDA 1
#define MSHADOW_USE_CBLAS 1
#define MSHADOW_USE_CUDA 1
#define MSHADOW_USE_CUDNN 1
#include <dmlc/base.h>
#include <dmlc/memory_io.h>

#include <mxnet/c_predict_api.h>
#include <mxnet/symbolic.h>
#include <mxnet/ndarray.h>

namespace EagleLib
{
    namespace Nodes
    {
        class MXnet_EXPORT MXNet: public Node
        {
        public:
            MO_DERIVE(MXNet, Node)
                PARAM(mo::ReadFile, model_file, mo::ReadFile())
                PARAM(mo::ReadFile, weight_file, mo::ReadFile())
            MO_END;
        protected:
            bool ProcessImpl();
            std::unique_ptr<mxnet::Executor> exec;
            std::vector<mxnet::TShape> out_shapes;
            std::unordered_map<std::string, size_t> key2arg;
            std::vector<mxnet::NDArray> arg_arrays;
            std::vector<mxnet::NDArray> aux_arrays;
            std::vector<mxnet::NDArray> out_arrays;
        };
        class MXnet_EXPORT MXNetC: public Node
        {
        public:
            MO_DERIVE(MXNetC, Node)
                INPUT(SyncedMemory, input, nullptr)
                PARAM(mo::ReadFile, model_file, mo::ReadFile())
                PARAM(mo::ReadFile, weight_file, mo::ReadFile())
                PARAM(int, width, 224)
                PARAM(int, height, 224)
            MO_END;
        protected:
            bool ProcessImpl();
        private:
            PredictorHandle handle;
        };
    }
}
