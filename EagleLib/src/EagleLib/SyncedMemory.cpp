#include "EagleLib/SyncedMemory.h"
#include <EagleLib/utilities/GpuMatAllocators.h>
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include <EagleLib/IO/cvMat.hpp>
#include <EagleLib/SyncedMemoryInl.hpp>
#include <MetaObject/Logging/Log.hpp>

#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/Buffers/NNStreamBuffer.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <boost/lexical_cast.hpp>

INSTANTIATE_META_PARAMETER(EagleLib::SyncedMemory);
INSTANTIATE_META_PARAMETER(std::vector<EagleLib::SyncedMemory>);
INSTANTIATE_META_PARAMETER(cv::Mat);
INSTANTIATE_META_PARAMETER(std::vector<cv::Mat>);


using namespace EagleLib;

SyncedMemory::SyncedMemory():
    _pimpl(new impl)
{

}

SyncedMemory::SyncedMemory(const cv::Mat& h_mat):
    _pimpl(new impl)
{
    _pimpl->h_data.resize(1, h_mat);
    _pimpl->d_data.resize(1);
    _pimpl->sync_flags.resize(1, HOST_UPDATED);
}

SyncedMemory::SyncedMemory(const cv::cuda::GpuMat& d_mat):
    _pimpl(new impl)
{
    _pimpl->h_data.resize(1);
    _pimpl->d_data.resize(1, d_mat);
    _pimpl->sync_flags.resize(1, DEVICE_UPDATED);
}

SyncedMemory::SyncedMemory(const cv::Mat& h_mat, const cv::cuda::GpuMat& d_mat):
    _pimpl(new impl)
{
    _pimpl->h_data.resize(1, h_mat);
    _pimpl->d_data.resize(1, d_mat);
    _pimpl->sync_flags.resize(1, SYNCED);
}

SyncedMemory::SyncedMemory(const std::vector<cv::cuda::GpuMat> & d_mats):
    _pimpl(new impl)
{
    _pimpl->d_data = d_mats;
    _pimpl->sync_flags.resize(d_mats.size(), DEVICE_UPDATED);
}

SyncedMemory::SyncedMemory(const std::vector<cv::Mat>& h_mats):
    _pimpl(new impl)
{
    _pimpl->h_data = h_mats;
    _pimpl->sync_flags.resize(h_mats.size(), HOST_UPDATED);
}

SyncedMemory::SyncedMemory(const std::vector<cv::Mat>& h_mat, const std::vector<cv::cuda::GpuMat>& d_mat, SYNC_STATE state):
    _pimpl(new impl)
{
    CV_Assert(h_mat.size() == d_mat.size());
    _pimpl->sync_flags.resize(h_mat.size(), state);
    _pimpl->h_data = h_mat;
    _pimpl->d_data = d_mat;
}
SyncedMemory::SyncedMemory(const std::vector<cv::Mat>& h_mat, const std::vector<cv::cuda::GpuMat>& d_mat, const std::vector<SYNC_STATE> state):
    _pimpl(new impl)
{
    CV_Assert(h_mat.size() == d_mat.size());
    for(int i = 0; i < h_mat.size(); ++i)
    {
        /*CV_Assert(h_mat[i].empty() == d_mat[i].empty());
        CV_Assert(h_mat[i].size() == d_mat[i].size());
        CV_Assert(h_mat[i].type() == d_mat[i].type());*/
    }
    _pimpl->sync_flags = state;
    _pimpl->h_data = h_mat;
    _pimpl->d_data = d_mat;
}

SyncedMemory::SyncedMemory(cv::MatAllocator* cpu_allocator, cv::cuda::GpuMat::Allocator* gpu_allocator):
    _pimpl(new impl)
{
    _pimpl->h_data = std::vector<cv::Mat>(1, cv::Mat());
    _pimpl->d_data = std::vector<cv::cuda::GpuMat>(1, cv::cuda::GpuMat(gpu_allocator));
    _pimpl->sync_flags = std::vector<SYNC_STATE>(1, SYNCED);
    _pimpl->h_data[0].allocator = cpu_allocator;
}

const cv::Mat&                
SyncedMemory::GetMat(cv::cuda::Stream& stream, int index) const
{
    if(index < 0 || index >= std::max(_pimpl->h_data.size(), _pimpl->d_data.size()))
        THROW(debug) << "Index (" << index << ") out of range [0," << std::max(_pimpl->h_data.size(), _pimpl->d_data.size()) << "]";
    if(_pimpl->sync_flags[index] == DO_NOT_SYNC)
        return _pimpl->h_data[index];
    if (_pimpl->sync_flags[index] == DEVICE_UPDATED)
    {
        _pimpl->d_data[index].download(_pimpl->h_data[index], stream);
        _pimpl->sync_flags[index] = SYNCED;
    }
    return _pimpl->h_data[index];
}

cv::Mat&                    
SyncedMemory::GetMatMutable(cv::cuda::Stream& stream, int index)
{
    if (index < 0 || index >= std::max(_pimpl->h_data.size(), _pimpl->d_data.size()))
        THROW(debug) << "Index (" << index << ") out of range [0," << std::max(_pimpl->h_data.size(), _pimpl->d_data.size()) << "]";
    if(_pimpl->sync_flags[index] == DO_NOT_SYNC)
        return _pimpl->h_data[index];
    if (_pimpl->sync_flags[index] == DEVICE_UPDATED)
        _pimpl->d_data[index].download(_pimpl->h_data[index], stream);
    _pimpl->sync_flags[index] = HOST_UPDATED;
    return _pimpl->h_data[index];
}

const cv::cuda::GpuMat&        
SyncedMemory::GetGpuMat(cv::cuda::Stream& stream, int index) const
{
    if (index < 0 || index >= std::max(_pimpl->h_data.size(), _pimpl->d_data.size()))
        THROW(debug) << "Index (" << index << ") out of range [0," << std::max(_pimpl->h_data.size(), _pimpl->d_data.size()) << "]";
    if (_pimpl->sync_flags[index] == DO_NOT_SYNC)
        return _pimpl->d_data[index];
    if (_pimpl->sync_flags[index] == HOST_UPDATED)
    {
        _pimpl->d_data[index].upload(_pimpl->h_data[index], stream);
        _pimpl->sync_flags[index] = SYNCED;
    }
    if(_pimpl->d_data.empty() && !_pimpl->h_data.empty())
    {
        // Something went wrong, probably set incorrectly by external program
    }
    return _pimpl->d_data[index];
}

cv::cuda::GpuMat&            
SyncedMemory::GetGpuMatMutable(cv::cuda::Stream& stream, int index)
{
    if (index < 0 || index >= std::max(_pimpl->h_data.size(), _pimpl->d_data.size()))
        THROW(debug) << "Index (" << index << ") out of range [0," << std::max(_pimpl->h_data.size(), _pimpl->d_data.size()) << "]";
    if (_pimpl->sync_flags[index] == DO_NOT_SYNC)
        return _pimpl->d_data[index];
    if (_pimpl->sync_flags[index] == HOST_UPDATED)
        _pimpl->d_data[index].upload(_pimpl->h_data[index], stream);
    _pimpl->sync_flags[index] = DEVICE_UPDATED;
    return _pimpl->d_data[index];
}

const std::vector<cv::Mat>&
SyncedMemory::GetMatVec(cv::cuda::Stream& stream) const
{
    if (_pimpl->sync_flags.size() && _pimpl->sync_flags[0] == DO_NOT_SYNC)
        return _pimpl->h_data;
    for (int i = 0; i < _pimpl->sync_flags.size(); ++i)
    {
        if (_pimpl->sync_flags[i] == DEVICE_UPDATED)
            _pimpl->d_data[i].download(_pimpl->h_data[i], stream);
    }
    return _pimpl->h_data;
}

std::vector<cv::Mat>&
SyncedMemory::GetMatVecMutable(cv::cuda::Stream& stream)
{
    if (_pimpl->sync_flags.size() && _pimpl->sync_flags[0] == DO_NOT_SYNC)
        return _pimpl->h_data;
    for (int i = 0; i < _pimpl->sync_flags.size(); ++i)
    {
        if (_pimpl->sync_flags[i] == DEVICE_UPDATED)
            _pimpl->d_data[i].download(_pimpl->h_data[i], stream);
        _pimpl->sync_flags[i] = HOST_UPDATED;
    }
    
    return _pimpl->h_data;
}

const std::vector<cv::cuda::GpuMat>&
SyncedMemory::GetGpuMatVec(cv::cuda::Stream& stream) const
{
    if (_pimpl->sync_flags.size() && _pimpl->sync_flags[0] == DO_NOT_SYNC)
        return _pimpl->d_data;
    for (int i = 0; i < _pimpl->sync_flags.size(); ++i)
    {
        if (_pimpl->sync_flags[i] == HOST_UPDATED)
            _pimpl->d_data[i].upload(_pimpl->h_data[i], stream);
    }
    return _pimpl->d_data;
}

std::vector<cv::cuda::GpuMat>&
SyncedMemory::GetGpuMatVecMutable(cv::cuda::Stream& stream)
{
    if (_pimpl->sync_flags.size() && _pimpl->sync_flags[0] == DO_NOT_SYNC)
        return _pimpl->d_data;
    for (int i = 0; i < _pimpl->sync_flags.size(); ++i)
    {
        if (_pimpl->sync_flags[i] == HOST_UPDATED)
            _pimpl->d_data[i].upload(_pimpl->h_data[i], stream);
        _pimpl->sync_flags[i] = DEVICE_UPDATED;
    }
    return _pimpl->d_data;
}
void SyncedMemory::ResizeNumMats(int new_size)
{
    _pimpl->h_data.resize(new_size);
    _pimpl->d_data.resize(new_size);
    _pimpl->sync_flags.resize(new_size);
}

SyncedMemory SyncedMemory::clone(cv::cuda::Stream& stream)
{
    SyncedMemory output;
    output._pimpl->h_data.resize(_pimpl->h_data.size());
    output._pimpl->d_data.resize(_pimpl->d_data.size());
    for(int i = 0; i < _pimpl->h_data.size(); ++i)
    {
        output._pimpl->h_data[i] = _pimpl->h_data[i].clone();
        _pimpl->d_data[i].copyTo(output._pimpl->d_data[i], stream);
    }
    output._pimpl->sync_flags = _pimpl->sync_flags;
    return output;
}
int SyncedMemory::GetNumMats() const
{
    if(_pimpl->h_data.size() == 0)
        return _pimpl->d_data.size();
    return _pimpl->h_data.size();
}
bool SyncedMemory::empty() const
{
    if(_pimpl->h_data.size() && _pimpl->d_data.size())
        return _pimpl->h_data[0].empty() && _pimpl->d_data[0].empty();
    if(_pimpl->h_data.size() && _pimpl->d_data.size() == 0)
        return _pimpl->h_data[0].empty();
    if(_pimpl->d_data.size() && _pimpl->h_data.size() == 0)
        return _pimpl->d_data[0].empty();
    return true;
}

bool SyncedMemory::Clone(cv::Mat& dest, cv::cuda::Stream& stream, int idx) const
{
    CV_Assert(_pimpl->sync_flags.size() > idx);
    if(_pimpl->sync_flags[idx] < DEVICE_UPDATED || _pimpl->sync_flags[idx] == DO_NOT_SYNC)
    {
        _pimpl->h_data[idx].copyTo(dest);
        return false;
    }else
    {
        _pimpl->d_data[idx].download(dest, stream);
        return true;
    }
}

bool SyncedMemory::Clone(cv::cuda::GpuMat& dest, cv::cuda::Stream& stream, int idx) const
{
    CV_Assert(_pimpl->sync_flags.size() > idx);
    if(_pimpl->sync_flags[idx] < DEVICE_UPDATED || _pimpl->sync_flags[idx] == DO_NOT_SYNC)
    {
        dest.upload(_pimpl->h_data[idx], stream);
        if(_pimpl->sync_flags[idx] != DO_NOT_SYNC)
        {
            dest.copyTo(_pimpl->d_data[idx], stream);
            _pimpl->sync_flags[idx] = SYNCED;
        }
        return true;
    }else
    {
        _pimpl->d_data[idx].copyTo(dest, stream);
        return true;
    }
}

void SyncedMemory::Synchronize(cv::cuda::Stream& stream) const
{
    for(int i = 0; i < _pimpl->h_data.size(); ++i)
    {
        if (_pimpl->sync_flags[i] == DO_NOT_SYNC)
            continue;
        if(_pimpl->sync_flags[i] == HOST_UPDATED)
            _pimpl->d_data[i].upload(_pimpl->h_data[i], stream);
        else if(_pimpl->sync_flags[i] == DEVICE_UPDATED)
            _pimpl->d_data[i].download(_pimpl->h_data[i], stream);
        _pimpl->sync_flags[i] = SYNCED;
    }
}


void SyncedMemory::ReleaseGpu(cv::cuda::Stream& stream)
{
    for(int i = 0; i < _pimpl->d_data.size(); ++i)
    {
        if(_pimpl->sync_flags[i] == DEVICE_UPDATED)
            _pimpl->d_data[i].download(_pimpl->h_data[i], stream);
    }
    if(dynamic_cast<DelayedDeallocator*>(cv::cuda::GpuMat::defaultAllocator()))
    {
        EagleLib::cuda::enqueue_callback([this]
        {
            for(int i = 0; i < _pimpl->d_data.size(); ++i)
            {
                _pimpl->d_data[i].release();
            }
        }, stream);
    }else
    {
        for(int i = 0; i < _pimpl->d_data.size(); ++i)
        {
            _pimpl->d_data[i].release();
        }
    }
}

cv::Size SyncedMemory::GetSize() const
{
    if(_pimpl->d_data.empty() || _pimpl->h_data.empty())
        return cv::Size();
    cv::Size output;
    if(_pimpl->d_data[0].empty())
        output = _pimpl->h_data[0].size();
    else
        output = _pimpl->d_data[0].size();
    return output;
}

int SyncedMemory::GetChannels() const
{
    if(_pimpl->d_data.empty() && _pimpl->h_data.empty())
        return 0;
    if(_pimpl->d_data.size() == 1 && !_pimpl->d_data[0].empty())
        return _pimpl->d_data[0].channels();
    if(_pimpl->h_data.size() == 1 && !_pimpl->h_data[0].empty())
        return _pimpl->h_data[0].channels();
    if(_pimpl->d_data.size() > 1 && _pimpl->d_data[0].channels() == 1)
        return _pimpl->d_data.size();
    if(_pimpl->h_data.size() > 1 && _pimpl->h_data[0].channels() == 1)
        return _pimpl->h_data.size();
    return 0;
}

std::vector<int> SyncedMemory::GetShape() const
{
    std::vector<int> output;
    output.push_back(std::max(_pimpl->d_data.size(), _pimpl->h_data.size()));
    if(_pimpl->d_data.empty() && _pimpl->h_data.empty())
        return output;
    if(_pimpl->d_data.empty())
    {
        output.push_back(_pimpl->h_data[0].rows);
        output.push_back(_pimpl->h_data[0].cols);
        output.push_back(_pimpl->h_data[0].channels());
    }else
    {
        if(_pimpl->h_data.empty())
        {
            output.push_back(_pimpl->d_data[0].rows);
            output.push_back(_pimpl->d_data[0].cols);
            output.push_back(_pimpl->d_data[0].channels());
        }else
        {
            output.push_back(std::max(_pimpl->d_data[0].rows, _pimpl->h_data[0].rows));
            output.push_back(std::max(_pimpl->d_data[0].cols, _pimpl->h_data[0].cols));
            output.push_back(std::max(_pimpl->d_data[0].channels(), _pimpl->h_data[0].channels()));
        }
    }
    return output;
}
int SyncedMemory::GetDepth() const
{
    CV_Assert(_pimpl->d_data.size() || _pimpl->h_data.size());
    if(_pimpl->d_data.size())
        return _pimpl->d_data[0].depth();
    return _pimpl->h_data[0].depth();
}

int SyncedMemory::GetType() const
{
    CV_Assert(_pimpl->d_data.size() || _pimpl->h_data.size());
    if (_pimpl->d_data.size())
        return _pimpl->d_data[0].type();
    return _pimpl->h_data[0].type();
}
int SyncedMemory::GetDim(int dim) const
{
    if(dim == 0)
        return _pimpl->d_data.size();
    if(dim == 1 && _pimpl->d_data.size())
        return _pimpl->d_data[0].rows;
    if(dim == 2 && _pimpl->d_data.size())
        return _pimpl->d_data[0].cols;
    if(dim == 3 && _pimpl->d_data.size())
        return _pimpl->d_data[0].channels();
    return 0;
}
SyncedMemory::SYNC_STATE SyncedMemory::GetSyncState(int index) const
{
    CV_Assert(index < _pimpl->sync_flags.size() && index >= 0);
    return _pimpl->sync_flags[index];
}
mo::Context* SyncedMemory::GetContext() const
{
    return _pimpl->_ctx;
}

void SyncedMemory::SetContext(mo::Context* ctx)
{
    _pimpl->_ctx = ctx;
}

using namespace mo;



TypedInputParameterPtr<SyncedMemory>::TypedInputParameterPtr(const std::string& name,
                                                             const SyncedMemory** userVar_, Context* ctx) :
        userVar(userVar_),
        ITypedInputParameter<SyncedMemory>(name, ctx),
        IParameter(name, Input_e, -1, ctx),
        ITypedParameter<SyncedMemory>(name, Input_e, -1, ctx)
{
}

bool TypedInputParameterPtr<SyncedMemory>::SetInput(std::shared_ptr<IParameter> param)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(ITypedInputParameter<SyncedMemory>::SetInput(param))
    {
        if(userVar)
        {
            if(this->input)
                *userVar = this->input->GetDataPtr();
            if(this->shared_input)
                *userVar = this->shared_input->GetDataPtr();
        }
        return true;
    }
    return false;
}

bool TypedInputParameterPtr<SyncedMemory>::SetInput(IParameter* param)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(ITypedInputParameter<SyncedMemory>::SetInput(param))
    {
        if(userVar)
        {
            if(this->input)
                *userVar = this->input->GetDataPtr();
            if(this->shared_input)
                *userVar = this->shared_input->GetDataPtr();
        }
        return true;
    }
    return false;
}

void TypedInputParameterPtr<SyncedMemory>::SetUserDataPtr(const SyncedMemory** user_var_)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    userVar = user_var_;
}

void TypedInputParameterPtr<SyncedMemory>::onInputUpdate(Context* ctx, IParameter* param)
{
    if(this->input)
    {
        this->Commit(this->input->GetTimestamp(), ctx);
        if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || (ctx == nullptr &&  this->_ctx == nullptr))
        {
            if(userVar)
                *userVar = this->input->GetDataPtr();
        }
    }else if(this->shared_input)
    {
        this->Commit(this->shared_input->GetTimestamp(), ctx);
        if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || (ctx == nullptr &&  this->_ctx == nullptr))
        {
            if(userVar)
                *userVar = this->shared_input->GetDataPtr();
        }
    }
}

bool TypedInputParameterPtr<SyncedMemory>::GetInput(long long ts)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(userVar)
    {
        if(this->shared_input)
        {
            *userVar = this->shared_input->GetDataPtr(ts, this->_ctx);
            if(*userVar != nullptr)
            {
                return !(*userVar)->empty();
            }
        }
        if(this->input)
        {
            *userVar = this->input->GetDataPtr(ts, this->_ctx);
            if(*userVar != nullptr)
            {
                return !(*userVar)->empty();
            }
        }
    }
    return false;
}

void TypedInputParameterPtr<SyncedMemory>::onInputDelete(IParameter const* param)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(param == this->input || param == this->shared_input.get())
    {
        this->shared_input.reset();
        this->input = nullptr;
    }
}

TEMPLATE_EXTERN template class EAGLE_EXPORTS TypedInputParameterPtr<SyncedMemory>;