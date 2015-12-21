#include "DataStreamManager.h"
#include <opencv2/core.hpp>
using namespace EagleLib;

DataStream::DataStream()
{
    stream_id = 0;
}

DataStream::~DataStream()
{

}
size_t DataStream::get_stream_id()
{
    return stream_id;
}
DataStreamManager* DataStreamManager::instance()
{
    static DataStreamManager* inst;
    if (inst == nullptr)
        inst = new DataStreamManager();
    return inst;
}

DataStreamManager::DataStreamManager()
{

}

DataStreamManager::~DataStreamManager()
{

}
std::shared_ptr<DataStream> DataStreamManager::create_stream()
{
    std::shared_ptr<DataStream> stream(new DataStream);
    stream->stream_id = streams.size();
    streams.push_back(stream);
    return stream;
}
std::shared_ptr<DataStream> DataStreamManager::get_stream(size_t id)
{
    CV_Assert(id < streams.size());
    return streams[id];
}