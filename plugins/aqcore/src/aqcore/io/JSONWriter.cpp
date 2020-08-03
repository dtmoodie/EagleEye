#include "JSONWriter.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/params/SubscriberAny.hpp>
#include <MetaObject/params/TParam.hpp>

#include <MetaObject/runtime_reflection/VisitorTraits.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

using namespace aq;
using namespace aq::nodes;

JSONWriter::JSONWriter()
{
    addParam(std::shared_ptr<mo::IParam>(new mo::SubscriberAny("input-0")));
}

JSONWriter::~JSONWriter()
{
    m_ar.reset();
    ofs.close();
}

bool JSONWriter::processImpl()
{
    if (m_ar == nullptr && output_file.string().size())
    {
        ofs.close();
        ofs.open(output_file.c_str(), std::ios::out);
        m_ar.reset(new mo::JSONSaver(ofs));
    }
    if (m_ar)
    {
        auto input_params = getInputs();
        bool found = false;
        mo::FrameNumber fn;
        for (auto param : input_params)
        {
            if (mo::IPublisher* input = param->getPublisher())
            {
                found = true;
                boost::optional<mo::Header> hdr = input->getNewestHeader();
                if (hdr)
                {
                    fn = hdr->frame_number;
                }
            }
        }
        if (found == false)
        {
            return false;
        }

        const std::string name = "frame_" + boost::lexical_cast<std::string>(fn.val);

        std::vector<mo::IDataContainerConstPtr_t> datas;
        for (auto param : input_params)
        {
            mo::IPublisher* publisher = param->getPublisher();
            if (publisher)
            {
                mo::IDataContainerConstPtr_t data = publisher->getData();
                if (data)
                {
                    datas.push_back(std::move(data));
                }
            }
        }

        (*m_ar)(&datas, name);
        return true;
    }
    return false;
}

void JSONWriter::on_output_file_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
{
    ofs.close();
    ofs.open(output_file.c_str(), std::ios::out);
    m_ar.reset(new mo::JSONSaver(ofs));
}

void JSONWriter::on_input_set(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
{
    auto inputs = getInputs();
    int count = 0;
    for (auto input : inputs)
    {
        if (!input->isInputSet())
        {
            return;
        }
        ++count;
    }
    std::shared_ptr<mo::SubscriberAny> ptr =
        std::make_shared<mo::SubscriberAny>("input-" + boost::lexical_cast<std::string>(count));
    addParam(std::static_pointer_cast<mo::IParam>(ptr));
}

MO_REGISTER_CLASS(JSONWriter)

JSONReader::JSONReader()
{
    input = new mo::SubscriberAny("input-0");
    input->setFlags(mo::ParamFlags::kOPTIONAL);
    addParam(std::shared_ptr<mo::IParam>(input));
}

bool JSONReader::processImpl()
{
    if (!ar && boost::filesystem::is_regular_file(input_file))
    {
        ifs.close();
        ifs.open(input_file.c_str(), std::ios::in);
        ar.reset(new cereal::JSONInputArchive(ifs));
    }
    if (!ar)
    {
        return false;
    }

    if (input)
    {
        if (mo::IPublisher* publisher = input->getPublisher())
        {
            boost::optional<mo::Header> hdr = publisher->getNewestHeader();
        }
    }
    return false;
}
