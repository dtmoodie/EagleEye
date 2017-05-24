#include "JSONWriter.hpp"
#include <MetaObject/params/TParam.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/serialization/SerializationFactory.hpp>
#include <MetaObject/params/InputParamAny.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
using namespace aq;
using namespace aq::Nodes;

/*class InputParamAny: public mo::InputParam
{
public:
    InputParamAny(const std::string& name):
        _update_slot(std::bind(&InputParamAny::on_param_update, this, std::placeholders::_1, std::placeholders::_2)),
        _delete_slot(std::bind(&InputParamAny::on_param_delete, this, std::placeholders::_1))
    {
        this->setName(name);
        _void_type_info = mo::TypeInfo(typeid(void));
        this->appendFlags(mo::Input_e);
    }
    virtual bool GetInput(long long ts = -1)
    {
        return true;
    }

    // This gets a pointer to the variable that feeds into this input
    virtual IParam* GetInputParam()
    {
        return input;
    }
    virtual bool setInput(std::shared_ptr<mo::IParam> param)
    {
        input = param.get();
        Commit();
        return true;
    }
    virtual bool setInput(mo::IParam* param = nullptr)
    {
        input = param;
        param->registerDeleteNotifier(&_delete_slot);
        param->registerUpdateNotifier(&_update_slot);
        Commit();
        return true;
    }

    virtual bool AcceptsInput(std::weak_ptr<mo::IParam> param) const
    {
        return true;
    }
    virtual bool AcceptsInput(mo::IParam* param) const
    {
        return true;
    }
    virtual bool AcceptsType(mo::TypeInfo type) const
    {
        return true;
    }
    mo::TypeInfo& getTypeInfo() const
    {
        return _void_type_info;
    }
    void on_param_update(mo::Context* ctx, mo::IParam* param)
    {
        Commit(-1, ctx); // Notify owning parent of update
    }
    void on_param_delete(mo::IParam const *)
    {
        input = nullptr;
    }
    IParam* input = nullptr;
    static mo::TypeInfo _void_type_info;
    mo::TSlot<void(mo::Context*, mo::IParam*)> _update_slot;
    mo::TSlot<void(mo::IParam const*)> _delete_slot;
};
mo::TypeInfo InputParamAny::_void_type_info;*/

JSONWriter::JSONWriter()
{
    addParameter(std::shared_ptr<mo::IParam>(new mo::InputParamAny("input-0")));
}
JSONWriter::~JSONWriter()
{
    ar.reset();
    ofs.close();
}
bool JSONWriter::processImpl()
{
    if(ar == nullptr && output_file.string().size())
    {
        ofs.close();
        ofs.open(output_file.c_str(), std::ios::out);
        ar.reset(new cereal::JSONOutputArchive(ofs));
    }
    if(ar)
    {
        auto input_params = getInputs();
        bool found = false;
        size_t fn = 0;
        for(auto param : input_params)
        {
            if(auto input = param->getInputParam())
            {
                found = true;
                fn = input->getFrameNumber();
            }
        }
        if(found == false)
            return false;

        std::string name = "frame_" + boost::lexical_cast<std::string>(fn);
        ar->setNextName(name.c_str());
        ar->startNode();
        for (auto param : input_params)
        {
            auto input_param = param->getInputParam();
            if(input_param)
            {
                auto func = mo::SerializationFactory::instance()->getJsonSerializationFunction(input_param->getTypeInfo());
                if (func)
                {
                    func(input_param, *ar);
                }
            }
        }
        ar->finishNode();
        return true;
    }
    return false;
}

void JSONWriter::on_output_file_modified( mo::Context* ctx, mo::IParam* param)
{
    ofs.close();
    ofs.open(output_file.c_str(), std::ios::out);
    ar.reset(new cereal::JSONOutputArchive(ofs));
}

void JSONWriter::on_input_set(mo::Context* ctx, mo::IParam* param)
{
    auto inputs = getInputs();
    int count= 0;
    for(auto input : inputs)
    {
        if(input->getInputParam() == nullptr)
        {
            return;
        }
        ++count;
    }
    addParameter(std::shared_ptr<mo::IParam>(new mo::InputParamAny("input-" + boost::lexical_cast<std::string>(count))));
}

MO_REGISTER_CLASS(JSONWriter)

JSONReader::JSONReader()
{
    input = new mo::InputParamAny("input-0");
    input->setFlags(mo::Optional_e);
    addParameter(std::shared_ptr<mo::IParam>(input));
}

bool JSONReader::processImpl()
{
    if(!ar && boost::filesystem::is_regular_file(input_file))
    {
        ifs.close();
        ifs.open(input_file.c_str(), std::ios::in);
        ar.reset(new cereal::JSONInputArchive(ifs));
    }
    if(!ar)
        return false;

    if(input)
    {
        if(auto input_param = input->getInputParam())
        {
            input_param->getTimestamp();
        }
    }
    return false;
}
