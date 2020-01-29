#include <MetaObject/object/MetaObjectFactory.hpp>

#include "MXNetOutputParser.hpp"
#include "mxnet-cpp/symbol.h"
#include "mxnet/c_api.h"

namespace aq
{
namespace mxnet
{

std::vector<rcc::shared_ptr<MXNetOutputParser>> MXNetOutputParser::createParsers(const ::mxnet::cpp::Symbol& sym)
{
    // [TODO] check if sym is a group of symbols instead of just one symbol, if so return multiple parsers
    std::vector<rcc::shared_ptr<MXNetOutputParser>> output;
    auto& factory = mo::MetaObjectFactory::instance();
    auto ctrs = factory.getConstructors(MXNetOutputParser::getHash());
    IObjectConstructor* best_ctr = nullptr;
    int best_val = 0;
    for (auto ctr : ctrs)
    {
        if (auto info = dynamic_cast<MXNetOutputParser::MXNetOutputParserInfo*>(ctr->GetObjectInfo()))
        {
            auto p = info->parserPriority(sym);
            if (p > best_val)
            {
                best_ctr = ctr;
            }
        }
    }
    if (best_ctr)
    {
        auto obj = best_ctr->Construct();
        obj->Init(true);
        output.push_back(obj);
    }
    return output;
}

bool MXNetOutputParser::readAttr(const ::mxnet::cpp::Symbol& sym, const std::string& name, std::string& attr)
{
    int success = 0;
    const char* output = nullptr;
    if (MXSymbolGetAttr(sym.GetHandle(), name.c_str(), &output, &success) == 0)
    {
        if (success)
        {
            attr = output;
            return true;
        }
    }
    return false;
}
}
}
