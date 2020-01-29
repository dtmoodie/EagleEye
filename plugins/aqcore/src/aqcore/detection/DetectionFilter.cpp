#include "DetectionFilter.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/params/TypeSelector.hpp>
namespace aq
{
namespace nodes
{

template <class DetType>
void DetectionFilter::apply()
{
    const DetType* in = mo::get<const DetType*>(input);
    MO_ASSERT(in);
    DetType out = *in;
    for (auto itr = out.end(); itr != out.begin();)
    {
        --itr;
        typename DetType::value_type& det = *itr;
        for (int i = det.classifications.size() - 1; i >= 0; --i)
        {
            if (det.classifications[i].cat)
            {
                const aq::Category* cat = det.classifications[i].cat;
                if (std::find(reject_classes.begin(), reject_classes.end(), cat->name) != reject_classes.end())
                {
                    det.classifications.erase(i);
                }
            }
        }
        if (det.classifications.size() == 0)
        {
            itr = out.erase(itr);
        }
    }
    //output_param.updateData(std::move(out), mo::tag::_param = input_param);
}

bool DetectionFilter::processImpl()
{
    mo::selectType<decltype(input_param)::TypeTuple>(*this, input_param.getTypeInfo());
    return true;
}
}
}

using namespace aq::nodes;
MO_REGISTER_CLASS(DetectionFilter)
