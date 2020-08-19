#include <ct/types/opencv.hpp>

#include "DetectionFilter.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/params/TypeSelector.hpp>
namespace aq
{
    namespace nodes
    {

        bool DetectionFilter::processImpl()
        {
            const auto* in = input;
            MO_ASSERT(in);
            auto out = *in;
            auto classifications = out.getComponentMutable<aq::detection::Classifications>();
            const auto& shape = classifications.getShape();
            for (auto index = shape[0]; index != 0;)
            {
                --index;
                auto& classification = classifications[index];
                const auto num_classifications = classification.size();
                for (size_t i = num_classifications - 1; i >= 0; --i)
                {
                    if (classification[i].cat)
                    {
                        const aq::Category* cat = classification[i].cat;
                        if (std::find(reject_classes.begin(), reject_classes.end(), cat->name) != reject_classes.end())
                        {
                            classification.erase(i);
                        }
                    }
                }
                if (classification.size() == 0)
                {
                    out.erase(index);
                }
            }
            // output_param.updateData(std::move(out), mo::tag::_param = input_param);
            return true;
        }
    } // namespace nodes
} // namespace aq

using namespace aq::nodes;
MO_REGISTER_CLASS(DetectionFilter)
