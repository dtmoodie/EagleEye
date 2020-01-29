#ifndef DETECTIONFILTER_HPP
#define DETECTIONFILTER_HPP
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/DetectionDescription.hpp>

#include <MetaObject/params/TMultiInput-inl.hpp>
#include <MetaObject/params/TMultiOutput.hpp>

#define MULTI_OUTPUT(NAME, ...)                                                                                        \
    mo::TMultiOutput<__VA_ARGS__> NAME##_param;                                                                        \
    template <class V, class... Args, mo::VisitationType FILTER>                                                       \
    inline void reflectHelper(V& visitor,                                                                              \
                              mo::VisitationFilter<FILTER> filter,                                                     \
                              mo::MemberFilter<mo::OUTPUT> param,                                                      \
                              mo::_counter_<__COUNTER__> cnt,                                                          \
                              Args&&... args)                                                                          \
    {                                                                                                                  \
        visitor(mo::Name(#NAME), mo::tagParam(NAME##_param), cnt, std::forward<Args>(args)...);                        \
        reflectHelper(visitor, filter, param, --cnt, std::forward<Args>(args)...);                                     \
    }                                                                                                                  \
    template <class V, class... Args, mo::VisitationType FILTER>                                                       \
    static inline void reflectHelperStatic(V& visitor,                                                                 \
                                           mo::VisitationFilter<FILTER> filter,                                        \
                                           mo::MemberFilter<mo::OUTPUT> param,                                         \
                                           mo::_counter_<__COUNTER__> cnt,                                             \
                                           Args&&... args)                                                             \
    {                                                                                                                  \
        visitor(mo::Name(#NAME), mo::tagType<ct::VariadicTypedef<__VA_ARGS__>>(), cnt, std::forward<Args>(args)...);  \
        reflectHelperStatic(visitor, filter, param, --cnt, std::forward<Args>(args)...);                               \
    }

namespace aq
{
namespace nodes
{

class DetectionFilter : public Node
{
  public:
    MO_DERIVE(DetectionFilter, Node)
        MULTI_INPUT(input, aq::DetectedObjectSet, aq::DetectionDescriptionSet, aq::DetectionDescriptionPatchSet)
        PARAM(std::vector<std::string>, allow_classes, {})
        PARAM(std::vector<std::string>, reject_classes, {})
        MULTI_OUTPUT(output, aq::DetectedObjectSet, aq::DetectionDescriptionSet)
    MO_END

    template <class DetType>
    void apply();

  protected:
    bool processImpl() override;
};
}
}

#endif // DETECTIONFILTER_HPP
