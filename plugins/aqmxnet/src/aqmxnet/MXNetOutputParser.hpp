#pragma once
#include <Aquila/core/Algorithm.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <boost/lexical_cast.hpp>

namespace mxnet
{
    namespace cpp
    {
        class Symbol;
        class NDArray;
    } // namespace cpp
} // namespace mxnet

namespace aq
{
    namespace mxnet
    {

        class MXNetOutputParser : public TInterface<MXNetOutputParser, Algorithm>
        {
          public:
            struct MXNetOutputParserInfo : public mo::IMetaObjectInfo
            {
                virtual int parserPriority(const ::mxnet::cpp::Symbol& sym) = 0;
            };

            typedef MXNetOutputParserInfo InterfaceInfo;

            static std::vector<rcc::shared_ptr<MXNetOutputParser>> createParsers(const ::mxnet::cpp::Symbol& sym);

            // call after creation to setup parser parameters
            virtual void setupParser(const ::mxnet::cpp::Symbol& sym,
                                     const CategorySet::ConstPtr& cats,
                                     const std::vector<::mxnet::cpp::NDArray>& outputs) = 0;
            // called before a set of mini batches
            virtual void preBatch(unsigned int total_batch_size) = 0;
            // Called after a mini batch
            virtual void postMiniBatch(const std::vector<cv::Rect>& batch_bb, const DetectedObjectSet& dets) = 0;
            // Called after a batch with the input parameter used into the neural net
            // parameter is used for pulling metadata such as timestamp, coordinate frame, etc, and passing into outputs
            virtual void postBatch(mo::IParam& param) = 0;

          protected:
            virtual bool processImpl() override { return true; }
            static bool readAttr(const ::mxnet::cpp::Symbol& sym, const std::string& name, std::string& attr);

            template <class T>
            static bool readAttr(const ::mxnet::cpp::Symbol& sym, const std::string& name, T& attr)
            {
                std::string attr_string;
                if (readAttr(sym, name, attr_string))
                {
                    return boost::conversion::try_lexical_convert(attr_string, attr);
                }
                return false;
            }
        };

    } // namespace mxnet
} // namespace aq
