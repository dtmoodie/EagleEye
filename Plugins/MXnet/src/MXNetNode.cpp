#include "MXNetNode.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace EagleLib;
using namespace EagleLib::Nodes;

class BufferFile {
public:
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
        :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            assert(false);
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        delete[] buffer_;
        buffer_ = NULL;
    }
};

void GetMeanFile(const std::string image_file, mx_float* image_data,
    const int channels, const cv::Size resize_size) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, 1);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    cv::resize(im_ori, im, resize_size);

    // Better to be read from a mean.nb file
    float mean = 117.0;

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    for (int i = 0; i < im.rows; i++) {
        uchar* data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (channels > 1)
            {
                mx_float b = static_cast<mx_float>(*data++) - mean;
                mx_float g = static_cast<mx_float>(*data++) - mean;
                *ptr_image_g++ = g;
                *ptr_image_b++ = b;
            }

            mx_float r = static_cast<mx_float>(*data++) - mean;
            *ptr_image_r++ = r;

        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(const char *filename) {
    std::ifstream fi(filename);

    if (!fi.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while (fi >> synset) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if (data[i] > best_accuracy) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
        synset[best_idx].c_str(), best_idx, best_accuracy);
}
using namespace mxnet;
bool MXNet::ProcessImpl()
{
    if(model_file_param.modified || weight_file_param.modified)
    {
        if(boost::filesystem::exists(model_file) && boost::filesystem::exists(weight_file))
        {
            /*        int num_input_nodes = 0;
            int num_output_nodes = 0;
            mxnet::Symbol sym;
            BufferFile ifs(model_file.string());
            std::stringstream ss;
            ss << ifs.GetBuffer();
            dmlc::JSONReader reader(&ss);
            mxnet::Symbol internal = sym.GetInternals();
            std::vector<std::string> all_out = internal.ListOutputs();
            num_output_nodes = all_out.size();
            std::vector<mxnet::Symbol> out_syms(all_out.size());
            for (mx_uint i = 0; i < out_syms.size(); ++i) {
                for (size_t j = 0; j < all_out.size(); ++j) {
                    out_syms[i] = internal[j];
                    break;
                }
            }
            sym = Symbol::CreateGroup(out_syms);

            // Load parameters
            
            std::unordered_map<std::string, NDArray> arg_params, aux_params;
            {
                std::unordered_set<std::string> arg_names, aux_names;
                std::vector<std::string> arg_names_vec = sym.ListArguments();
                std::vector<std::string> aux_names_vec = sym.ListAuxiliaryStates();
                for (size_t i = 0; i < arg_names_vec.size(); ++i) {
                    arg_names.insert(arg_names_vec[i]);
                }
                for (size_t i = 0; i < aux_names_vec.size(); ++i) {
                    aux_names.insert(aux_names_vec[i]);
                }
                std::vector<NDArray> data;
                std::vector<std::string> names;
                BufferFile params(model_file.string());
                dmlc::MemoryFixedSizeStream fi((void*)params.GetBuffer(), params.GetLength());  // NOLINT(*)
                NDArray::Load(&fi, &data, &names);
                CHECK_EQ(names.size(), data.size())
                    << "Invalid param file format";
                for (size_t i = 0; i < names.size(); ++i) {
                    if (!strncmp(names[i].c_str(), "aux:", 4)) {
                        std::string name(names[i].c_str() + 4);
                        if (aux_names.count(name) != 0) {
                            aux_params[name] = data[i];
                        }
                    }
                    if (!strncmp(names[i].c_str(), "arg:", 4)) {
                        std::string name(names[i].c_str() + 4);
                        if (arg_names.count(name) != 0) {
                            arg_params[name] = data[i];
                        }
                    }
                }
            }

            std::unordered_map<std::string, TShape> known_shape;
            /*for (mx_uint i = 0; i < num_input_nodes; ++i) {
                known_shape[std::string(input_keys[i])] =
                    TShape(input_shape_data + input_shape_indptr[i],
                        input_shape_data + input_shape_indptr[i + 1]);
            }*/
            /*std::vector<TShape> arg_shapes;
            std::vector<std::string> arg_names = sym.ListArguments();
            std::vector<std::string> aux_names = sym.ListAuxiliaryStates();
            out_shapes.resize(sym.ListOutputs().size());
            std::vector<TShape> aux_shapes(aux_names.size());
            for (size_t i = 0; i < arg_names.size(); ++i) {
                std::string key = arg_names[i];
                key2arg[key] = i;
                if (known_shape.count(key) != 0) {
                    arg_shapes.push_back(known_shape[key]);
                }
                else {
                    arg_shapes.push_back(TShape());
                }
            }
            CHECK(sym.InferShape(&arg_shapes, &out_shapes, &aux_shapes))
                << "The shape information of is not enough to get the shapes";

            Context ctx = Context::Create(Context::DeviceType::kGPU, 0);

            arg_arrays.clear();
            aux_arrays.clear();
            for (size_t i = 0; i < arg_shapes.size(); ++i) {
                NDArray nd = NDArray(arg_shapes[i], ctx);
                if (arg_params.count(arg_names[i]) != 0) {
                    CopyFromTo(arg_params[arg_names[i]], &nd);
                }
                arg_arrays.push_back(nd);
            }
            for (size_t i = 0; i < aux_shapes.size(); ++i) {
                NDArray nd = NDArray(aux_shapes[i], ctx);
                if (aux_params.count(aux_names[i]) != 0) {
                    CopyFromTo(aux_params[aux_names[i]], &nd);
                }
                aux_arrays.push_back(nd);
            }

            std::map<std::string, Context> ctx_map;
            std::vector<NDArray> grad_store(arg_arrays.size());
            std::vector<OpReqType> grad_req(arg_arrays.size(), kNullOp);
            exec.reset(Executor::Bind(sym, ctx, ctx_map,
                arg_arrays,
                grad_store, grad_req,
                aux_arrays));
            out_arrays = exec->outputs();
        
*/
            model_file_param.modified = false;
            weight_file_param.modified = false;
        }
    }
    return false;
}

bool MXNetC::ProcessImpl()
{
    if (model_file_param.modified || weight_file_param.modified)
    {
        if (boost::filesystem::exists(model_file) && boost::filesystem::exists(weight_file))
        {
            BufferFile json_data(model_file.string());
            BufferFile param_data(weight_file.string());
            int dev_type = 2;  // 1: cpu, 2: gpu
            int dev_id = 0;  // arbitrary.
            mx_uint num_input_nodes = 1;  // 1 for feedforward
            const char* input_key[1] = { "data" };
            const char** input_keys = input_key;
            int channels = 3;

            const mx_uint input_shape_indptr[2] = { 0, 4 };
            // ( trained_width, trained_height, channel, num)
            const mx_uint input_shape_data[4] = { 1,
                static_cast<mx_uint>(channels),
                static_cast<mx_uint>(width),
                static_cast<mx_uint>(height) };
            handle = 0;  // alias for void *

                                      //-- Create Predictor
            MXPredCreate((const char*)json_data.GetBuffer(),
                (const char*)param_data.GetBuffer(),
                static_cast<size_t>(param_data.GetLength()),
                dev_type,
                dev_id,
                num_input_nodes,
                input_keys,
                input_shape_indptr,
                input_shape_data,
                &handle);

            // Just a big enough memory 1000x1000x3
            int image_size = width * height * channels;
            std::vector<mx_float> image_data = std::vector<mx_float>(image_size);
        }
    }
    if(handle)
    {
        cv::Mat tmp;
        if(input->GetSize() != cv::Size(width, height))
        {
            cv::Mat fullsize = input->GetMat(Stream());
            Stream().waitForCompletion();
            cv::resize(fullsize, tmp, cv::Size(width, height));
        }else
        {
            tmp = input->GetMat(Stream());
            Stream().waitForCompletion();
        }
        if(tmp.depth() != CV_32F)
        {
            tmp.convertTo(tmp, CV_32F);
        }
        MXPredSetInput(handle, "data", (float*)tmp.data, tmp.size().area());
        MXPredForward(handle);
        mx_uint *shape = 0;
        mx_uint shape_len;
        mx_uint output_index = 0;
        //-- Get Output Result
        MXPredGetOutputShape(handle, output_index, &shape, &shape_len);
        size_t size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
        std::vector<std::vector<SyncedMemory>> output;
        


    }
    return true;
}