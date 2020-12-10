#include <memory>
#include "NNEvaluation/DNNTensorflow/interface/DNNEvaluatorSavedModel.hh"

#include <iostream>
#include <fstream>
#include <vector>

NNEvaluation::DNNEvaluatorSavedModel::DNNEvaluatorSavedModel(const std::string modelPath, bool verbose)
    : modelPath_(modelPath)
    , metaGraph_(nullptr)
    , session_(nullptr)
    , verbose_(verbose)
{
    // show tf debug logs
    tensorflow::setLogging("0");
    
    if (verbose_) std::cout << "Model path: "<<modelPath_ << std::endl;
    // Parse scaler configuration
    std::string scalerPath = modelPath_ + "scaler.txt";
    std::ifstream inputfile_scaler{scalerPath};
    if (verbose_) std::cout << "Reading file: "<< scalerPath << std::endl;
    if(inputfile_scaler.fail())  
    { 
        std::cout << "error" << std::endl;
        exit(1);
    }else{ 
        // Now read mean, scale factors for each variable
        float m_,s_;
        std::string varname{};
        while (inputfile_scaler >> varname >> m_ >> s_){
            scaler_factors_.push_back({m_, s_});
            if (verbose_) std::cout << "variable: "<< varname << " " << m_ << " " << s_ << std::endl; 
        }  
    }   
    inputfile_scaler.close();

    // Save number of variables
    n_inputs_ = scaler_factors_.size();
    if (verbose_) std::cout << "Working with " << n_inputs_ << " variables" << std::endl;

    // Parse TF metadata
    std::string tfmetadataPath = modelPath_ + "tf_metadata.txt";
    std::ifstream inputfile_tfmetadata{tfmetadataPath};
    if (verbose_) std::cout << "Reading file: "<< tfmetadataPath << std::endl;
    if(inputfile_tfmetadata.fail())  
    { 
        std::cout << "error" << std::endl;
        exit(1);
    }else{ 
        // The first line contains the inputand output tensornames
        inputfile_tfmetadata >> input_tensor_name_ >> output_tensor_name_;
    }   
    inputfile_tfmetadata.close();

    // Initialise the sessions: import TF graph
    // initialise(); // refactored!
    // load the graph
    graphPath_ = modelPath_ + "model_output";

}

NNEvaluation::DNNEvaluatorSavedModel::~DNNEvaluatorSavedModel()
{
    // close the session
    tensorflow::closeSession(session_);
    delete session_;
    session_ = nullptr;

    // delete the graph
    delete metaGraph_;
    metaGraph_ = nullptr;
}

void NNEvaluation::DNNEvaluatorSavedModel::open_session(){
    if (session_ready_) return;

    metaGraph_ = tensorflow::loadMetaGraph(graphPath_); // TODO check what happens if file not present
    // create a new session and add the graphDef
    session_ = tensorflow::createSession(metaGraph_, graphPath_);
    session_ready_ = true;
    std::cout << "Tensorflow session ready" <<std::endl;
}

float NNEvaluation::DNNEvaluatorSavedModel::scale_variable(int var_index, float & var){
    auto [mean,scale] = scaler_factors_.at(var_index);
    return (var - mean) / scale;
}

std::vector<float> NNEvaluation::DNNEvaluatorSavedModel::analyze(std::vector<float> data)
{
    // Check if tensorflow session is open, if not open it
    open_session();

    tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, n_inputs_ });
    float* d = input.flat<float>().data();
    for (uint i = 0; i < n_inputs_; i++, d++)
    {
        *d = scale_variable(i, data[i]);
        if(verbose_) std::cout << data[i] << "(" << *d << ") | ";
    }

    // define the output and run
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session_, { { input_tensor_name_, input } }, { output_tensor_name_ }, &outputs);

    std::vector<float> vecresult;
   
    if ( outputs[0].shape().dims() != 2) {
      std::cout << "Tensor has NOT dimension 2. Not yet implemented!" << std::endl;
      exit(1);
    }
    // case of vector-like tensors, e.g. shape: [1,X]
    if ( outputs[0].shape().dim_size(0) == 1 ) { 
      for (int i=0; i<outputs[0].shape().dim_size(1); ++i){
        vecresult.push_back(outputs[0].matrix<float>()(0, i));
      }
    }
    // matrix-like tensors -> not yet implemented
    else {
      std::cout << "Matrix-like tensors have not been implemented yet!" << std::endl;
      exit(1);
    }

    if (verbose_) {
        std::cout << "Outputs: ";
        std::for_each(vecresult.begin(), vecresult.end(), [](float &r){std::cout << r << " ";});
        std::cout << std::endl;
    }

    return vecresult;
}


std::vector<std::vector<float>> NNEvaluation::DNNEvaluatorSavedModel::analyze_batch(std::vector<std::vector<float>> data)
{
    // Check if tensorflow session is open, if not open it
    open_session();
    uint batch_size = data.size();

    tensorflow::Tensor input(tensorflow::DT_FLOAT, { batch_size, n_inputs_ });
    auto input_tensor_mapped = input.tensor<float, 2>();
    //float* d = input.flat<float>().data();
    if (verbose_) std:: cout << "-----------------" << std::endl;
    for (uint b = 0; b< batch_size; b++){
        for (uint i = 0; i < n_inputs_; i++)
        {   
            input_tensor_mapped(b,i) = scale_variable(i, data[b][i]);
            if(verbose_) std::cout <<  "["<< b <<","<<i<<"] " << data[b][i] << "->" <<  input_tensor_mapped(b,i) << ") | ";

        }
        if (verbose_) std::cout << std::endl;
    }    

    // define the output and run
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session_, { { input_tensor_name_, input } }, { output_tensor_name_ }, &outputs);

    if(verbose_) std::cout << "Reading output. N. outputs:" << outputs.size() << " " << outputs[0].shape()  << std::endl;

    std::vector<std::vector<float>> batch_results;
    
    if ( outputs[0].shape().dims() != 2) {
      std::cout << "Tensor has NOT dimension 2. Not yet implemented!" << std::endl;
      exit(1);
    }
     
    else if ( outputs[0].shape().dims() == 2) {
        // case of [batch, Y] tensors
        for (uint b =0; b< batch_size; b++){
            auto r = outputs[0].tensor<float,2>();
            std::vector<float> vecresult;
            for (int i=0; i<outputs[0].shape().dim_size(1); ++i){
                vecresult.push_back(r(b, i));
            }
            batch_results.push_back(vecresult);
        }
    } 
    return batch_results;
}

