#include <memory>
#include "NNEvaluation/DNNTensorflow/interface/DNNEvaluator.hh"

#include <iostream>
#include <fstream>

NNEvaluation::DNNEvaluator::DNNEvaluator(const std::string modelPath, bool verbose)
    : modelPath_(modelPath)
    , graphDef_(nullptr)
    , session_(nullptr)
    , verbose_(verbose)
{
    // show tf debug logs
    tensorflow::setLogging("1");
    
    std::cout << "Model path: "<<modelPath_ << std::endl;
    // Parse scaler configuration
    std::string scalerPath = modelPath_ + "scaler.txt";
    std::ifstream inputfile_scaler{scalerPath};
    std::cout << "Reading file: "<< scalerPath << std::endl;
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
            std::cout << "variable: "<< varname << " " << m_ << " " << s_ << std::endl; 
        }  
    }   
    inputfile_scaler.close();

    // Save number of variables
    n_inputs_ = scaler_factors_.size();
    std::cout << "Working with " << n_inputs_ << " variables" << std::endl;

    // Parse TF metadata
    std::string tfmetadataPath = modelPath_ + "tf_metadata.txt";
    std::ifstream inputfile_tfmetadata{tfmetadataPath};
    std::cout << "Reading file: "<< tfmetadataPath << std::endl;
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
    graphPath_ = modelPath_ + "model.pb";

}

NNEvaluation::DNNEvaluator::~DNNEvaluator()
{
    // close the session
    tensorflow::closeSession(session_);
    session_ = nullptr;

    // delete the graph
    delete graphDef_;
    graphDef_ = nullptr;
}

void NNEvaluation::DNNEvaluator::open_session(){
    if (verbose_) std::cout << "Opening Tensorflow session" << std::endl;
    if (session_ready_) return;

    graphDef_ = tensorflow::loadGraphDef(graphPath_); // TODO check what happens if file not present
    // create a new session and add the graphDef
    session_ = tensorflow::createSession(graphDef_);
    session_ready_ = true;
    std::cout << "Tensorflow session ready" <<std::endl;
}

float NNEvaluation::DNNEvaluator::scale_variable(int var_index, float & var){
    auto [mean,scale] = scaler_factors_.at(var_index);
    return (var - mean) / scale;
}

std::vector<float> NNEvaluation::DNNEvaluator::analyze(std::vector<float> data)
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

    return vecresult;
}

