#include <memory>
#include "NNEvaluation/DNNTensorflow/interface/DNNEvaluator.hh"

#include <iostream>
#include <fstream>

NNEvaluation::DNNEvaluator::DNNEvaluator(const std::string graphPath, 
                                        const std::string configsPath)
    : graphPath_(graphPath)
    , configsPath_(configsPath)
    , graphDef_(nullptr)
    , session_(nullptr)
{
    // show tf debug logs
    tensorflow::setLogging("0");

    std::ifstream inputfile_scaler{configsPath_};
    std::cout << "Reading file: "<< configsPath_ << std::endl;
    if(inputfile_scaler.fail())  
    { 
        std::cout << "error" << std::endl;
        exit(1);
    }else{ 
        // The first line contains the inputand output tensornames
        inputfile_scaler >> input_tensor_name_ >> output_tensor_name_;
        // Now read mean, scale factors for each variable
        float m_,s_;
        while (! inputfile_scaler.eof()){
        inputfile_scaler >> m_ >> s_; 
        scaler_factors_.push_back({m_, s_});
        }  
    }   
    inputfile_scaler.close();

    // Save number of variables
    n_inputs_ = scaler_factors_.size();
    
    // Initialise the sessions
    initialise();
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

void NNEvaluation::DNNEvaluator::initialise()
{
    // load the graph
    std::cout << "loading graph from " << graphPath_ << std::endl;
    graphDef_ = tensorflow::loadGraphDef(graphPath_);

    // create a new session and add the graphDef
    session_ = tensorflow::createSession(graphDef_);
}

float NNEvaluation::DNNEvaluator::scale_variable(int var_index, float & var){
    auto [mean,scale] = scaler_factors_.at(var_index);
    return (var - mean) / scale;
}

float NNEvaluation::DNNEvaluator::analyze(float* data)
{
    tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, n_inputs_ });
    float* d = input.flat<float>().data();
    for (uint i = 0; i < n_inputs_; i++, d++)
    {
        *d = scale_variable(i, data[i]);
    }

    // define the output and run
    std::cout << "session.run" << std::endl;
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session_, { { input_tensor_name_, input } }, { output_tensor_name_ }, &outputs);

    float result = outputs[0].matrix<float>()(0, 0);
    // check and print the output
    return result;
}

