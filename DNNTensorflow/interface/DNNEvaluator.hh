#ifndef DNNEvaluator_
#define DNNEvaluator_

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
namespace NNEvaluation{

class DNNEvaluator {
public:
    explicit DNNEvaluator(const std::string graphPath, 
                        const std::string configsPath);
    ~DNNEvaluator();

    void initialise();
    float analyze(float* data);
private:
    
    float scale_variable(int var_index, float & var);

    std::string graphPath_;
    std::string configsPath_;
    unsigned int n_inputs_;
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    tensorflow::GraphDef* graphDef_;
    tensorflow::Session* session_;
    // Extracted from file
    std::vector<std::pair<float,float>> scaler_factors_;
};
};

#endif