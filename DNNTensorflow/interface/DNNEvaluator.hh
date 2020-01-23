#ifndef DNNEvaluator_
#define DNNEvaluator_

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
namespace NNEvaluation{

class DNNEvaluator {
public:
    explicit DNNEvaluator(const std::string modelPath);
    ~DNNEvaluator();

    void initialise();
    float analyze(std::vector<float> data);
private:
    
    float scale_variable(int var_index, float & var);

    std::string modelPath_;
    unsigned int n_inputs_;
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    tensorflow::GraphDef* graphDef_;
    tensorflow::Session* session_;
    // Extracted from file: ?? and the rest? is it not extracetd from file? FIXME
    std::vector<std::pair<float,float>> scaler_factors_;
};

}; // namespace NNEvaluation

#endif