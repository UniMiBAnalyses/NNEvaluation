#ifndef DNNEvaluator_
#define DNNEvaluator_


#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
namespace NNEvaluation{

class DNNEvaluator {
public:
    explicit DNNEvaluator(const std::string modelPath, bool verbose=false);
    ~DNNEvaluator();

    std::vector<float> analyze(std::vector<float> data);
    std::vector<std::vector<float>> analyze_batch(std::vector<std::vector<float>> data);
private:
    
    float scale_variable(int var_index, float & var);
    void open_session();

    std::string modelPath_;
    unsigned int n_inputs_;
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    std::string graphPath_;
    tensorflow::GraphDef* graphDef_;
    tensorflow::Session* session_;
    bool verbose_;
    // Extracted from file: ?? and the rest? is it not extracetd from file? FIXME
    std::vector<std::pair<float,float>> scaler_factors_;
    bool session_ready_ = false;
};

}; // namespace NNEvaluation

#endif
