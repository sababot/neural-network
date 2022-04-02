#include "../include/load_data.h"

#include "eigen3/Eigen/Dense"

struct Set
{
    int count;
    Eigen::MatrixXd inputs;
    Eigen::MatrixXd outputs;
    Eigen::VectorXd classes;
    Eigen::VectorXd biases;
};

class data_set()
{
public:
    // Global
    int input_count, output_count;
   
    struct set training_set;
    struct set validation_set;
    struct set test_set;

private:
    void init_data_set()
    {
        import_inputs();
        import_outputs();
        import_classes();
        import_biases();
    }

    void import_inputs()
    {

    }

    void import_outputs()
    {

    }

    void import_classes()
    {
        
    }

    void import_biases()
    {

    }
};
