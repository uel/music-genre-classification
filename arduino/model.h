#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"

void model_setup();
unsigned char model_predict();
int append_input_data(const float* data, unsigned int size);
