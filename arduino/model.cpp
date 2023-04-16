#include "model_data.h"
#include "model.h"
#include "arduino.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* output = nullptr;
TfLiteTensor* input = nullptr;
static unsigned int input_bytes_loaded = 0;

constexpr int kTensorArenaSize = 60000;
uint8_t tensor_arena[kTensorArenaSize];

int append_input_data(const float* data, unsigned int size)
{
    int offset = input_bytes_loaded/13;
    for (int i = 0; i < size; i++) {
      int8_t x_quantized = data[i] / input->params.scale + input->params.zero_point;
      input->data.uint8[i*128 + offset] = x_quantized;
    }
    input_bytes_loaded += size;
    return input_bytes_loaded;
}

void model_setup() {
  tflite::InitializeTarget();

  model = tflite::GetModel(g_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version %d not equal to supported version");
    return;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

unsigned char model_predict()
{
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.print("Invoke failed on index: ");
    Serial.println(invoke_status);
    return 255;
  }

  TfLiteTensor* output = interpreter->output(0);

  unsigned char max_index = 255;
  float max_val = -1;
  for (int i = 0; i < output->bytes; i++) {
    float y = (output->data.uint8[i] - output->params.zero_point) * output->params.scale;
    if (y > max_val) {
      max_index = i;
      max_val = y;
    }
  }

  input_bytes_loaded = 0;
  return max_index;
}
