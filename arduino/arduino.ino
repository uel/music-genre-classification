
#include <PDM.h>
#include <fastmfcc.h>
#include "model.h"

short frameBuffer[8192]; // 4 frames
volatile unsigned int bufferSize = 0;
volatile unsigned int overflowCalls = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();

  int bytes2read = min(bytesAvailable, 8192*2-bufferSize);
  bytes2read -= bytes2read % 2048;
  if ( bytes2read > 0 )
  {
    PDM.read(&frameBuffer[bufferSize/2], bytes2read);
    bufferSize += bytes2read;
  }
  else if ( bytesAvailable >= 16384 )
  {
    overflowCalls++;
    PDM.read(frameBuffer, 8192*2);
  }
}

void setup() {
    Serial.begin(19200);
    while(!Serial);
    delay(500);
    
    //Serial.println("Starting...");
    PDM.onReceive(onPDMdata);
    PDM.setBufferSize(16384);
    PDM.setGain(50);
    //Serial.println("Starting PDM");
    if (!PDM.begin(1, 16000)) {
      Serial.println("Failed to start PDM!");
      while (1);
    }

    model_setup();
}

const float mel_means[] = {446.02563, 36.123276, -7.691543, 14.370121, -0.68318045, 5.825986, 3.0853918, -0.99041206, 4.941904, 5.33153, 4.3454432, 5.936014, 2.1101217}; 
const float mel_std[] =   {91.35619, 29.026442, 20.318783, 12.534469, 10.299112, 8.434362, 7.7773743, 7.7836747, 8.236834, 7.762929, 7.3308835, 7.83173, 7.062603};
float temp_mfcc[13];
void loop() {
  onPDMdata();
  if ( bufferSize >= 2048 ) {
      int frames2read = bufferSize/2048;
      for ( int i = 0; i < frames2read; i++)
      {
        long int t1 = millis();
        MFCC(&frameBuffer[i*1024], temp_mfcc);
        //MFCC(test_data, temp_mfcc);
        long int t2 = millis();

        for (int j = 0; j < 13; j++) { //Normalize by mean/std of whole dataset
          temp_mfcc[j] = (temp_mfcc[j] - mel_means[j]) / mel_std[j];
          //Serial.print(temp_mfcc[j]); Serial.print(" ");
        } //Serial.println();

        unsigned int input_bytes_loaded = append_input_data(temp_mfcc, 13);
        if (input_bytes_loaded == 128*13) {
            //Serial.println("Predicting genre...");
            unsigned char result = model_predict();
            Serial.println(result);
            //Serial.print("Time taken MFCC: "); Serial.print(t2-t1); Serial.println(" milliseconds");
            //Serial.print("Result: "); Serial.println(result);
            //Serial.print("Overflow calls: "); Serial.println(overflowCalls);
            overflowCalls = 0;
        } 
      }
      bufferSize -= (frames2read*2048);
  }
}
