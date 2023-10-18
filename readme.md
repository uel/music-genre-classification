# Music genre classification on Arduino

This project classifies music genres on Arduino based boards using tensorflow lite and convolutional neural networks. Training data comes from the [GTZAN Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset.  Uses [fastmfcc](https://github.com/uel/fastmfcc) to preprocess the input audio MFCCs resulting in a realtime perfomance. Average accuracy is 68% across the 10 classes on 8 second audio inputs.

![ui](ui/ui.png)
