# Guided Deep Embedded Clustering (GDEC) regularization for multi-feature medical signal classification

## I) Introduction

This repository presents the code the experiments of the submitted paper *Guided Deep Embedded Clustering regularization for multi-feature medical signal classification*.

## II) Configuration

Install the different libraries needed to execute the different codes:

    pip install -r requirements.txt

To be able to run the different codes, you need to start by running the following command:

- For Linux systems:

        export PYTHONPATH="${PYTHONPATH}:path_to_the_gdec_code"

- For Windows systems:
    
        set PYTHONPATH=%PYTHONPATH%;path_to_the_gdec_code
        
## III) Proposed Method

![alt text](https://github.com/gdec-submission/gdec/blob/main/figs/Method/GlobalPipeline.jpg)

We propose an end-to-end multi-feature regularized hybrid CNN-Transformer exploiting both the temporal information through the raw signal and the spectral information through a time-frequency representation. Regularization is done through: (1) Two iterated losses allowing to guide the training of each single feature encoder and (2) deep embedded clustering (DEC) applied to the joint embedding space, allowing to improve generalization and partially handle imbalanced datasets. 

### A) Global Pipeline

Our method is composed of five modules:
- **Encoding module**: This module extract embedded representations for each input representation. The obtained embeddings have the same dimension in order to allow a fusion by sum.
- **Fusion module**: This module takes the output embeddings of the encoding module and fuse them using two different strategies: (1) a concatenation straty and (2) a weighted sum strategy with learnable parameters.
- **Guided training module**: This module guides the training of the embedded representations of each input representation using an iterated loss [(Tjandra et al., 2016)](https://arxiv.org/abs/1910.10324).
- **Classification module**: This module uses the final fused embedding to classify the input samples.
- **Deep clustering module**: This module does unsupervised clustering using the fused embedding representations based on Deep Embedded Clustering [(Xie et al., 2016)](https://arxiv.org/abs/1511.06335).

### B) Used single feature encoders

#### i. Time-frequency encoder

![alt text](https://github.com/gdec-submission/gdec/blob/main/figs/Method/2D_CNN_Encoder.jpg)

#### ii. Raw signal encoder

![alt text](https://github.com/gdec-submission/gdec/blob/main/figs/Method/1DCNN_Transformer_Encoder.jpg)

## IV) Code Sturcture

- data: This folder will store the different datasets used to test our proposed method.
- src: This folder contains the source codes necessary to run the different experiment. More details about this folder can be found in the README.md file of this folder.
- parameters_files: This folder contains json files with the parameters of different experiments. 
- results: This folder will store results of the different experiments.
- figs: This fodler contains the different figures used as illustrations in this Git repository.


## V) Examples

**TODO**

## VI) How to use our method with other models ?

Our method was designed to be generic and applicable to other models.

First, you can try by replacing the input features with other input features (for instance, a binary pattern image instead of the raw signal or a cochleagram instead of a log-magnitude spectrogram). Then, you can change the encoder of each input feature.

Moreover, it is also possible to use more than two input representations (this has not been tested), and the principle remains the same: each intermediate representation of a single feature can be guided using an iterated loss, and the final fused embedding space can be regularized using DEC.

Finally, our method can be applied to single feature models (but it has not been tested), by applying the guided training to an intermediate encoding of the input representation, and by applying DEC to the final encoding (used for classification) of the input representation.

## VI) Contact

If you need help to use our repository or if you find any bugs, do not hesitate to contact us (yamil(dot)vindas(at)creatis.insa-lyon.fr).
