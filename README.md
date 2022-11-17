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
- src: This folder contains the source codes necessary to run the different experiment. 
- parameters_files: This folder contains json files with the parameters of different experiments. 
- results: This folder will store results of the different experiments.
- figs: This fodler contains the different figures used as illustrations in this Git repository.


## V) Examples

**Careful**: The following instructions make the assumption that you execute the code from the repository where it is located.

### A) Experiment 1

#### i. Training/obtaining models.

To obtain the results of single feature models, two options are possible:
- To get the pre-trained and pre-computed results:

        python training_model_one_feature.py
    
- To train one model from scratch (to obtain the different single features this code has to be executed several times with different parameters files):

        python training_model_one_feature.py --parameters_file path_to_parameter_file
    
The parameters files for the single feature experiments are located in gdec/parameters_files/Experiment_1/DatasetName.

The same remarks are valid to train/obtain late and intermediate fusion multi-feature models.

#### ii. Visualizing the obtained metrics.

To visualize the results, you have to go to the gdec/src/utils/ folder and execute the following command line:

        python plot_results_experiment_1.py --results_folder path_to_experiment_results_folder/metrics/
        
 When visualizing all the metrics, you should obtain results similar to the ones in our work:
 
 | Datase | Model                | Modality   | Fusion Method | MCC             | F1-Score       | Accuracy        |
|--------|----------------------|------------|---------------|-----------------|----------------|-----------------|
| PTB    | 1D CNN-Transformer   | Raw signal | -             | 98,31 ± 0,43  | 99,16 ± 0,22 | "99,32 ± 0,17  |
| PTB    | 2D CNN               | TFR        | -             | 97,03 ± 1,22  | 98,51 ± 0,61 | "98,80 ± 0,50  |
| PTB    |                      | GAF        | Weight. Sum   |                 |                |                 |
| PTB    | Ahmad et al. (2021)  | MTF        | Weight. Sum   | -               | 98             | 99,2          |
| PTB    |                      | RP         | Weight. Sum   |                 |                |                 |
| PTB    | Vindas et al. (2022) | Both       | Weight. Sum   | 99,29 ± 0,21  | 99,65 ± 0,10 | 99,71 ± 0,08 |
| PTB    | Late Fusion (ours)   |            | Weight. Sum   | 98,45 ± 0,49  | 99,22 ± 0,25 | 99,38 ± 0,20  |
| PTB    | Ours (No Reg.)       |            | Cat.          | 97,11 ± 0,43  | 98,60 ± 0,22 | 98,84 ± 0,18  |
| PTB    | Ours (No Reg.)       |            | Weight. Sum   | 97,29 ± 0,50  | 98,64 ± 0,25 | 98,91 ± 0,20  |
| PTB    | Ours (Reg.)          |            | Cat.          | 99,28 ± 0,11 | 99,64 ± 0,05 | 99,71 ±0,04   |
| PTB    | Ours (Reg.)          |            | Weight. Sum   | 99,18 ± 0,25  | 99,59 ± 0,13 | 99,67 ± 0,10  |
| ESR    | 1D CNN-Transformer   | Raw signal | -             | 95,14 ± 1,67  | 97,55 ± 0,87 | 98,40 ± 0,59  |
| ESR    | 2D CNN               | TFR        | -             | 92,81 ± 3,53  | 96,33 ± 1,88 | 97,59 ± 1,35  |
| ESR    | Hilal et al. (2022)  | Raw signal | -             | 99,09         | 98,89        | 98,67         |
| ESR    | Xu et al. (2020)     |            | -             | -               | 98,59        | 99,39         |
| ESR    | Late Fusion (ours)   | Both       | Weight. Sum   | 97,45 ± 1,49  | 98,71 ± 0,77 | 99,16 ± 0,51  |
| ESR    | Ours (No Reg.)       |            | Cat.          | 93,40 ± 1,32  | 96,67 ± 0,68 | 97,89 ± 0,45  |
| ESR    | Ours (No Reg.)       |            | Weight. Sum   | 93,01 ± 2,22  | 96,45 ± 1,22 | 97,77 ± 0,69  |
| ESR    | Ours (Reg.)          |            | Cat.          | 96,51 ± 0,46  | 98,25 ± 0,23 | 98,88 ± 0,15  |
| ESR    | Ours (Reg.)          |            | Weight. Sum   | 98,65 ± 0,70  | 98,42 ± 0,35 | 98,98 ± 0,23  |

 
 **TODO**
 
 #### iii. Visualizing projections
        
**WARNING**: The following instructions are valid only for models trained from scratch and not for the downloaded models.

To plot the projections of models trained from scratch, you can use the codes *gdec/src/Experiment_1/plot_embeddings_one_feature.py* and *gdec/src/Experiment_1/plot_embeddings_multi_feature.py* (see their help for more information).

Using these codes, you can get figures similar to the next ones:

![alt text](https://github.com/gdec-submission/gdec/blob/main/figs/Results/Experiment_1/Projections_PTB.jpg)

        
### B) Experiments 2 and 3

You can execute experiments 2 and 3 by using the codes *influence_guided_training.py* and *influence_dec.py*, respectively. Each code can be executed with or without the option --parameters_file.

To plot the results of the experiments, you can use the code *gdec/src/utils/* with the following options:
- *results_file*: Mandatory, it corresponds to the file containing the metrics of the experiment (usually located in the metrics folder of the results experiment folder).
- *selected_epoch*: Optional, it corresponds to the particular epoch to compute the different metrics.
- *last_epochs_use*: Optional, it corresponds to the last epochs to use to compute the mean metrics.
- *plot_curves*: Optional, it corresponds to a boolean to plot the loss and metrics curves.
- *plot_val_metrics*: Optional. True if wanted to compute the val metrics and plot it. If there are no validation metrics, this argument should be False.

## VI) How to use our method with other models ?

Our method was designed to be generic and applicable to other models.

First, you can try by replacing the input features with other input features (for instance, a binary pattern image instead of the raw signal or a cochleagram instead of a log-magnitude spectrogram). Then, you can change the encoder of each input feature.

Moreover, it is also possible to use more than two input representations (this has not been tested), and the principle remains the same: each intermediate representation of a single feature can be guided using an iterated loss, and the final fused embedding space can be regularized using DEC.

Finally, our method can be applied to single feature models (but it has not been tested), by applying the guided training to an intermediate encoding of the input representation, and by applying DEC to the final encoding (used for classification) of the input representation.

## VI) Contact

If you need help to use our repository or if you find any bugs, do not hesitate to contact us (yamil(dot)vindas(at)creatis.insa-lyon.fr).
