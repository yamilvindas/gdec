#!/usr/bin/env python3
"""
    This code downloads some useful data to run some experiments.
"""
import os
import requests
import argparse

def download_file(file_url, dir_store, file_name, verbose=True):
    """
        Download a file from the given url and stores it in the given directory
        with the given name

        Arguments:
        ----------
        file_url: str
            Url to the file that we want to download
        dir_store: str
            Path of the directory where the downloaded file should be stored.
        file_name: str
            Name of the file when download it
        verbose: bool
            True if you want to print some information about the requested file
    """
    # Searching if the directory name exists
    if os.path.exists(dir_store+'/'+file_name) and verbose:
        print("The file {} already exists".format(dir_store+'/'+file_name))
    else:
        # Downloading the file
        file = requests.get(file_url)
        open(dir_store+'/'+file_name, 'wb').write(file.content)
        if (verbose):
            print("File downloaded successfully and stored in {}".format(dir_store+'/'+file_name))

def download_dataset(dataset_name='ecg_categorization', local_data_directory='../../data/'):
    """
        Downloads one of the used datasets in the experiments (PTB or ESR).

        Arguments:
        ----------
        dataset_name:str
            Name of the dataset to download. Two choices:
                - ECG_Categorization: corresponding two ECG datasets: PTB and MIT-BIH.
                - ESR: corresponding to one EEG dataset for epileptic seizure
                recognition.
        local_data_directory: str
            Path to the local directory where the datasets should be stored
    """
    if (dataset_name.lower() == 'ecg_categorization'):
        if (not os.path.exists(local_data_directory+'/ECG_Heartbeat_Categorization_Dataset/')):
            print("\n=======> Starting download of the ECG Heartbeat Categorization dataset")
            # Creating the directories
            os.mkdir(local_data_directory+'/ECG_Heartbeat_Categorization_Dataset/')

            # Files to download
            files_to_download = [
                                    'mitbih_train.csv',
                                    'mitbih_test.csv',
                                    'ptbdb_abnormal.csv',
                                    'ptbdb_normal.csv'
                                ]

            # Downloading the different files
            for file_to_download in files_to_download:
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/Guided_DEC/data/ECG_Heartbeat_Categorization_Dataset/'+file_to_download,\
                                dir_store=local_data_directory+'/ECG_Heartbeat_Categorization_Dataset/',\
                                file_name=file_to_download,\
                                verbose=False
                            )
            print("\n=======> End download of the ECG Heartbeat Categorization dataset")
        else:
            print("=======> ECG Heartbeat Categorization dataset has alredy been downloaded!\n")
    elif (dataset_name.lower() == 'esr'):
        if (not os.path.exists(local_data_directory+'/EEG_Epileptic_Seizure_Recognition/')):
            print("\n=======> Starting download of the Epileptic Seizure Recognition dataset")
            # Creating the directories
            os.mkdir(local_data_directory+'/EEG_Epileptic_Seizure_Recognition/')

            # Files to download
            files_to_download = [
                                    'data.hdf5',
                                    'Epileptic Seizure Recognition.csv'
                                ]

            # Downloading the different files
            for file_to_download in files_to_download:
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/Guided_DEC/data/EEG_Epileptic_Seizure_Recognition/'+file_to_download,\
                                dir_store=local_data_directory+'/EEG_Epileptic_Seizure_Recognition/',\
                                file_name=file_to_download,\
                                verbose=False
                            )
            print("\n=======> End download of the Epileptic Seizure Recognition dataset")
        else:
            print("=======> Epileptic Seizure Recognition dataset has alredy been downloaded!\n")
    else:
        raise ValueError("Dataset {} is not valid for download".format(dataset_name))

def download_results_experiment(experiment_name='Exp-1_SingleFeature', local_data_directory='../../results/'):
    """
        Downloads one of the used datasets in the experiments (PTB or ESR).

        Arguments:
        ----------
        experiment_name:str
            Name of the experiment from which we want to get the pre-computed results.
            Different choices:
                - Exp-1_SingleFeature: Trained single feature models of experiment 1
                - Exp-1_MultiFeature: Trained intermediate fusion multi-feature
                  models of experiment 1
                - Exp-1_LateFusion: Trained late fusion multi-feature models
        local_data_directory: str
            Path to the local directory where the results should be stored
    """
    # Getting the results directory
    if ('exp-1' in experiment_name.lower()):
        local_data_directory += '/Experiment_1/'
    else:
        raise ValueError('Experiment name {} is not valid'.format(experiment_name))

    # Creating the directories
    exp_folders = []
    if ('exp-1_singlefeature' == experiment_name.lower()):
        exp_folders.append(local_data_directory + '/Exp1_ClassicSpec_ESR/')
        exp_folders.append(local_data_directory + '/Exp1_ClassicSpec_PTB/')
        exp_folders.append(local_data_directory + '/Exp1_RawSignal_ESR/')
        exp_folders.append(local_data_directory + '/Exp1_RawSignal_PTB/')
    elif ('exp-1_multifeature' == experiment_name.lower()):
        exp_folders.append(local_data_directory + '/Exp1_FullModel_WeightedSumWithSoftmax_ESR/')
        exp_folders.append(local_data_directory + '/Exp1_FullModel_WeightedSumWithSoftmax_PTB/')
        exp_folders.append(local_data_directory + '/Exp1_FullModel_Concatenation_ESR/')
        exp_folders.append(local_data_directory + '/Exp1_FullModel_Concatenation_PTB/')
    elif ('exp-1_latefusion' == experiment_name.lower()):
        exp_folders.append(local_data_directory + '/Exp1_LateFusion_ESR/')
        exp_folders.append(local_data_directory + '/Exp1_LateFusion_PTB/')
    else:
        raise ValueError("Experiment name {} is not valid.".format(experiment_name))

    # Creating the experiments folders
    sub_folders = ["metrics", "model", "params_exp"]
    for exp_folder in exp_folders:
        if (not os.path.exists(exp_folder)):
            print("\n=======> Starting download of the results of the experiment {}".format(experiment_name))
            # Creating the main exp folder
            os.mkdir(exp_folder)

            # Experiment name
            exp_name = exp_folder.split('/')[-2]

            for sub_folder in sub_folders:
                # Creating subfolder
                os.mkdir(exp_folder+'/'+sub_folder+'/')

                # Files to download
                if (sub_folder == 'metrics'):
                    files_to_download_exp = ['results_exp-{}_rep-{}.pth'.format(exp_name, i) for i in range(10)]
                elif (sub_folder == 'model'):
                    files_to_download_exp = ['final_model_{}_rep-{}.pth'.format(exp_name, i) for i in range(10)]
                elif (sub_folder == 'params_exp'):
                    files_to_download_exp = ['network_architecture.py', 'params_0.pth']

                # Downloading the different files
                for file_to_download in files_to_download_exp:
                    # print('https://www.creatis.insa-lyon.fr/~vindas/Guided_DEC/results/Experiment_1/{}/{}/'.format(exp_name, sub_folder)+file_to_download)
                    download_file(
                                    file_url='https://www.creatis.insa-lyon.fr/~vindas/Guided_DEC/results/Experiment_1/{}/{}/'.format(exp_name, sub_folder)+file_to_download,\
                                    dir_store=local_data_directory+'/{}/{}/'.format(exp_name, sub_folder),\
                                    file_name=file_to_download,\
                                    verbose=False
                                )
            print("\n=======> End download of the results of the experiment {}. The location is {}".format(experiment_name, exp_folder))
        else:
            print("=======> Experiment results of {} have alredy been downloaded! The location is {}\n".format(experiment_name, exp_folder))


#==============================================================================#
def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--dataset_to_download', default=None, help="Dataset to downlaod. Two options: ECG_Categorization or ESR", type=str)
    ap.add_argument('--exp_results_to_download', default=None, help="Experiment results to downlaod. Different options: Exp-1_SingleFeature, Exp-1_MultiFeature and Exp-1_LateFusion", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    dataset_to_download = args['dataset_to_download']
    exp_results_to_download = args['exp_results_to_download']

    #==========================================================================#
    # Downloading dataset
    if (dataset_to_download is not None):
        download_dataset(dataset_to_download)

    # Download the results of the experiments
    if (exp_results_to_download is not None):
        download_results_experiment(experiment_name=exp_results_to_download, local_data_directory='../../results/')

if __name__=='__main__':
    main()
