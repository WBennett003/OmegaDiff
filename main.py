# -*- coding: utf-8 -*-
from main_ctrl import Enzyme
import json

def main():
    with open("run_config.json") as f: #reads the config file to run the code
        config = json.load(f)
    
    print("Building ML Enivroment...")
    # Create Experiment enviroment - Build Model and Memory map of the Dataset
    experiment = Enzyme(
        token_size=config['dataset-config']['amino-acid-token-size'],
        chem_size=config['dataset-config']['chemical-reaction-hash-size'],
        sequence_length=config['dataset-config']['polypeptide-length'],
        layers=config['model-config']['omegaplm-layers'],
        ds_file=config['dataset-config']['dataset-dir'],
        model_weight_dir=config['model-config']['load-weights-dir'],
        train_val_test_split=config['dataset-config']['train-val-test-split'],
        crep_dir=config['model-config']['load-crep-weights-dir'],
        wandb_config=config['wandb-config']
    )
    print("Done!!!")

    print("Starting pretraining!, view progress on wandb logs!")
    # Run pretraining
    experiment.pretrain(
        EPOCHS=config['pretraining-config']['epochs'],
        BATCH_SIZE=config['pretraining-config']['batch-size'],
        EPOCH_SIZE=config['pretraining-config']['epoch-size'],
        lr=config['pretraining-config']['learning-rate'],
        verbose=config['pretraining-config']['verbose-step-size'],
        wab=config['pretraining-config']['wandb-logging'],
        scaleing=config['pretraining-config']['scale-gradients'],
        save_dir=config['model-config']['save-crep-weights-dir'],
        schedule_type=config['pretraining-config']['schedule-type']
    )
    print("Finished pretraining!\n")

    #Run the training enviroments, where the inputed enzyme sequence is modified by masking, removing or mutating either random or targeted residues
    print(f"Starting Training! There are {len(config['training-config'].keys())} training loops")
    for i, training_session in enumerate(config['training-config'].keys()):
        print(f"Starting Training Session {i}:{len(config['training-config'].keys())} | {training_session}!!!")
        experiment.train(
            session_name=training_session,
            EPOCHS=config['training-config'][training_session]['epochs'],
            EPOCH_SIZE=config['training-config'][training_session]['epoch-size'],
            BATCH_SIZE=config['training-config'][training_session]['batch-size'],
            lr=config['training-config'][training_session]['learning-rate'],
            s=config['training-config'][training_session]['classifier-guidance'],
            wab=config['training-config'][training_session]['wandb-logging'],
            target_mask=config['training-config'][training_session]['target-mask-size'],
            mask_rate=config['training-config'][training_session]['mask-target-pct'],
            mutate_rate=config['training-config'][training_session]['mutate-target-pct'],
            verbose_step=config['training-config'][training_session]['verbose-step-size'],
            scaleing=config['training-config'][training_session]['scale-gradients'],
            sampler=config['training-config'][training_session]['sampler'],
        )
        print(f"Finished Training Session {i}:{len(config['training-config'].keys())} | {training_session}!!!\n")


if __name__ == '__main__':
    main()
