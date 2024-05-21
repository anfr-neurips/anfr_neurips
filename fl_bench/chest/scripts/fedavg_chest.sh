RESULT_ROOT= # Path to the directory where the results will be saved
algorithms_dir= # Full Path to the parent directory job_configs

seed=2024
config=federated
clients=client_cxr14,client_padchest,client_cxp_young,client_cxp_old
gpu=0,0,0,0

scheduler=warmcosine
steps_or_epochs=steps
steps=200
rounds=20
lr=5e-4

batch_size=128
cache_rate=1.0
optim_resume=resume
grad_clipping_type=none
criterion=bce


for method in fedavg; do
    for model_name in resnet_50_supervised; do
        
        # preferred name for the output folder
        name="${model_name}_${method}_${steps}_${steps_or_epochs}_${lr}_${scheduler}_bs_${batch_size}"
        target_config=${algorithms_dir}/chest_${config}

        # preferably call the setup script with the full path, and specify the wanted arguments as below
        # the setup script will write the relevant parameters to the corresponding job config file
        python3 src/setup.py \
        -job ${target_config} \
        -method ${method} \
        -num_rounds ${rounds} \
        -lr ${lr} \
        --batch_size ${batch_size} \
        --cache_rate ${cache_rate} \
        --criterion ${criterion} \
        --grad_clipping_type ${grad_clipping_type} \
        --scheduler ${scheduler} \
        --optim_resume ${optim_resume} \
        --steps_or_epochs ${steps_or_epochs} \
        -local_steps ${steps} \
        -model_name ${model_name} \
        -seed ${seed} \
        --opt_args '{"class": "Adam"}' \
        --amp \

        nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -c ${clients} -gpu ${gpu}
    done
done