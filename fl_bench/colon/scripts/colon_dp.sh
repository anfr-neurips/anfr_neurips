RESULT_ROOT= #
algorithms_dir= #

config=federated_dp
gpu=0,0,0
num_clients=3
scheduler=none
fedprox_mu=0.0
optim_resume=resume
steps_or_epochs=steps
steps=500
rounds=25
lr=5e-5

grad_clipping_type=none
grad_clipping_value=0.00
batch_size=64
opt_args='{"class": "Adam"}'
for seed in 1; do
    for method in fedavg; do
        for model_name in anfr_resnet50_supervised; do

            name="${model_name}_${method}_scheduler_${scheduler}_lr_${lr}_batch_${batch_size}_grad_${grad_clipping_type}_${grad_clipping_value}_seed_${seed}_dp_w_eps"
            target_config=${algorithms_dir}/colon_${config}

            python3 ../src/setup.py \
            -job ${target_config} \
            -method ${method} \
            -alpha 0.5 \
            -local_steps ${steps} \
            -num_rounds ${rounds} \
            -lr ${lr} \
            --batch_size ${batch_size} \
            --grad_clipping_type ${grad_clipping_type} \
            --grad_clipping_value ${grad_clipping_value} \
            --scheduler ${scheduler} \
            --optim_resume ${optim_resume} \
            --fedprox_mu ${fedprox_mu} \
            --steps_or_epochs ${steps_or_epochs} \
            -model_name ${model_name} \
            -seed ${seed} \
            --opt_args "${opt_args}" \
            # --amp \
            
            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n ${num_clients} -gpu ${gpu}
        done
    done
done