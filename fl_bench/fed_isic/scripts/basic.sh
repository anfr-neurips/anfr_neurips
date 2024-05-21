RESULT_ROOT= #
algorithms_dir="/user/fl_bench/fed_isic/job_configs"

config=federated_per

seed=2
gpu=0,0,0,0,0,0

scheduler=onecycle
optim_resume=resume
steps_or_epochs=steps
steps=200
rounds=80
lr=5e-4

grad_clipping_type=none
grad_clipping_value=1.0
batch_size=64


for seed in 1 2 3; do
    for method in fedavg; do
        for grad_clipping_type in none; do
            for model_name in resnet_50_pretrained ; do

                name="${model_name}_${method}_${grad_clipping_type}_${batch_size}_${scheduler}_${optim_resume}_seed_${seed}"
                target_config=${algorithms_dir}/fed_isic_${config}

                python3 /user/fl_bench/fed_isic/src/setup.py \
                -job ${target_config} \
                -method ${method} \
                -local_steps ${steps} \
                -num_rounds ${rounds} \
                -lr ${lr} \
                --batch_size ${batch_size} \
                --grad_clipping_type ${grad_clipping_type} \
                --grad_clipping_value ${grad_clipping_value} \
                --scheduler ${scheduler} \
                --optim_resume ${optim_resume} \
                --steps_or_epochs ${steps_or_epochs} \
                -model_name ${model_name} \
                -seed ${seed} \
                --opt_args '{"class": "Adam"}' \
                --amp \
                
                nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}
            done
        done
    done
done