export GLOO_SOCKET_IFNAME=eth0
export WANDB_MODE=disabled

finetuned_models=("tulu2-70b-ipl-ipo-3W-full")
inference_models=("tulu2-dpo-7b")
test_data_types=("generation" "multichoice")

# gen test results with new prompts
for finetuned_model in "${finetuned_models[@]}"
    do
    finetuned_model_path=YOUROUTPUTPATH/${finetuned_model}/

    for test_data_type in "${test_data_types[@]}"
        do
        case ${test_data_type} in 
            "multichoice")
                ENTRY=codes/get_model_infer_batch_logits.py
                E_TARLEN=1
                N_LEN=4096
                ;;
            "generation")
                ENTRY=codes/get_model_infer_batch.py
                E_TARLEN=4096
                N_LEN=4096
                ;;
        esac

        # gen new prompts
        new_prompt_file=YOUROUTPUTPATH/${test_data_type}_${finetuned_model}.jsonl
        if [ -f ${new_prompt_file} ]; then
            echo "new prompt file exits"
        else
            python3 codes/get_model_infer_batch.py \
                --model-path ${finetuned_model_path} \
                --question-file data/test-meta-${test_data_type}-prompt.jsonl \
                --answer-file ${new_prompt_file} \
                --max-target-len ${N_LEN} \
                --num-gpus 8 \
                --num-partitions 1 \
                --temperature 0.8 \
                --top-p 0.95
        fi
        
        # gen test results
        for inference_model in "${inference_models[@]}"
            do
            python3 ${ENTRY} \
                --model-path YOURMODELPATH/tulu2-dpo-${inference_model}/ \
                --question-file YOUROUTPUTPATH/${test_data_type}_${finetuned_model}.jsonl \
                --answer-file YOUROUTPUTPATH/${test_data_type}_${finetuned_model}_${inference_model}.jsonl \
                --max-target-len ${E_TARLEN} \
                --num-gpus 8 \
                --num-partitions 1 \
                --temperature 0.8 \
                --top-p 0.95
            done
        done
    done
