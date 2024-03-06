export CUDA_VISIBLE_DEVICES=0

python main.py \
        model_name "-" \
        model_path "-" \
        noise_type "sythetic" \
        task_type "zh-en" \
        noise_data_type_name "word" \
        src_file_path "--" \
        tgt_file_path "--" \
        clean_src_dev_path "-" \
        character_src_dev_path "-" \
        word_src_dev_path "-" \
        multi_src_dev_path "-" \
        res_path "-" \
        redundancy_res_path "-" \
        prompt_path "-" \
        top_5_for_itself "-" \
        top_5_for_clean "-" \
        top_5_for_all "-" \




