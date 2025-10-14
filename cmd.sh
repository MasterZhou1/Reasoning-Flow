bash run_sweep.sh

python export_html.py --sweep_dir results/final_dataset/sweep_out --out_dir results/final_dataset/html_out

# or

python cot-hidden-dynamic.py --hf_model /home/users/your_name/store/pretrain/Qwen/Qwen3-0.6B --data_file data/all_final_data.json --pooling step_mean --accumulation cumulative --similarity_order 0 --save_dir results/demo/Qwen3-0.6B/pool=step_mean_acc=cumulative_k=16_ord=0 --color_scale '["#d0e1f2","#6aaed6","#2171b5","#08306b"]' --hide_axis_text
python cot-hidden-dynamic.py --hf_model /home/users/your_name/store/pretrain/Qwen/Qwen3-0.6B --data_file data/all_final_data.json --pooling step_mean --accumulation cumulative --similarity_order 1 --save_dir results/demo/Qwen3-0.6B/pool=step_mean_acc=cumulative_k=16_ord=1 --color_scale '["#d0e1f2","#6aaed6","#2171b5","#08306b"]' --hide_axis_text
python cot-hidden-dynamic.py --hf_model /home/users/your_name/store/pretrain/Qwen/Qwen3-0.6B --data_file data/all_final_data.json --pooling step_mean --accumulation cumulative --similarity_order 3 --save_dir results/demo/Qwen3-0.6B/pool=step_mean_acc=cumulative_k=16_ord=3 --color_scale '["#d0e1f2","#6aaed6","#2171b5","#08306b"]' --hide_axis_text


# or

python compute_similarity_averages.py --hf_models /home/users/your_name/store/pretrain/Qwen/Qwen3-0.6B --data_file data/all_final_data.json --orders 0,1,2,3 --pooling step_mean --accumulation cumulative --load_in_8bit --device cuda:0 --save_dir results/averages
