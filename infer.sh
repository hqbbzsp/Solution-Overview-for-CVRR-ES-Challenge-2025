FPS_MAX_FRAMES=90 VIDEO_MAX_PIXELS=$((400*28*28)) VIDEO_TOTAL_PIXELS=$((90 *400*28*28)) FPS=1.5 \
CUDA_VISIBLE_DEVICES=2,3,5 python test.py \
  --model_path ./ModelWeight/Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --input_json ./CVPR25_VideoVLM/Dataset/test_data/test.json \
  --output_json ./CVPR25_VideoVLM/test_result/Qwen2.5-VL-72B-Instruct-AWQ-RAW/result_90frame.json \
  --video_root ./CVPR25_VideoVLM/Dataset/test_data/test-videos \
> "./CVPR25_VideoVLM/test_result/Qwen2.5-VL-72B-Instruct-AWQ-RAW/log_90frame.txt"
