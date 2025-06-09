FPS_MAX_FRAMES=120 VIDEO_MAX_PIXELS=$((400*28*28)) VIDEO_TOTAL_PIXELS=$((120 *400*28*28)) FPS=2 \
CUDA_VISIBLE_DEVICES=2,3,5 python test.py \
  --model_path ./ModelWeight/Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --input_json ./CVPR25_VideoVLM/Dataset/test_data/test.json \
  --output_json ./CVPR25_VideoVLM/test_result/Qwen2.5-VL-72B-Instruct-AWQ-RAW/result.json \
  --video_root ./CVPR25_VideoVLM/Dataset/test_data/test-videos \
> "./CVPR25_VideoVLM/test_result/Qwen2.5-VL-72B-Instruct-AWQ-RAW/log.txt"
