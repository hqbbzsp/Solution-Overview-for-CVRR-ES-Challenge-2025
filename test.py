import os
import json
import argparse
from tqdm import tqdm
from swift.llm import PtEngine, VllmEngine,RequestConfig, BaseArguments, InferRequest, get_model_tokenizer, get_template
from swift.tuners import Swift
import torch


def run_inference(engine, video_path, question, request_config):
    # return "_"
    messages = [
        {"role": "user", "content": f"<video> Answer the following questions based solely on the video content:\nIn the video, {question}\n Just give me the answer."}
    ]
    infer_request = InferRequest(messages=messages, videos=[video_path])
    try:
        response = engine.infer([infer_request], request_config)
        return response[0].choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] Error during inference on {video_path}: {e}")
        return ""
    # response = engine.infer([infer_request], request_config)
    # return response[0].choices[0].message.content.strip()

def main(model_path, input_json, output_json, video_root):
    # === 模型加载 ===
    # args = BaseArguments.from_pretrained(model_path)
    # model, tokenizer = get_model_tokenizer(args.model)
    # model = Swift.from_pretrained(model, model_path)
    # template = get_template(args.template, tokenizer, args.system)
    # engine = PtEngine.from_model_template(model, template),torch_dtype=torch.float16
    engine =  PtEngine(model_path,max_batch_size=8, attn_impl="flash_attn",torch_dtype=torch.float16)
    # engine = VllmEngine(model_path)
    # engine = VllmEngine(model_path, max_model_len=8192, limit_mm_per_prompt={'video': 1})
    request_config = RequestConfig(max_tokens=256, temperature=0.1)

    # === 读取 JSON ===
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_json, 'w', encoding='utf-8') as f:
        f.write("[\n")  # 写入 JSON 数组开始符
        for idx, sample in enumerate(tqdm(data, desc="Running inference")):
            video_rel_path = sample["video_path"]
            video_path = os.path.join(video_root, video_rel_path)
            assert os.path.exists(video_path), f"{video_path} 不存在"

            # if "3 Minutes city busy traffic intersection  time-lapse" not in video_path:
            #     continue

            question = sample["Q"]
            answer = run_inference(engine, video_path, question, request_config)
            sample["A"] = answer

            json.dump(sample, f, ensure_ascii=False, indent=2)
            if idx != len(data) - 1:
                f.write(",\n")
            else:
                f.write("\n")  # 最后一个元素不加逗号
        f.write("]\n")  # 写入 JSON 数组结束符
    print(f"✅ 推理完成，结果保存在 {output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video QA JSON 推理脚本")
    parser.add_argument('--model_path', type=str, required=True, help="适配器路径")
    parser.add_argument('--input_json', type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument('--output_json', type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument('--video_root', type=str, required=True, help="视频文件的根目录")

    args = parser.parse_args()
    main(args.model_path, args.input_json, args.output_json, args.video_root)
