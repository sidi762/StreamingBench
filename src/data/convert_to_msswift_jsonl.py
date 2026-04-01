import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List


def iter_samples(data: Any) -> Iterable[Dict[str, Any]]:
    """Yield sample dicts from both standard and SQA-style nested lists."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                for sub_item in iter_samples(item):
                    yield sub_item


def normalize_options(options: Any) -> List[str]:
    if not isinstance(options, list):
        return []
    return [str(opt).strip() for opt in options]


def normalize_answer(question: Dict[str, Any], options: List[str]) -> str:
    if 'answer' in question and question['answer'] not in (None, ''):
        answer = str(question['answer']).strip()
        if len(answer) == 1 and answer.isalpha() and options:
            prefix = f"{answer.upper()}."
            for opt in options:
                if opt.strip().upper().startswith(prefix):
                    return opt
        return answer

    if 'ground_truth_output' in question and question['ground_truth_output'] not in (None, ''):
        return str(question['ground_truth_output']).strip()

    return ''


def build_user_prompt(sample: Dict[str, Any], question: Dict[str, Any], options: List[str]) -> str:
    lines: List[str] = []

    task_type = question.get('task_type') or sample.get('task_type')
    if task_type:
        lines.append(f"Task: {task_type}")

    time_range = sample.get('time')
    if time_range:
        lines.append(f"Time range: {time_range}")

    ts = question.get('time_stamp')
    if ts:
        lines.append(f"Question timestamp: {ts}")

    video_path = sample.get('video_path')
    if video_path:
        lines.append(f"Video: {video_path}")

    lines.append(f"Question: {question.get('question', '').strip()}")

    if options:
        lines.append("Options:")
        lines.extend(options)

    return "\n".join(lines).strip()


def to_messages(sample: Dict[str, Any], question: Dict[str, Any], options: List[str], system_prompt: str) -> List[Dict[str, str]]:
    user_prompt = build_user_prompt(sample, question, options)
    answer = normalize_answer(question, options)
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
        {'role': 'assistant', 'content': answer},
    ]


def convert_file(input_path: str, output_fp, system_prompt: str) -> int:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    count = 0
    for sample in iter_samples(data):
        questions = sample.get('questions', [])
        if not isinstance(questions, list):
            continue

        for question in questions:
            if not isinstance(question, dict):
                continue

            options = normalize_options(question.get('options'))
            record = {
                'messages': to_messages(sample, question, options, system_prompt)
            }
            output_fp.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    return count


def resolve_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for item in inputs:
        expanded = sorted(glob.glob(item))
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(item):
            paths.append(item)
    unique_paths = []
    seen = set()
    for p in paths:
        abs_path = os.path.abspath(p)
        if abs_path not in seen and os.path.isfile(abs_path):
            seen.add(abs_path)
            unique_paths.append(abs_path)
    return unique_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert StreamingBench QA JSON files to MS-Swift JSONL format.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='Input JSON file paths or glob patterns, e.g. "./questions_*.json"',
    )
    parser.add_argument('--output', required=True, help='Output JSONL path.')
    parser.add_argument(
        '--system-prompt',
        default='You are a helpful multimodal assistant.',
        help='System message content used for every sample.',
    )
    args = parser.parse_args()

    input_paths = resolve_inputs(args.inputs)
    if not input_paths:
        raise FileNotFoundError('No valid input files found from --inputs.')

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = 0
    with open(args.output, 'w', encoding='utf-8') as output_fp:
        for input_path in input_paths:
            total += convert_file(input_path, output_fp, args.system_prompt)

    print(f'Converted {total} QA pairs from {len(input_paths)} files to: {args.output}')


if __name__ == '__main__':
    main()
