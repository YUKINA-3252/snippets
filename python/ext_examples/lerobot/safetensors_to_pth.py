import argparse
from safetensors.torch import load_file
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='outputs/train/wrapping/model.safetensors')
    parser.add_argument('--output_path', type=str, default='outputs/train/wrapping/model.pth')
    return parser.parse_args()

def convert_safetensors_to_pth(input_path, output_path):
    state_dict = load_file(input_path)
    torch.save(state_dict, output_path)

if __name__ == '__main__':
    args = parse_args()
    convert_safetensors_to_pth(args.input_path, args.output_path)
