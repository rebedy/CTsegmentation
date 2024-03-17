import torch

from args import parse_args
from models import get_model


if __name__ == "__main__":

    args = parse_args()

    net = get_model(args.net, args)
    net.load_state_dict(torch.load(args.pthfile))

    with torch.no_grad():
        dummy_input = torch.zeros((1, 5, 512, 512), dtype=torch.float32)
        # Save CPU Model
        net.to(device="cpu")
        dummy_input = dummy_input.to("cpu")
        module_cpu = torch.jit.trace(net.forward, dummy_input)
        module_cpu.save("sample_model_cpu.pt")

        # Save CUDA Model
        net.to(device="cuda")
        dummy_input = dummy_input.to("cuda")
        module_cuda = torch.jit.trace(net.forward, dummy_input)
        module_cuda.save("sample_model_cuda.pt")
