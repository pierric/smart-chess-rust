from argparse import ArgumentParser

from jina import Executor, requests, dynamic_batching, Deployment, Flow
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
import torch
import numpy as np

from module import load_model
from torch._prims_common import check


class ChessInput(BaseDoc):
    boards: NdArray[8, 8, 112]
    meta: NdArray[7]


class ChessOutput(BaseDoc):
    action: NdArray[4672]
    score: float


class ChessExecutor(Executor):
    def __init__(self, n_res_blocks, checkpoint, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        if checkpoint.endswith(".ckpt"):
            self.model = load_model(
                n_res_blocks=n_res_blocks,
                device=device,
                checkpoint=checkpoint,
                inference=True,
                compile=True,
            )
        elif checkpoint.endswith(".pt2"):
            self.model = torch._inductor.aoti_load_package(checkpoint)
        elif checkpoint.endswith(".pt"):
            self.model = torch.jit.load(checkpoint)
        else:
            raise RuntimeError(f"Unsupported checkpoint {checkpoint}")
        print("model loaded.")

    @requests(on="/infer")
    @dynamic_batching(preferred_batch_size=8, timeout=2)
    def infer(self, docs: DocList[ChessInput], **kwargs) -> DocList[ChessOutput]:
        tensors = []

        for doc in docs:
            b = torch.from_numpy(doc.boards.astype(np.float32)).permute((2, 0, 1))
            m = (
                torch.from_numpy(doc.meta.astype(np.float32))
                .repeat_interleave(64)
                .reshape(7, 8, 8)
            )
            tensors.append(torch.cat((b, m)))

        inp = torch.stack(tensors).to(device=self.device)

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device, dtype=torch.bfloat16, cache_enabled=False
            ):
                action, score = self.model(inp)

        action = list(action.to(torch.float).cpu().numpy())
        score = score.to(torch.float).cpu().numpy().flatten().tolist()

        return DocList[ChessOutput](
            ChessOutput(action=a, score=s) for a, s in zip(action, score)
        )


def main():
    parser = ArgumentParser("serve")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-res-blocks", default=9)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-p", "--port", default=8900)
    args = parser.parse_args()

    # with Deployment(
    #    uses=ChessExecutor,
    #    uses_with={
    #        "n_res_blocks": args.n_res_blocks,
    #        "device": args.device,
    #        "checkpoint": args.checkpoint,
    #    },
    #    port=args.port,
    # ) as dep:
    #    dep.block()
    f = (
        Flow()
        .config_gateway(protocol=["grpc"], port=args.port)
        .add(
            name="model",
            uses=ChessExecutor,
            uses_with={
                "n_res_blocks": args.n_res_blocks,
                "device": args.device,
                "checkpoint": args.checkpoint,
            },
        )
    )
    with f:
        f.block()


if __name__ == "__main__":
    main()
