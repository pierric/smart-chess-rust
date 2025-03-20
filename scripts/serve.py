from argparse import ArgumentParser

from jina import Executor, requests, dynamic_batching, Deployment
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
import torch
import numpy as np

from module import load_model


class ChessInput(BaseDoc):
    boards: NdArray[119, 8, 8]


class ChessOutput(BaseDoc):
    action: NdArray[4672]
    score: float


class ChessExecutor(Executor):
    def __init__(self, n_res_blocks, checkpoint, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.model = load_model(
            n_res_blocks=n_res_blocks,
            device=device,
            checkpoint=checkpoint,
            inference=True,
            compile=True,
        )
        print("model loaded.")

    @requests(on="/infer")
    @dynamic_batching(preferred_batch_size=5, timeout=100)
    def infer(self, docs: DocList[ChessInput], **kwargs) -> DocList[ChessOutput]:
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device, dtype=torch.bfloat16, cache_enabled=False
            ):
                inp = np.stack([doc.boards for doc in docs])
                inp = torch.from_numpy(inp).to(device=self.device)
                action, score = self.model(inp)

        action = list(action.cpu().numpy())
        score = score.cpu().numpy().flatten().tolist()

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

    with Deployment(
        uses=ChessExecutor,
        uses_with={
            "n_res_blocks": args.n_res_blocks,
            "device": args.device,
            "checkpoint": args.checkpoint,
        },
        port=args.port,
    ) as dep:
        dep.block()


if __name__ == "__main__":
    main()
