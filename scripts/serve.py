from argparse import ArgumentParser

from jina import Executor, requests, dynamic_batching, Deployment, Flow
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
import torch
import numpy as np

from module import load_model


class ChessInput(BaseDoc):
    boards: NdArray[8, 8, 112]
    meta: NdArray[8, 8, 7]


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
                print("received", len(docs), "docs")
                docs = [
                    np.concatenate(
                        (doc.boards.astype(np.float32), doc.meta.astype(np.float32)),
                        axis=2,
                    )
                    for doc in docs
                ]
                inp = np.stack(docs)
                inp = torch.from_numpy(inp.transpose(0, 3, 1, 2)).to(device=self.device)
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
