use crate::chess::{BoardState, Color, Move, Step, _encode};
use crate::docarray::{DocListProto, DocProto};
use crate::game::{post_process_distr, Game};
use crate::jina::data_request_proto::data_content_proto::Documents::Docs;
use crate::jina::data_request_proto::DataContentProto;
use crate::jina::jina_single_data_request_rpc_client::JinaSingleDataRequestRpcClient;
use crate::jina::{DataRequestProto, HeaderProto};
use crate::mcts::ArcRefNode;
use ndarray::{Array, ArrayBase, ArrayD, Dimension, Ix};
use std::collections::HashMap;

pub struct ChessService {
    pub endpoint: String,
    runtime: tokio::runtime::Runtime,
}

impl ChessService {
    #[allow(dead_code)]
    pub fn new(endpoint: &str) -> Self {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        Self {
            endpoint: String::from(endpoint),
            runtime: runtime,
        }
    }
}

impl Game<BoardState> for ChessService {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        let legal_moves = state.legal_moves();

        if legal_moves.is_empty() {
            // the model always tell the score at the white side
            let outcome = match state.outcome().unwrap().winner {
                None => 0.0,
                Some(Color::White) => 1.0,
                Some(Color::Black) => -1.0,
            };
            return (Vec::new(), Vec::new(), outcome);
        }

        let (encoded_boards, encoded_meta) = _encode(&node, state);
        let turn = node.borrow().step.1;

        assert!(turn == state.turn());

        #[allow(dead_code)]
        #[derive(Debug)]
        enum Failure {
            ConnectionError(tonic::transport::Error),
            GrpcError(tonic::Status),
        }

        self.runtime
            .block_on(async {
                let mut client = JinaSingleDataRequestRpcClient::connect(self.endpoint.clone())
                    .await
                    .map_err(Failure::ConnectionError)?;
                let req = tonic::Request::new(DataRequestProto {
                    header: Some(HeaderProto {
                        request_id: String::from("infer000"),
                        status: None,
                        exec_endpoint: Some(String::from("/infer")),
                        target_executor: None,
                        timeout: None,
                    }),
                    parameters: None,
                    routes: vec![],
                    data: Some(DataContentProto {
                        documents: Some(Docs(DocListProto {
                            docs: vec![DocProto {
                                data: HashMap::from([
                                    (
                                        String::from("boards"),
                                        as_nodeproto(ndarray_to_protobuf(&encoded_boards)),
                                    ),
                                    (
                                        String::from("meta"),
                                        as_nodeproto(ndarray_to_protobuf(&encoded_meta)),
                                    ),
                                ]),
                            }],
                        })),
                    }),
                });
                let res = client
                    .process_single_data(req)
                    .await
                    .map_err(Failure::GrpcError)?;
                let docs = match res.into_inner().data.unwrap().documents {
                    Some(Docs(docs)) => docs.docs,
                    _ => {
                        panic!("response not as DocListProto");
                    }
                };
                assert!(docs.len() == 1);
                let doc = &docs[0].data;

                let action = doc.get("action").unwrap();

                let action = action
                    .content
                    .as_ref()
                    .and_then(|c| match c {
                        node_proto::Content::Ndarray(v) => ndarray_from_protobuf(v),
                        _ => None,
                    })
                    .expect("action should be a ndarray");

                let score = doc
                    .get("score")
                    .and_then(|c| match c.content {
                        Some(node_proto::Content::Float(v)) => Some(v as f32),
                        _ => None,
                    })
                    .expect("score should be a float");

                let moves_distr = _get_move_distribution(action, turn, &legal_moves);
                let moves_distr = post_process_distr(moves_distr, argmax);

                let next_steps = legal_moves
                    .into_iter()
                    .map(|m| Step(Some(m), !turn))
                    .collect();

                Ok::<_, Failure>((next_steps, moves_distr, score))
            })
            .unwrap()
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

use crate::docarray::{node_proto, DenseNdArrayProto, NdArrayProto, NodeProto};

trait KnownDType {
    fn name() -> &'static str;
}

impl KnownDType for f32 {
    fn name() -> &'static str {
        "<f4"
    }
}

impl KnownDType for i8 {
    fn name() -> &'static str {
        "b"
    }
}

impl KnownDType for i32 {
    fn name() -> &'static str {
        "<i4"
    }
}

fn slice_as_bytes<A: bytemuck::NoUninit>(data: &[A]) -> Vec<u8> {
    Vec::from(bytemuck::cast_slice(data))
}

fn vec_from_bytes<A: Clone + bytemuck::Pod>(vec: &Vec<u8>) -> Vec<A> {
    Vec::from(bytemuck::cast_slice(vec.as_slice()))
}

fn ndarray_to_protobuf<A: KnownDType + bytemuck::NoUninit, D: Dimension>(
    array: &Array<A, D>,
) -> NdArrayProto {
    let data = slice_as_bytes(array.as_slice().unwrap());
    NdArrayProto {
        dense: Some(DenseNdArrayProto {
            buffer: data,
            shape: Vec::from(array.shape())
                .into_iter()
                .map(|u| u as u32)
                .collect(),
            dtype: String::from(A::name()),
        }),
        parameters: None,
    }
}

fn as_nodeproto(obj: NdArrayProto) -> NodeProto {
    NodeProto {
        content: Some(node_proto::Content::Ndarray(obj)),
        docarray_type: Some(node_proto::DocarrayType::Type(String::from("ndarray"))),
    }
}

fn ndarray_from_protobuf<A: KnownDType + bytemuck::Pod>(proto: &NdArrayProto) -> Option<ArrayD<A>> {
    let dense = proto.dense.as_ref()?;
    let shape: Vec<Ix> = dense.shape.iter().map(|v| *v as Ix).collect();
    let data = vec_from_bytes(&dense.buffer);
    if A::name() != dense.dtype {
        None
    } else {
        ArrayBase::from_shape_vec(shape, data).ok()
    }
}

fn _get_move_distribution(distr: ArrayD<f32>, turn: Color, legal_moves: &Vec<Move>) -> Vec<f32> {
    legal_moves
        .iter()
        .map(|m| {
            let idx = match turn {
                Color::Black => m.rotate().encode() as usize,
                Color::White => m.encode() as usize,
            };
            distr.get(idx).unwrap().exp()
        })
        .collect()
}
