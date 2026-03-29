use tch::Tensor;

use crate::chess::post_process_distr;
use crate::chess::{BoardState, Color, Move, Step, _encode};
use crate::game::{tensor_to_f32, Game};
use crate::mcts::ArcRefNode;

pub trait TchModel {
    fn forward(&self, inp1: Tensor, inp2: Tensor) -> (Tensor, Tensor);
    fn device(&self) -> tch::Device;
}

#[allow(dead_code)]
pub struct ChessTS {
    pub model: tch::CModule,
    pub device: tch::Device,
}

#[cfg(feature = "aotinductor")]
pub struct ChessEP {
    pub model: aotinductor::ModelPackage,
    pub device: tch::Device,
}

impl TchModel for ChessTS {
    fn forward(&self, inp1: Tensor, inp2: Tensor) -> (Tensor, Tensor) {
        let out = self
            .model
            .forward_is(&[tch::jit::IValue::from(inp1), tch::jit::IValue::from(inp2)])
            .unwrap();
        <(Tensor, Tensor)>::try_from(out).unwrap()
    }

    fn device(&self) -> tch::Device {
        self.device
    }
}

impl Game<BoardState> for ChessTS {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        chess_tch_predict(self, node, state, argmax, false)
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

#[cfg(feature = "aotinductor")]
impl TchModel for ChessEP {
    fn forward(&self, inp1: Tensor, inp2: Tensor) -> (Tensor, Tensor) {
        let outs = self.model.run(&vec![inp1, inp2]);
        assert!(outs.len() == 2);
        let mut iter = outs.into_iter();
        let r1 = iter.next().unwrap();
        let r2 = iter.next().unwrap();
        (r1, r2)
    }

    fn device(&self) -> tch::Device {
        self.device
    }
}

#[cfg(feature = "aotinductor")]
impl Game<BoardState> for ChessEP {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        chess_tch_predict(self, node, state, argmax, false)
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

#[allow(dead_code)]
pub fn chess_tch_predict<M: TchModel>(
    chess: &M,
    node: &ArcRefNode<Step>,
    state: &BoardState,
    argmax: bool,
    return_full_distr: bool,
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

    let device = chess.device();

    let encoded_boards = Tensor::try_from(encoded_boards)
        .unwrap()
        .to_device_(device, tch::Kind::BFloat16, true, false)
        .permute([2, 0, 1])
        .unsqueeze(0);
    let encoded_meta = Tensor::try_from(encoded_meta)
        .unwrap()
        .to_device_(device, tch::Kind::BFloat16, true, false)
        .unsqueeze(0);

    let (full_distr, score) = chess.forward(encoded_boards, encoded_meta);

    let score = tensor_to_f32(score).unwrap();

    if !score.is_finite() {
        //let uuid = ShortUuid::generate();
        //let filename = format!("/tmp/{}.tensor", uuid);
        //inp_debug.save(std::path::Path::new(&filename)).unwrap();
        println!("Warning: score is bad {:?}", score);
        //println!("input tensor is saved as {}", filename);
    }

    let moves_distr = _get_move_distribution(full_distr, turn, &legal_moves, return_full_distr);
    let moves_distr = post_process_distr(moves_distr, argmax, turn);

    let next_steps = legal_moves
        .into_iter()
        .map(|m| Step(Some(m), !turn))
        .collect();

    return (next_steps, moves_distr, score);
}

fn _get_move_distribution(
    model_distr_output: Tensor,
    turn: Color,
    legal_moves: &Vec<Move>,
    return_full: bool,
) -> Vec<f32> {
    let full_distr = model_distr_output
        .to_dtype(tch::Kind::Float, true, false)
        .to_device(tch::Device::Cpu);

    if return_full {
        Vec::<f32>::try_from(full_distr.squeeze().exp()).unwrap()
    } else {
        // rotate if the next move is black
        let encoded_moves: Vec<i64> = legal_moves
            .iter()
            .map(|m| {
                if turn == Color::Black {
                    m.rotate().encode() as i64
                } else {
                    m.encode() as i64
                }
            })
            .collect();
        let encoded_moves = Tensor::from_slice(&encoded_moves);
        Vec::<f32>::try_from(full_distr.take(&encoded_moves).exp()).unwrap()
    }
}
