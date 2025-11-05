use crate::chess::post_process_distr;
use crate::chess::{BoardState, Color, Move, Step, _encode};
use crate::game::Game;
use crate::mcts::ArcRefNode;
use ndarray::{Array1, Array3, Axis};
use std::ops::Index;

#[allow(dead_code)]
pub struct ChessOnnx {
    pub session: std::cell::RefCell<ort::session::Session>,
}

impl Game<BoardState> for ChessOnnx {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        let legal_moves = state.legal_moves();

        if legal_moves.is_empty() {
            let outcome = match state.outcome().unwrap().winner {
                None => 0.0,
                Some(Color::White) => 1.0,
                Some(Color::Black) => -1.0,
            };
            return (Vec::new(), Vec::new(), outcome);
        }

        let (encoded_boards, encoded_meta) = _encode(node, state);

        let turn = node.borrow().step.1;
        let depth = node.borrow().depth;

        let (moves_distr, score) = call_onnx_model(
            &mut *self.session.borrow_mut(),
            encoded_boards,
            encoded_meta,
            turn,
            &legal_moves,
        )
        .unwrap();

        let moves_distr = post_process_distr(moves_distr, argmax, (turn, depth));
        let next_steps = legal_moves
            .into_iter()
            .map(|m| Step(Some(m), !turn))
            .collect();
        return (next_steps, moves_distr, score);
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        node.borrow().step.1 == Color::Black
    }
}

fn call_onnx_model(
    session: &mut ort::session::Session,
    boards: Array3<i8>,
    meta: Array1<i32>,
    turn: Color,
    steps: &Vec<Move>,
) -> ort::Result<(Vec<f32>, f32)> {
    let boards = boards
        .map(|v| *v as f32)
        .permuted_axes([2, 0, 1])
        .insert_axis(Axis(0));
    let meta = meta.map(|v| *v as f32);
    let inp1 = ort::value::Tensor::from_array(boards)?;
    let inp2 = ort::value::Tensor::from_array(meta.insert_axis(Axis(0)))?;
    let out = session.run(ort::inputs!["boards" => inp1, "meta" => inp2])?;

    let full_distr: ort::value::TensorRef<f32> = out[0].downcast_ref()?;
    let score: ort::value::TensorRef<f32> = out[1].downcast_ref()?;
    let score = *(score.index([0, 0]));
    //let score = score * (if turn == Color::Black { -1. } else { 1. });

    // rotate if the next move is black
    let moves_distr: Vec<f32> = steps
        .iter()
        .map(|m| {
            let mi = if turn == Color::Black {
                m.rotate().encode() as i64
            } else {
                m.encode() as i64
            };
            full_distr.index([0, mi]).exp()
        })
        .collect();

    Ok((moves_distr, score))
}
