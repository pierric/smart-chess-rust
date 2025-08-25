use crate::mcts::ArcRefNode;
use ndarray::{Array1, Array3};
use tch::Tensor;

pub trait Game<S>
where
    S: State,
{
    fn predict(
        &self,
        node: &ArcRefNode<S::Step>,
        state: &S,
        argmax: bool,
    ) -> (Vec<S::Step>, Vec<f32>, f32);

    fn reverse_q(&self, node: &ArcRefNode<S::Step>) -> bool;
}

pub trait TchModel {
    fn forward(&self, input: Tensor) -> (Tensor, Tensor);
    fn device(&self) -> tch::Device;
}

pub trait State {
    type Step;
    fn dup(&self) -> Self;
    fn advance(&mut self, step: &Self::Step);
}

pub fn post_process_distr(distr: Vec<f32>, argmax: bool) -> Vec<f32> {
    if argmax {
        let i: usize = distr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let mut distr = vec![0.; distr.len()];
        distr[i] = 1.;
        distr
    } else {
        let sum = distr.iter().sum::<f32>() + 1e-5;

        if !sum.is_finite() {
            println!("Warning: {:?}", distr);
        }

        if sum < 0.5 {
            println!("Warning: move distribution sums up to only {}", sum);
        }

        distr.iter().map(|x| x / sum).collect()
    }
}

pub fn prepare_tensors(boards: Array3<i8>, meta: Array1<i32>, device: tch::Device) -> Tensor {
    // copy to device in sync mode
    // otherwise may cause corruption in data (mps)
    let encoded_boards = Tensor::try_from(boards)
        .unwrap()
        .pin_memory(tch::Device::Cuda(0))
        .to_device_(device, tch::Kind::BFloat16, true, false)
        .permute([2, 0, 1]);
    let encoded_meta = Tensor::try_from(meta)
        .unwrap()
        .pin_memory(tch::Device::Cuda(0))
        .to_device_(device, tch::Kind::BFloat16, true, false)
        //.repeat([8, 8, 1]);
        .repeat_interleave_self_int(64, None, None)
        .reshape([7, 8, 8]);

    Tensor::cat(&[encoded_boards, encoded_meta], 0)
        .contiguous() // aot model expects contiguous tensor
        .unsqueeze(0)
}

pub fn tensor_to_f32(t: Tensor) -> Option<f32> {
    let v = t.to_dtype(tch::Kind::Float, true, false);
    f32::try_from(&v).ok()
}
