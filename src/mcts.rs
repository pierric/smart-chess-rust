use std::iter::Sum;
use recursive_reference::*;
use rand::thread_rng;
use rand_distr::{Distribution, Dirichlet};
use crate::game::Game;

pub struct Node<T> {
    pub step: T,
    pub q_value: f32,
    pub num_act: i32,
    children: Vec<Box<Node<T>>>,
}

fn uct(sqrt_total_num_vis: f32, prior: f32, move_q: f32, move_n_act:i32, reverse_q: bool, cpuct: f32) -> f32 {
    const EPSILON: f32 = 1e-4;

    let average_award: f32 = move_q / (move_n_act as f32 + EPSILON) * (if reverse_q {-1.} else {1.});

    // plus EPSILON to ensure that exploration factor isn't zero
    // in case q and n_act are zero, the choice will fully based on the prior
    let exploration: f32 = (sqrt_total_num_vis + EPSILON) / (1. + move_n_act as f32) * cpuct * prior;
    return average_award + exploration
}

fn find_max<T>(collection: T) -> Option<usize> where T: Iterator, T::Item: PartialOrd {
    // will raise an error in unwrap() if nan or inf occurs
    return collection.enumerate().max_by(|u, v| u.1.partial_cmp(&v.1).unwrap()).map(|p| p.0);
}

fn backward<T>(ptr: &mut RecRef<Node<T>>, reward: f32) {
    loop {
        ptr.q_value += reward;
        if RecRef::pop(ptr).is_none() {
            break;
        }
    }

    ptr.q_value += reward;
}

impl<T> Node<T> {
    fn select<G>(& mut self, game: &G, reverse_q: bool, cpuct: f32) -> (RecRef<Node<T>>, f32)
        where G: Game<T> {
        //Descend in the tree until some leaf, exploiting the knowledge to choose
        //the best child each time.
        let mut ptr: RecRef<Node<T>> = RecRef::new(self);

        loop {
            // the more times the node is visited, the less likely to be explored
            let (moves, prior, outcome) = game.predict(&*ptr, false);

            if ptr.children.is_empty() {
                // reaching a leaf node, either game is done, or it isn't finished, then
                // we expand the node (in the future if ever chosen the same node, will go
                // one level deeper). In both case, the predicted outcome is the result.
                for step in moves {
                    ptr.children.push(Box::new(Node {
                        step: step,
                        q_value: 0.,
                        num_act: 0,
                        children: Vec::new(),
                    }))
                }

                return (ptr, outcome)
            }

            // otherwise, explore by the predicted distrubtion + the Dir(0.03) noise
            // https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
            let dir = Dirichlet::new_with_size(0.03, prior.len()).unwrap();
            let prior_rand = prior.into_iter()
                .zip(dir.sample(&mut thread_rng()))
                .map(|(p, n)| p * 0.75 + n * 0.25);
            let sqrt_total_num_vis = f32::sqrt(i32::sum(ptr.children.iter().map(|c| c.num_act)) as f32);
            let uct_children: Vec<f32> = prior_rand.zip(ptr.children.iter()).map(
                |(prior, child)| {
                    uct(sqrt_total_num_vis, prior, child.q_value, child.num_act, reverse_q, cpuct)
                }
            ).collect();
            RecRef::extend(
                &mut ptr,
                |node: &mut Node<T>| node.children[find_max(uct_children.into_iter()).unwrap()].as_mut()
            );
            ptr.num_act += 1;
        }
    }

    pub fn mcts<G>(&mut self, game: &G, n_rollout: i32, reverse_q: bool, cpuct: Option<f32>)
        where G: Game<T> {
        const CPUCT: f32 = 1.2;
        let cpuct = cpuct.unwrap_or(CPUCT);

        for _ in 0..n_rollout {
            let (mut path, reward) = self.select(game, reverse_q, cpuct);
            backward(&mut path, reward);
        }
    }
}
