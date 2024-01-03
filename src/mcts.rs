use crate::game::{Game, State};
use rand::thread_rng;
use rand_distr::{Dirichlet, Distribution};
use recursive_reference::*;
use std::iter::Sum;
use std::ptr::NonNull;
use std::ops::Deref;

pub struct Node<T> {
    pub step: T,
    pub q_value: f32,
    pub num_act: i32,
    pub parent: Option<NonNull<Node<T>>>,
    pub children: Vec<Box<Node<T>>>,
}

pub struct CursorMut<T> {
    pub current: NonNull<Node<T>>,
}

fn uct(
    sqrt_total_num_vis: f32,
    prior: f32,
    move_q: f32,
    move_n_act: i32,
    reverse_q: bool,
    cpuct: f32,
) -> f32 {
    const EPSILON: f32 = 1e-4;

    let average_award: f32 =
        move_q / (move_n_act as f32 + EPSILON) * (if reverse_q { -1. } else { 1. });

    // plus EPSILON to ensure that exploration factor isn't zero
    // in case q and n_act are zero, the choice will fully based on the prior
    let exploration: f32 =
        (sqrt_total_num_vis + EPSILON) / (1. + move_n_act as f32) * cpuct * prior;
    return average_award + exploration;
}

fn find_max<T>(collection: T) -> Option<usize>
where
    T: Iterator,
    T::Item: PartialOrd,
{
    // will raise an error in unwrap() if nan or inf occurs
    return collection
        .enumerate()
        .max_by(|u, v| u.1.partial_cmp(&v.1).unwrap())
        .map(|p| p.0);
}

fn backward<T>(mut ptr: RecRef<Node<T>>, reward: f32) {
    loop {
        ptr.q_value += reward;
        if RecRef::pop(&mut ptr).is_none() {
            break;
        }
    }

    // updating the root isn't super necessary, as the root
    // does participate in select during one mcts step
    ptr.q_value += reward;
}

fn select<'a, G, S>(
    game: &G,
    node: &'a mut Node<S::Step>,
    state: &mut S,
    cpuct: f32,
) -> (RecRef<'a, Node<S::Step>>, Vec<S::Step>, f32)
where
    G: Game<S>,
    S: State,
{
    //Descend in the tree until some leaf, exploiting the knowledge to choose
    //the best child each time.
    let mut ptr: RecRef<Node<S::Step>> = RecRef::new(node);

    loop {
        let node = ptr.deref();
        let (steps, prior, outcome) = game.predict(node, &state, false);
        let reverse_q = game.reverse_q(node);

        ptr.num_act += 1;

        if ptr.children.is_empty() || steps.is_empty() {
            // either game is end or yet to be explored
            return (ptr, steps, outcome);
        }

        let best = if ptr.children.len() == 1 {
            0
        } else {
            // otherwise, explore by the predicted distrubtion + the Dir(0.03) noise
            // https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
            if prior.len() < 2 {
                println!("Error: {:?} {:?} {:?}", ptr.children.len(), steps.len(), prior.len());
                println!("len: {:?}", RecRef::size(&ptr));
            }
            let dir = Dirichlet::<f64>::new_with_size(0.03, prior.len()).unwrap();
            let prior_rand: Vec<f32> = prior
                .iter()
                .zip(dir.sample(&mut thread_rng()))
                .map(|(p, n)| p * 0.75 + n as f32 * 0.25)
                .collect();
            let sqrt_total_num_vis =
                f32::sqrt(i32::sum(ptr.children.iter().map(|c| c.num_act)) as f32);
            let uct_children: Vec<f32> = prior_rand
                .iter()
                .zip(ptr.children.iter())
                .map(|(prior, child)| {
                    uct(
                        sqrt_total_num_vis,
                        *prior,
                        child.q_value,
                        child.num_act,
                        reverse_q,
                        cpuct,
                    )
                })
                .collect();
            // let uct: Vec<(f32, f32, i32, f32)> = prior_rand.iter().zip(ptr.children.iter()).zip(uct_children.iter()).map(|((p, c), u)| (*p, c.q_value, c.num_act, *u)).collect();
            // println!("sqrt total n {:?}", sqrt_total_num_vis);
            // println!("uct {:?}", uct);
            find_max(uct_children.into_iter()).unwrap()
        };

        state.advance(&ptr.children[best].step);
        RecRef::extend(&mut ptr, |node: &mut Node<S::Step>| {
            node.children[best].as_mut()
        });
    }
}

pub fn mcts<G, S>(
    game: &G,
    node: &mut Node<S::Step>,
    state: &S,
    n_rollout: i32,
    cpuct: Option<f32>,
) where
    G: Game<S>,
    S: State,
{
    let default_cpuct: f32 = 1.2;
    let cpuct = cpuct.unwrap_or(default_cpuct);

    for _ in 0..n_rollout {
        let mut local_state = state.dup();
        let (mut path, steps, reward) = select(game, node, &mut local_state, cpuct);

        // path points at a leaf node, either game is done, or it isn't finished
        path.children = steps
            .into_iter()
            .map(|step| {
                Box::new(Node {
                    step: step,
                    q_value: 0.,
                    num_act: 0,
                    parent: Some(NonNull::from(&*path)),
                    children: Vec::new(),
                })
            })
            .collect();

        backward(path, reward);
    }
}

impl<T> Node<T> {
    pub fn as_cursor_mut(&mut self) -> CursorMut<T> {
        CursorMut {
            current: NonNull::from(self),
        }
    }
}

impl<T> CursorMut<T> {
    pub fn current(&mut self) -> &mut Node<T> {
        unsafe { self.current.as_mut() }
    }

    pub fn move_children(&mut self, index: usize) {
        let child = unsafe { self.current.as_ref().children[index].as_ref() };
        self.current = NonNull::from(child);
    }
}
