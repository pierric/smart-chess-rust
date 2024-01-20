use crate::game::{Game, State};
use rand::{thread_rng, Rng};
use rand_distr::{Dirichlet, Distribution};
use recursive_reference::*;
use std::iter::Sum;
use std::ptr::NonNull;
use std::ops::Deref;
use rand::distributions::WeightedIndex;

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
    let average_award: f32 =
        move_q / (move_n_act as f32 + 1e-4) * (if reverse_q { -1. } else { 1. });

    // plus 0.01 to ensure that exploration factor isn't zero
    // in case q and n_act are zero, the choice will fully based on the prior
    let exploration: f32 =
        (sqrt_total_num_vis + 0.01) / (1. + move_n_act as f32) * cpuct * prior;
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

fn backward<T>(mut ptr: RecRef<Node<T>>, reward: f32) where T: std::fmt::Debug {
    // let mut last: String = String::from("");
    // use string_builder::Builder;
    // let mut builder = Builder::default();

    loop {
        ptr.q_value += reward;

        // builder.append(format!("{:?} ", ptr.step));
        // let sz = RecRef::size(&ptr);
        // if sz == 2 {
        //     last = format!("{:?}", ptr.step);
        // }

        if RecRef::pop(&mut ptr).is_none() {
            break;
        }
    }

    // if last == "(Some(Move[f2f4]), Black)" {
    //     println!("backward: reward {} last {:?} path {:?}", reward, last, builder.string());
    // }
}

fn select<'a, G, S>(
    game: &G,
    node: &'a mut Node<S::Step>,
    state: &mut S,
    cpuct: f32,
    noise: &Vec<f64>,
) -> (RecRef<'a, Node<S::Step>>, Vec<S::Step>, f32)
where
    G: Game<S>,
    S: State,
    S::Step: std::fmt::Debug,
{
    //Descend in the tree until some leaf, exploiting the knowledge to choose
    //the best child each time.
    let mut ptr: RecRef<Node<S::Step>> = RecRef::new(node);
    let mut root = true;

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
            let prior_rand: Vec<f32> = if !root {prior} else {
                prior.iter()
                    .zip(noise.iter())
                    .map(|(p, n)| p * 0.75 + *n as f32 * 0.25)
                    .collect()
            };
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
            find_max(uct_children.into_iter()).unwrap()
        };

        state.advance(&ptr.children[best].step);
        RecRef::extend(&mut ptr, |node: &mut Node<S::Step>| {
            node.children[best].as_mut()
        });
        root = false;
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
    S::Step: std::fmt::Debug,
{
    let default_cpuct: f32 = 1.2;
    let cpuct = cpuct.unwrap_or(default_cpuct);

    let num_legal_moves = state.legal_moves().len();
    let dirichlet = Dirichlet::<f64>::new_with_size(0.03, num_legal_moves).unwrap();
    let noise = dirichlet.sample(&mut thread_rng());

    for _ in 0..n_rollout {
        let mut local_state = state.dup();
        let (mut path, steps, reward) = select(game, node, &mut local_state, cpuct, &noise);

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

pub fn step<S>(
    cursor: &mut CursorMut<S::Step>,
    state: &mut S,
    temp: f32,
) -> Option<S::Step> where S: State, S::Step: Copy {
    let num_act_vec: Vec<_> = cursor.current().children.iter().map(|a| a.num_act).collect();

    if num_act_vec.len() == 0 {
        return None
    }

    let choice: usize = if temp == 0.0 {
        let max = num_act_vec.iter().max().unwrap();
        let indices: Vec<usize> = num_act_vec.iter().enumerate().filter(|a| a.1 == max).map(|a| a.0).collect();
        let n: usize = thread_rng().gen_range(0..indices.len());
        indices[n]
    } else {
        let power = 1.0 / temp;
        let weights = WeightedIndex::new(num_act_vec.iter().map(|n| (*n as f32).powf(power))).unwrap();
        weights.sample(&mut thread_rng())
    };

    cursor.move_children(choice);
    let step = &cursor.current().step;
    State::advance(state, step);
    Some(*step)
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
