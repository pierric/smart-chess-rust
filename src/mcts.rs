use crate::game::{Game, State};
use rand::distributions::WeightedIndex;
use rand::thread_rng;
use rand_distr::{Dirichlet, Distribution};
use serde::{Serialize, Serializer};
use serde::ser::{SerializeStruct, SerializeSeq};
use std::iter::Sum;
use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::cell::{RefCell, Ref, RefMut};
use std::fmt::Display;

pub type ArcRefNode<T> = Arc<RefCell<Node<T>>>;
type WeakRefNode<T> = Weak<RefCell<Node<T>>>;

pub struct Node<T> {
    pub step: T,
    pub depth: u32,
    pub q_value: f32,
    pub num_act: i32,
    pub parent: Option<WeakRefNode<T>>,
    pub children: Vec<ArcRefNode<T>>,
}

unsafe impl<T: Send> Send for Node<T> {}

pub struct ChildrenList<'a, T>(&'a Vec<ArcRefNode<T>>);

impl<'a, T: Serialize> Serialize for ChildrenList<'a, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for elem in self.0 {
            seq.serialize_element(&*elem.borrow())?;
        }
        seq.end() 
    }
}

impl<T: Serialize> Serialize for Node<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut node = serializer.serialize_struct("Node", 5)?;
        node.serialize_field("step", &self.step)?;
        node.serialize_field("depth", &self.depth)?;
        node.serialize_field("q", &self.q_value)?;
        node.serialize_field("num_act", &self.num_act)?;
        node.serialize_field("children", &ChildrenList(&self.children))?;
        node.end()
    }
}

#[derive(Clone)]
pub struct Cursor<T>(ArcRefNode<T>);

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
    let exploration: f32 = (sqrt_total_num_vis + 0.01) / (1. + move_n_act as f32) * cpuct * prior;
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

fn backward<T>(path: &VecDeque<ArcRefNode<T>>, reward: f32)
where
    T: Display,
{
    for node in path {
        let mut v: RefMut<'_, _> = node.borrow_mut();
        v.num_act += 1;
        v.q_value += reward;
    }

    // let mut last: String = String::from("");
    // use string_builder::Builder;
    // let mut builder = Builder::default();

    // loop {
    //     ptr.q_value += reward;

    //     // builder.append(format!("{:?} ", ptr.step));
    //     // let sz = RecRef::size(&ptr);
    //     // if sz == 2 {
    //     //     last = format!("{:?}", ptr.step);
    //     // }

    //     if RecRef::pop(&mut ptr).is_none() {
    //         break;
    //     }
    // }

    // if last == "(Some(Move[f2f4]), Black)" {
    //     println!("backward: reward {} last {:?} path {:?}", reward, last, builder.string());
    // }
}

fn get_noise(size: usize) -> Option<Vec<f64>> {
    if size < 2 {
        None
    } else {
        let dirichlet = Dirichlet::<f64>::new_with_size(0.3, size).unwrap();
        Some(dirichlet.sample(&mut thread_rng()))
    }
}

fn select<'a, G, S>(
    game: &G,
    node: &ArcRefNode<S::Step>,
    state: &mut S,
    cpuct: f32,
    with_noise: bool,
) -> (VecDeque<ArcRefNode<S::Step>>, Vec<S::Step>, f32)
where
    G: Game<S> + ?Sized,
    S: State,
    S::Step: Display,
{
    //Descend in the tree until some leaf, exploiting the knowledge to choose
    //the best child each time.
    let mut path: VecDeque<ArcRefNode<S::Step>> = VecDeque::from([node.clone()]);

    loop {
        let (recent_node, path_len) = {
            (path.back().unwrap().clone(), path.len())
        };

        let (steps, prior, outcome) = game.predict(&recent_node, &state, false);
        let reverse_q = game.reverse_q(&recent_node);
        let children = &recent_node.borrow().children;

        if children.is_empty() || steps.is_empty() {
            // either game is end or yet to be explored
            // the path is the from root (the current play) to the leaf where
            // all are what have been explored.
            // NOTE:act_num update of the root is not increased.
            // later, the root will also be skipped for the reward.
            // but we keep it in the path because it might be the only node
            // (that have been explored), and we will expand the tree from
            // there
            return (path, steps, outcome);
        }

        let best_child = if children.len() == 1 {
            &children[0]
        } else {
            let is_root = path_len == 1;

            let prior_rand: Vec<f32> = if !is_root || !with_noise {
                prior.clone()
            } else {
                match get_noise(steps.len()) {
                    None => prior.clone(),
                    Some(noise) => prior
                    .iter()
                    .zip(noise.iter())
                    .map(|(p, n)| p * 0.75 + *n as f32 * 0.25)
                    .collect(),
                }
            };

            let sqrt_total_num_vis =
                f32::sqrt(i32::sum(children.iter().map(|c| c.borrow().num_act)) as f32);
            let uct_children: Vec<f32> = prior_rand
                .iter()
                .zip(children.iter())
                .map(|(prior, child)| {
                    uct(
                        sqrt_total_num_vis,
                        *prior,
                        child.borrow().q_value,
                        child.borrow().num_act,
                        reverse_q,
                        cpuct,
                    )
                })
                .collect();
            if uct_children.is_empty() || uct_children.iter().any(|v| !v.is_finite()) ||
                uct_children.len() != children.len()
            {
                println!("path len: {}", path_len);
                println!("len uct: {}", uct_children.len());
                println!("len children: {}", children.len());
                println!("len prior: {}", prior.len());
                println!("prior: {:?}", prior_rand);
                println!("uct: {:?}", uct_children);
                println!("sqrt total: {:?}", sqrt_total_num_vis);
                panic!("!!!! CHECK the values.");
            }
            let idx = find_max(uct_children.into_iter()).unwrap();
            &children[idx]
        };

        state.advance(&best_child.borrow().step);
        path.push_back(best_child.clone());
    }
}

pub fn mcts<G, S>(game: &G, node: &ArcRefNode<S::Step>, state: &S, n_rollout: i32, cpuct: Option<f32>, with_noise: bool)
where
    G: Game<S> + ?Sized,
    S: State,
    S::Step: Display,
{
    let default_cpuct: f32 = 1.2;
    let cpuct = cpuct.unwrap_or(default_cpuct);

    //let num_legal_moves = state.legal_moves().len();
    //let noise = if /*node.depth >= 4 || */ num_legal_moves < 2 {
    //    None
    //} else {
    //    let dirichlet = Dirichlet::<f64>::new_with_size(0.3, num_legal_moves).unwrap();
    //    Some(dirichlet.sample(&mut thread_rng()))
    //};

    for _ in 0..n_rollout {
        let mut local_state = state.dup();
        let (mut path, steps, reward) = select(game, node, &mut local_state, cpuct, with_noise);

        // path points at a leaf node, either game is done, or it isn't finished
        let cur: &Arc<_> = path.back().unwrap();
        let depth = cur.borrow().depth;
        let children = steps
            .into_iter()
            .map(|step| {
                Arc::new(RefCell::new(Node {
                    step: step,
                    depth: depth + 1,
                    q_value: 0.,
                    num_act: 0,
                    parent: Some(Arc::downgrade(cur)),
                    children: Vec::new(),
                }))
            })
            .collect();
        cur.borrow_mut().children = children;

        // skip the root (not participating the decision) for reward.
        path.pop_front();
        backward(&mut path, reward);
    }
}

#[allow(dead_code)]
pub fn step<S>(cursor: &mut Cursor<S::Step>, state: &mut S, temp: f32) -> Option<S::Step>
where
    S: State,
    S::Step: Copy,
    S::Step: Display,
{
    let num_act_vec: Vec<_> = cursor
        .current()
        .children
        .iter()
        .map(|a| a.borrow().num_act)
        .collect();

    if num_act_vec.len() == 0 {
        return None;
    }

    let choice: usize = if temp == 0.0 {
        let max = num_act_vec.iter().max().unwrap();
        num_act_vec
            .iter()
            .position(|v| v == max)
            .unwrap()
    } else {
        let power = 1.0 / temp;
        let weights =
            WeightedIndex::new(num_act_vec.iter().map(|n| (*n as f32).powf(power))).unwrap();
        weights.sample(&mut thread_rng())
    };

    cursor.navigate_down(choice);
    // discarding the exploration of the sub-nodes. It isn't
    // clearly said in AlphaZero's paper, but a restart helps
    // the dirichlet noise to apply.
    cursor.current_mut().children = vec![];

    let step = &cursor.current().step;
    State::advance(state, step);
    Some(*step)
}

impl<T> Cursor<T> {
    pub fn new(data: Node<T>) -> (Self, ArcRefNode<T>) {
        // don't move this root! necessary to ensure the root alive.
        // as the children has only a weak back reference, the parent might
        // be recycled.
        let arc = Arc::new(RefCell::new(data));
        (Cursor(arc.clone()), arc)
    }

    pub fn from_arc(arc: ArcRefNode<T>) -> Self {
        Cursor(arc)
    }

    pub fn current(&self) -> Ref<'_, Node<T>> {
        self.0.borrow()
    }

    #[allow(dead_code)]
    pub fn current_mut(&self) -> RefMut<'_, Node<T>> {
        self.0.borrow_mut()
    }

    pub fn arc(&self) -> &ArcRefNode<T> {
        &self.0
    }

    #[allow(dead_code)]
    pub fn as_weak(&self) -> WeakRefNode<T> {
        Arc::downgrade(&self.0)
    }

    //#[allow(dead_code)]
    //pub fn as_mut(&mut self) -> &mut Node<T> {
    //    let mut_ref = Arc::get_mut(&mut self.0);
    //    if mut_ref.is_none() {
    //        panic!("Not possible to update because shared use")
    //    }
    //    mut_ref.unwrap()
    //}

    pub fn navigate_down(&mut self, index: usize) {
        let next = {
            let children = &self.0.borrow().children;
            if index > children.len() {
                panic!("navigating to an nonexistent child.");
            }
            children[index].clone()
        };
        self.0 = next;
    }

    pub fn navigate_up(&mut self) {
        let parent = self.0.borrow().parent.clone();
        match parent {
            None => {
                panic!("navigating to the parent, but already at the root");
            },
            Some(parent) => {
                let parent_arc = parent.upgrade();
                if parent_arc.is_none() {
                    panic!("navigating to the parent, but failed to turn the weak ref into an arc")
                }
                self.0 = parent_arc.unwrap();
            }
        }
    }
}

