use std::fs::File;
use std::io::Write;
use crate::chess;

pub struct Trace {
    steps: Vec<(Option<chess::Move>, f32, Vec<(chess::Move, i32, f32)>)>,
    outcome: Option<chess::Outcome>,
}

#[allow(dead_code)]
impl Trace {
    pub fn new() -> Self {
        Trace { steps: Vec::new(), outcome: None }
    }

    pub fn save(&self, filename: &str) {
        let json = serde_json::json!({
            "steps": self.steps,
            "outcome": self.outcome,
        });

        let mut file = File::create(filename).unwrap();
        file.write_all(json.to_string().as_bytes()).unwrap();
    }

    #[allow(dead_code)]
    pub fn push(&mut self, mov: Option<chess::Move>, q_value: f32, children: Vec<(chess::Move, i32, f32)>) {
        self.steps.push((mov, q_value, children));
    }

    pub fn set_outcome(&mut self, outcome: chess::Outcome) {
        self.outcome = Some(outcome);
    }
}
