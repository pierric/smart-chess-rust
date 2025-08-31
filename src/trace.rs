use serde::Serialize;
use std::fs::File;
use std::io::Write;

pub struct Trace<M, O> {
    steps: Vec<(Option<M>, f32, Vec<(M, i32, f32, f32)>)>,
    outcome: Option<O>,
}

#[allow(dead_code)]
impl<M, O> Trace<M, O>
where
    M: Serialize,
    O: Serialize,
{
    pub fn new() -> Self {
        Trace {
            steps: Vec::new(),
            outcome: None,
        }
    }

    pub fn save(&self, filename: &str) {
        let json = serde_json::json!({
            "steps": self.steps,
            "outcome": self.outcome,
        });

        let mut file = File::create(filename).unwrap();
        let json_str = serde_json::to_string_pretty(&json).unwrap();
        file.write_all(json_str.as_bytes()).unwrap();
    }

    #[allow(dead_code)]
    pub fn push(&mut self, mov: Option<M>, q_value: f32, children: Vec<(M, i32, f32, f32)>) {
        self.steps.push((mov, q_value, children));
    }

    pub fn set_outcome(&mut self, outcome: O) {
        self.outcome = Some(outcome);
    }
}
