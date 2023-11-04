use pyo3::prelude::*;

mod chess;
mod knightmoves;
mod underpromotions;
mod queenmoves;
mod mcts;
mod game;


fn get_python_path() -> Py<PyAny> {
    return Python::with_gil(|py| {
        py.import("sys").and_then(|m| m.getattr("path")).unwrap().into()
    });
}

fn main() {
    let r = Python::with_gil(|py| {
        let nn = py.import("nn")?.getattr("ChessModule")?.call0()?;
        Ok::<Py<PyAny>, PyErr>(nn.into())
    });


    match r {
        Ok(_) => {}
        Err(e) => {
            let path = get_python_path();
            println!("Error:\n {}\n when importing python module 'nn' from path: {}", e, path);
        }
    }
}
