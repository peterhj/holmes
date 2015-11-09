use board::{Stone, Action};

pub mod convnet;
pub mod search;

pub trait Agent {
  fn reset(&mut self);
  fn board_dim(&mut self, board_dim: usize);
  fn komi(&mut self, komi: f32);
  fn player(&mut self, stone: Stone);

  fn apply_action(&mut self, turn: Stone, action: Action);
  fn undo(&mut self);
  fn act(&mut self, turn: Stone) -> Action;
}
