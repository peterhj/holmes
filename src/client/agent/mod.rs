use board::{Stone, Action};

use std::sync::{Arc, Barrier};
use std::sync::mpsc::{Sender, Receiver};
use std::thread::{JoinHandle, spawn};

pub mod parallel_search;

#[derive(Clone, Copy, Debug)]
pub enum AgentMsg {
  StartMatch{our_stone: Stone, board_size: i32, main_time_secs: i32, byoyomi_time_secs: i32},
  CheckTime,
  RecvTime,
  SubmitAction{turn: Stone, action: Action},
  RecvAction{move_number: i32, turn: Stone, action: Action},
  FinishMatch,
}

pub trait AsyncAgent {
  fn spawn_runloop(barrier: Arc<Barrier>, agent_in_rx: Receiver<AgentMsg>, agent_out_tx: Sender<AgentMsg>) -> JoinHandle<()>;
}

pub struct HelloAsyncAgent;

impl AsyncAgent for HelloAsyncAgent {
  fn spawn_runloop(barrier: Arc<Barrier>, agent_in_rx: Receiver<AgentMsg>, agent_out_tx: Sender<AgentMsg>) -> JoinHandle<()> {
    spawn(move || {
      let mut started_match = false;
      let mut match_our_stone = None;
      loop {
        match agent_in_rx.recv() {
          Ok(AgentMsg::StartMatch{our_stone, ..}) => {
            println!("DEBUG: agent: start match");
            started_match = true;
            match_our_stone = Some(our_stone);
          }
          Ok(AgentMsg::RecvTime) => {
            println!("DEBUG: agent: recv time");
            assert!(started_match);
          }
          Ok(AgentMsg::RecvAction{turn, ..}) => {
            println!("DEBUG: agent: recv action");
            assert!(started_match);
            if turn != match_our_stone.unwrap() {
              agent_out_tx.send(AgentMsg::SubmitAction{
                turn:     match_our_stone.unwrap(),
                action:   Action::Pass,
              }).unwrap();
            }
          }
          Ok(AgentMsg::FinishMatch) => {
            println!("DEBUG: agent: finish match");
            break;
          }
          _ => {}
        }
      }
      barrier.wait();
    })
  }
}