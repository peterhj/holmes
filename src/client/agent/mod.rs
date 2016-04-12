use board::{Stone, Point, Action};

use std::path::{PathBuf};
use std::sync::{Arc, Barrier};
use std::sync::mpsc::{Sender, Receiver};
use std::thread::{JoinHandle, spawn};

pub mod parallel_search;

#[derive(Clone, Debug)]
pub enum AgentMsg {
  Ready,
  RequestMatch{
    passive: bool,
    opponent: String,
    our_stone: Stone,
    board_size: i32,
    main_time_secs: i32,
    byoyomi_time_secs: i32,
  },
  AcceptMatch{
    passive: bool,
    opponent: String,
    our_stone: Stone,
    board_size: i32,
    main_time_secs: i32,
    byoyomi_time_secs: i32,
  },
  StartMatch{
    skip_as_black: bool,
    opponent: String,
    our_stone: Stone,
    board_size: i32,
    main_time_secs: i32,
    byoyomi_time_secs: i32,
  },
  CheckTime,
  RecvTime,
  SubmitAction{
    turn:   Stone,
    action: Action,
    // FIXME(20160316): just send current game result with every action
    // (including dead stones, live stones, territory, and est. outcome).
    set_dead_stones:  bool,
    dead_stones:  Vec<Vec<Point>>,
    live_stones:  Vec<Vec<Point>>,
    territory:    Vec<Vec<Point>>,
    outcome:      Option<Stone>,
  },
  RecvAction{
    turn:         Stone,
    action:       Action,
    move_number:  Option<i32>,
    time_left_s:  Option<i32>,
  },
  FinishMatch,
}

pub trait AsyncAgent {
  fn spawn_runloop(barrier: Arc<Barrier>, agent_in_rx: Receiver<AgentMsg>, agent_out_tx: Sender<AgentMsg>, load_save_path: Option<PathBuf>) -> JoinHandle<()>;
}

/*pub struct HelloAsyncAgent;

impl AsyncAgent for HelloAsyncAgent {
  fn spawn_runloop(barrier: Arc<Barrier>, agent_in_rx: Receiver<AgentMsg>, agent_out_tx: Sender<AgentMsg>, load_save_path: Option<PathBuf>) -> JoinHandle<()> {
    spawn(move || {
      let mut started_match = false;
      let mut match_our_stone = None;
      loop {
        match agent_in_rx.recv() {
          Ok(AgentMsg::AcceptMatch{our_stone, ..}) => {
            println!("DEBUG: agent: start match");
            started_match = true;
            match_our_stone = Some(our_stone);
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
}*/
