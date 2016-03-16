use board::{RuleSet, PlayerRank, Stone, Action};
use client::agent::{AgentMsg, AsyncAgent};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorkerBuilder, ConvnetPolicyWorker,
};
use search::parallel_tree::{
  MonteCarloSearchConfig,
  TreePolicyConfig,
  HorizonConfig,
  SharedTree,
  SearchWorkerConfig,
  SearchWorkerBatchConfig,
  MonteCarloSearchResult,
  ParallelMonteCarloSearchServer,
  ParallelMonteCarloSearch,
};
use txnstate::{TxnStateConfig, TxnState};
use txnstate::extras::{TxnStateNodeData};

use cuda::runtime::{CudaDevice};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::sync::{Arc, Barrier};
use std::sync::mpsc::{Sender, Receiver, TryRecvError};
use std::thread::{JoinHandle, sleep_ms, spawn};
use time::{Timespec, get_time};

enum AgentStateMachine {
  Reset,
  OurTurn,
  OpponentTurn,
  Terminated,
}

impl Default for AgentStateMachine {
  fn default() -> AgentStateMachine {
    AgentStateMachine::Reset
  }
}

struct AgentImpl {
  state_cfg:    TxnStateConfig,
  search_cfg:   MonteCarloSearchConfig,
  tree_cfg:     TreePolicyConfig,

  state_machine:    AgentStateMachine,
  //match_started:    bool,
  our_stone:        Option<Stone>,
  komi:             f32,
  main_time_s:      i32,
  byoyomi_time_s:   i32,
  //start_time:       Option<Timespec>,

  history:  Vec<(TxnState<TxnStateNodeData>, Stone, Action)>,
  ply:      usize,
  state:    TxnState<TxnStateNodeData>,
  tree:     Option<SharedTree>,

  rng:      Xorshiftplus128Rng,
  server:   ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
}

impl AgentImpl {
  pub fn new() -> AgentImpl {
    let state_cfg = TxnStateConfig{
      rules:    RuleSet::KgsJapanese.rules(),
      ranks:    [PlayerRank::Dan(9), PlayerRank::Dan(9)],
      komi:     6.5,
    };
    let search_cfg = MonteCarloSearchConfig{
      batch_size:   256,
      // FIXME(20160310): Number of rollouts should be specified during search.
      num_rollouts: 0,
    };
    let tree_cfg = TreePolicyConfig{
      horizon_cfg:  HorizonConfig::Fixed{max_horizon: 20},
      visit_thresh: 1,
      mc_scale:     1.0,
      prior_equiv:  16.0,
      rave:         false,
      rave_equiv:   0.0,
    };
    let num_workers = CudaDevice::count().unwrap();
    let rounded_batch_size = (search_cfg.batch_size + num_workers - 1) / num_workers * num_workers;
    let worker_batch_capacity = rounded_batch_size / num_workers;
    AgentImpl{
      state_cfg:    state_cfg,
      search_cfg:   search_cfg,
      tree_cfg:     tree_cfg,
      state_machine:    AgentStateMachine::Reset,
      our_stone:        None,
      komi:             6.5,
      main_time_s:      0,
      byoyomi_time_s:   0,
      //start_time:       None,
      history:  vec![],
      ply:      0,
      state:    TxnState::new(state_cfg, TxnStateNodeData::new()),
      tree:     None,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      server:   ParallelMonteCarloSearchServer::new(
          state_cfg,
          num_workers, 1, worker_batch_capacity,
          ConvnetPolicyWorkerBuilder::new(tree_cfg, num_workers, 1, worker_batch_capacity),
      ),
    }
  }

  pub fn wait_ready(&mut self) {
    // XXX(20160308): Send a message (any message) to the server, and when
    // we get back a response, it is probably good to go.
    self.server.wait_ready();
  }

  pub fn ponder(&mut self) {
    let shared_tree = if self.tree.is_none() {
      let shared_tree = SharedTree::new(self.tree_cfg);
      self.tree = Some(shared_tree.clone());
      shared_tree
    } else {
      self.tree.as_ref().unwrap().clone()
    };

    let worker_cfg = SearchWorkerConfig{
      batch_cfg:    SearchWorkerBatchConfig::Fixed{num_batches: 1},
      tree_batch_size:      None,
      rollout_batch_size:   self.search_cfg.batch_size,
    };

    let mut search = ParallelMonteCarloSearch::new();
    let (_, stats) = search.join(
        worker_cfg,
        &mut self.server,
        self.our_stone.unwrap(),
        &self.state,
        shared_tree,
        &mut self.rng,
    );
  }

  pub fn search(&mut self, remaining_time_ms: usize) -> (Action, MonteCarloSearchResult) {
    let shared_tree = if self.tree.is_none() {
      let shared_tree = SharedTree::new(self.tree_cfg);
      self.tree = Some(shared_tree.clone());
      shared_tree
    } else {
      self.tree.as_ref().unwrap().clone()
    };

    // XXX(20160308): Use the time management strategy of [Huang, Coulom, Lin 2010].
    let allocated_time_ms =
        remaining_time_ms as f32
        / (60.0 + 0.0f32.max(60.0 - self.ply as f32));
    assert!(allocated_time_ms >= 0.0);
    println!("DEBUG: agent: search: time budget: {} ms", allocated_time_ms);

    let worker_cfg = SearchWorkerConfig{
      batch_cfg:    SearchWorkerBatchConfig::TimeLimit{
        // XXX(20160308): 200 ms corresponds to the expected RTT.
        budget_ms:  allocated_time_ms.floor() as usize,
        tol_ms:     200,
      },
      tree_batch_size:      None,
      rollout_batch_size:   self.search_cfg.batch_size,
    };

    let mut search = ParallelMonteCarloSearch::new();
    let (res, stats) = search.join(
        worker_cfg,
        &mut self.server,
        self.our_stone.unwrap(),
        &self.state,
        shared_tree,
        &mut self.rng,
    );

    (res.action, res)
  }

  pub fn step(&mut self, turn: Stone, action: Action) -> Result<(), ()> {
    let prev_state = self.state.clone();
    if self.state.try_action(turn, action).is_err() {
      self.state.undo();
      return Err(());
    } else {
      self.state.commit();
    }

    self.history.push((prev_state, turn, action));
    self.ply += 1;

    // Update the tree, if possible.
    let mut advance_success = false;
    if let Some(ref tree) = self.tree {
      if tree.try_advance(turn, action) {
        advance_success = true;
      }
    }
    if !advance_success {
      self.tree = None;
    }

    Ok(())
  }
}

pub struct ParallelSearchAsyncAgent;

impl AsyncAgent for ParallelSearchAsyncAgent {
  fn spawn_runloop(barrier: Arc<Barrier>, agent_in_rx: Receiver<AgentMsg>, agent_out_tx: Sender<AgentMsg>) -> JoinHandle<()> {
    spawn(move || {
      let mut agent = AgentImpl::new();
      agent.wait_ready();
      agent_out_tx.send(AgentMsg::Ready).unwrap();
      loop {
        match agent_in_rx.try_recv() {
          Ok(AgentMsg::RequestMatch{passive, opponent, our_stone, board_size, main_time_secs, byoyomi_time_secs}) => {
            println!("DEBUG: agent: request match");
            agent_out_tx.send(AgentMsg::AcceptMatch{
              passive:      passive,
              opponent:     opponent.clone(),
              our_stone:    our_stone,
              board_size:   board_size,
              main_time_secs:       main_time_secs,
              byoyomi_time_secs:    byoyomi_time_secs,
            }).unwrap();
          }

          Ok(AgentMsg::StartMatch{our_stone, board_size, main_time_secs, byoyomi_time_secs, ..}) => {
            println!("DEBUG: agent: start match");

            agent.our_stone = Some(our_stone);
            // FIXME(20160308): StartMatch should also contain the komi.
            //agent.komi = komi;
            agent.main_time_s = main_time_secs;
            agent.byoyomi_time_s = byoyomi_time_secs;

            agent.history.clear();
            agent.state.reset();
            agent.ply = 0;
            agent.tree = None;

            if Stone::Black == our_stone {
              agent.state_machine = AgentStateMachine::OurTurn;
              let remaining_time_ms = 1000 * agent.main_time_s as usize;
              println!("DEBUG: agent: remaining_time: {} ms", remaining_time_ms);
              let (action, res) = agent.search(remaining_time_ms);
              agent_out_tx.send(AgentMsg::SubmitAction{
                turn:     agent.our_stone.unwrap(),
                action:   action,
                // FIXME(20160316): just send current game result with every action
                // (including dead stones, live stones, territory, and est. outcome).
                dead_stones:  res.dead_stones,
                live_stones:  res.live_stones,
                territory:    res.territory,
                outcome:      res.outcome,
              }).unwrap();
              agent.state_machine = AgentStateMachine::OpponentTurn;
            } else {
              agent.state_machine = AgentStateMachine::OpponentTurn;
              agent.ponder();
            }
          }

          Ok(AgentMsg::RecvTime) => {
            println!("DEBUG: agent: recv time");
          }

          Ok(AgentMsg::RecvAction{turn, action, time_left_s, ..}) => {
            println!("DEBUG: agent: recv action");
            //assert!(agent.match_started);

            // Update the current state.
            agent.step(turn, action);

            if turn != agent.our_stone.unwrap() {
              agent.state_machine = AgentStateMachine::OurTurn;
              let our_action = if Action::Pass == action {
                Action::Pass
              } else {
                let remaining_time_ms = 1000 * time_left_s.unwrap() as usize;
                println!("DEBUG: agent: remaining_time: {} ms", remaining_time_ms);
                agent.search(remaining_time_ms)
              };
              agent_out_tx.send(AgentMsg::SubmitAction{
                turn:     agent.our_stone.unwrap(),
                action:   our_action,
              }).unwrap();
              agent.state_machine = AgentStateMachine::OpponentTurn;
            } else {
              agent.state_machine = AgentStateMachine::OpponentTurn;
              agent.ponder();
            }
          }

          Ok(AgentMsg::FinishMatch) => {
            println!("DEBUG: agent: finish match");
            break;
          }

          Ok(msg) => {
            // FIXME(20160308): other message types are ignored.
            println!("DEBUG: agent: unhandled message: {:?}", msg);
          }

          Err(TryRecvError::Empty) => {
            match agent.state_machine {
              AgentStateMachine::OurTurn => {
                // Already submitted our action; okay to do nothing.
                sleep_ms(100);
              }
              AgentStateMachine::OpponentTurn => {
                //println!("DEBUG: agent: pondering (3)...");
                agent.ponder();
                //println!("DEBUG: agent: finished pondering");
              }
              AgentStateMachine::Terminated => {
                break;
              }
              _ => {
                sleep_ms(100);
              }
            }
          }

          Err(_) => {
            println!("WARNING: agent: channel read error");
            //break;
          }
        }
      }
      barrier.wait();
    })
  }
}
