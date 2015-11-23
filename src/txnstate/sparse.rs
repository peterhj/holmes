use board::{Board, Stone, Point, Action};
use txnstate::{TxnStateData, TxnState, TxnPosition, TxnChainsList};
use txnstate::extras::{TxnStateLegalityData, TxnStateNodeData};

use vec_map::{VecMap};

// XXX: The lower part (0x0000 thru 0xffff inclusive) of the gammas table is for
// 3x3 shape patterns.
thread_local!(static SPARSE_GAMMAS_LOADED: RefCell<bool> = RefCell::new(false));
thread_local!(static SPARSE_GAMMAS: RefCell<VecMap<f32>> = RefCell::new(VecMap::with_capacity(0x10100)));

/*#[derive(Clone)]
pub struct TxnStateSparseFeaturesTrainingData {
  features: Vec<(Point, Vec<u32>)>,
}

impl TxnStateSparseFeaturesTrainingData {
  pub fn from_state(state: &TxnState<TxnStateNodeData>) -> TxnStateSparseFeaturesTrainingData {
    let turn = state.current_turn();
    let mut legal_points = vec![];
    state.get_data().legality.fill_legal_points(turn, &mut legal_points);

    let mut features = Vec::with_capacity(legal_points.len());

    // Shape pattern features.
    for &point in legal_points.iter() {
      let pattern = state.current_pat3x3(point).to_invariant();
      let pattern_idx = pattern.idx();

      features.push((point, vec![pattern_idx]));
    }

    TxnStateSparseFeaturesTrainingData{
      features: features,
    }
  }
}*/

#[derive(Clone)]
pub struct TxnStateSparseFeaturesData {
  //features: Vec<(Point, f32, Vec<u32>)>,
  legality:       TxnStateLegalityData,
  legal_gammas:   Vec<VecMap<f32>>,
  shape_features: Vec<VecMap<u32>>,
}

impl TxnStateSparseFeaturesData {
  pub fn from_state(state: &TxnState<TxnStateNodeData>) -> TxnStateSparseFeaturesData {
    /*SPARSE_GAMMAS_LOADED.with(|loaded| {
      if !*loaded.borrow() {
        SPARSE_GAMMAS.with(|gammas| {
          // TODO(20151117): Load gammas from file.
        });
        *loaded.borrow_mut() = true;
      }
    });

    let turn = state.current_turn();
    let mut legal_points = vec![];
    state.get_data().legality.fill_legal_points(turn, &mut legal_points);

    let mut features = Vec::with_capacity(legal_points.len());

    SPARSE_GAMMAS.with(|gammas| {
      let gammas = gammas.borrow();

      // Shape pattern features.
      for &point in legal_points.iter() {
        let pattern = state.current_pat3x3(point).to_invariant();
        let pattern_idx = pattern.idx();
        let mut gamma = gammas[pattern_idx as usize];

        features.push((point, gamma, vec![pattern_idx]));
      }
    });

    TxnStateSparseFeaturesData{
      features: features,
    }*/

    let mut data = TxnStateSparseFeaturesData{
      legality: state.get_data().legality.clone(),
      legal_gammas: vec![
        VecMap::with_capacity(Board::SIZE),
        VecMap::with_capacity(Board::SIZE),
      ],
      shape_features: vec![
        VecMap::with_capacity(Board::SIZE),
        VecMap::with_capacity(Board::SIZE),
      ],
    };

    // Shape features.
    for &turn in [Stone::Black, Stone::White].iter() {
      state.get_data().legality.for_each_legal_point(turn, |point| {
        let pattern = state.current_pat3x3(point).to_invariant();
        let pattern_idx = pattern.idx();
        //features.push((point, gamma, vec![pattern_idx]));
      });
    }

    // TODO(20151119)
    unimplemented!();
  }
}

impl TxnStateData for TxnStateSparseFeaturesData {
  fn reset(&mut self) {
    // TODO(20151117)
    unimplemented!();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    // TODO(20151117)
    unimplemented!();
  }
}
