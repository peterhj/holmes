use gtp_board::{Coord};

use std::str::{from_utf8};

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct Sgf {
  pub black_player: String,
  pub black_rank:   String,
  pub white_player: String,
  pub white_rank:   String,
  pub date:         String,
  pub place:        String,
  pub result:       String,
  pub board_dim:    i64,
  pub rules:        Option<String>,
  pub komi:         Option<f64>,
  pub moves:        Vec<(String, String)>,
  pub black_pos:    Vec<String>,
  pub white_pos:    Vec<String>,
}

impl Sgf {
  pub fn from_text(text: &[u8]) -> Sgf {
    let raw = parse_raw_sgf(text);
    Sgf::from_raw(&raw)
  }

  pub fn from_raw(raw: &RawSgf) -> Sgf {
    let mut black_player: String  = Default::default();
    let mut black_rank:   String  = Default::default();
    let mut white_player: String  = Default::default();
    let mut white_rank:   String  = Default::default();
    let mut date:         String  = Default::default();
    let mut place:        String  = Default::default();
    let mut result:       String  = Default::default();
    let mut board_dim:    i64     = Default::default();
    let mut rules:        Option<String> = Default::default();
    let mut komi:         Option<f64> = Default::default();
    let mut moves:        Vec<(String, String)> = Default::default();
    let mut black_pos:    Vec<String> = Default::default();
    let mut white_pos:    Vec<String> = Default::default();
    for property in raw.nodes[0].properties.iter() {
      match property {
        &Property::Root(ref root) => match root {
          &RootProperty::BoardSize(sz) => board_dim = sz,
          _ => {}
        },
        &Property::GameInfo(ref game_info) => match game_info {
          &GameInfoProperty::BlackPlayer(ref x) => black_player = x.clone(),
          &GameInfoProperty::BlackRank(ref x)   => black_rank = x.clone(),
          &GameInfoProperty::WhitePlayer(ref x) => white_player = x.clone(),
          &GameInfoProperty::WhiteRank(ref x)   => white_rank = x.clone(),
          &GameInfoProperty::Date(ref x)        => date = x.clone(),
          &GameInfoProperty::Place(ref x)       => place = x.clone(),
          &GameInfoProperty::Result(ref x)      => result = x.clone(),
          &GameInfoProperty::Rules(ref x)       => rules = Some(x.clone()),
          &GameInfoProperty::GoKomi(x)          => komi = x,
          _ => {}
        },
        &Property::Setup(ref setup) => match setup {
          &SetupProperty::AddBlack(ref x) => black_pos = x.iter().map(|c| c.to_string()).collect(),
          &SetupProperty::AddWhite(ref x) => white_pos = x.iter().map(|c| c.to_string()).collect(),
        },
        _ => {}
      }
    }
    for node in &raw.nodes[1 ..] {
      let (p, maybe_coord) = match &node.properties[0] {
        &Property::Move(MoveProperty::Black(maybe_coord)) => {
          ("B".to_string(), maybe_coord)
        }
        &Property::Move(MoveProperty::White(maybe_coord)) => { //moves.push(maybe_coord.map_or("pass".to_string(), |c| c.to_string())),
          ("W".to_string(), maybe_coord)
        }
        _ => { continue; }
      };
      let c = maybe_coord.map_or("Pass".to_string(), |c| c.to_string());
      moves.push((p, c));
    }
    Sgf{
      black_player: black_player,
      black_rank:   black_rank,
      white_player: white_player,
      white_rank:   white_rank,
      date:         date,
      place:        place,
      result:       result,
      board_dim:    board_dim,
      rules:        rules,
      komi:         komi,
      moves:        moves,
      black_pos:    black_pos,
      white_pos:    white_pos,
    }
  }
}

#[derive(Debug)]
pub enum Property {
  Move(MoveProperty),
  Setup(SetupProperty),
  Root(RootProperty),
  GameInfo(GameInfoProperty),
}

#[derive(Debug)]
pub enum MoveProperty {
  Black(Option<Coord>),
  White(Option<Coord>),
  //KO,
  //Number,
}

#[derive(Debug)]
pub enum SetupProperty {
  AddBlack(Vec<Coord>),
  AddWhite(Vec<Coord>),
}

#[derive(Debug)]
pub enum RootProperty {
  Game(i64),
  FileFormat(i64),
  CharSet(String),
  Application(String),
  Variations(i64),
  BoardSize(i64),
}

#[derive(Debug)]
pub enum GameInfoProperty {
  BlackPlayer(String),
  BlackRank(String),
  WhitePlayer(String),
  WhiteRank(String),
  Date(String),
  Place(String),
  Event(String),
  Round(String),
  Result(String),
  Rules(String),
  Overtime(String),
  TimeLimit(Option<f64>),
  Commentary(String),
  User(String),
  Source(String),
  GoKomi(Option<f64>),
  GoHandicap(Option<i64>),
  Unknown(Vec<u8>, Vec<u8>),
  Error(Vec<u8>),
}

#[derive(Debug)]
pub struct Node {
  pub properties: Vec<Property>,
}

#[derive(Debug)]
pub struct RawSgf {
  pub nodes: Vec<Node>,
}

pub fn parse_raw_sgf(text: &[u8]) -> RawSgf {
  let mut nodes = Vec::new();
  let mut ptr = 0;
  while ptr < text.len() {
    match text[ptr] {
      b'\n' | b'\r' | b' ' => {
        ptr += 1;
      }
      b'(' => {
        ptr += 1;
        ptr = parse_sequence(text, ptr, &mut nodes);
      }
      b')' => {
        break;
      }
      x => panic!("FATAL: unimpl: '{}'", x as char),
    }
  }
  RawSgf{nodes: nodes}
}

fn parse_sequence(text: &[u8], mut ptr: usize, nodes: &mut Vec<Node>) -> usize {
  loop {
    //println!("DEBUG: in parse_sequence loop: {}", ptr);
    match text[ptr] {
      b'\n' | b'\r' | b' ' => {
        ptr += 1;
      }
      b';' => {
        ptr += 1;
        ptr = parse_node(text, ptr, nodes);
      }
      b')' => {
        ptr += 1;
        return ptr;
      }
      x => panic!("FATAL: unimpl: '{}'", x as char),
    }
  }
}

fn parse_node(text: &[u8], mut ptr: usize, nodes: &mut Vec<Node>) -> usize {
  //println!("DEBUG: in parse_node: {}", ptr);
  let mut props = Vec::new();
  loop {
    match text[ptr] {
      b'\n' | b'\r' | b' ' =>{
        ptr += 1;
        continue;
      }
      b';' | b')' => {
        let node = Node{properties: props};
        nodes.push(node);
        return ptr;
      }
      _ => {
        ptr = parse_property(text, ptr, &mut props);
      }
    }
  }
}

fn parse_property(text: &[u8], mut ptr: usize, props: &mut Vec<Property>) -> usize {
  //println!("DEBUG: in parse_property: {}", ptr);
  if text[ptr ..].len() >= 3 {
    let mut value: Vec<u8> = Vec::new();
    let prop = match (text[ptr], text[ptr + 1], text[ptr + 2]) {
      (b'B', b'[', _) => {
        ptr += 1;
        ptr = read_property_value(text, ptr, &mut value);
        if value.len() > 0 && &value[0 .. 2] != b"tt" {
          Property::Move(MoveProperty::Black(Some(Coord::from_sgf(&value))))
        } else {
          Property::Move(MoveProperty::Black(None))
        }
      }
      (b'W', b'[', _) => {
        ptr += 1;
        ptr = read_property_value(text, ptr, &mut value);
        if value.len() > 0 && &value[0 .. 2] != b"tt" {
          Property::Move(MoveProperty::White(Some(Coord::from_sgf(&value))))
        } else {
          Property::Move(MoveProperty::White(None))
        }
      }
      (b'G', b'M', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::Root(RootProperty::Game(s.parse().unwrap()))
      }
      (b'F', b'F', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::Root(RootProperty::FileFormat(s.parse().unwrap()))
      }
      (b'S', b'Z', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::Root(RootProperty::BoardSize(s.parse().unwrap()))
      }
      (b'C', b'A', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::Root(RootProperty::CharSet(s))
      }
      (b'S', b'T', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::Root(RootProperty::Variations(s.parse().unwrap()))
      }
      (b'A', b'P', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::Root(RootProperty::Application(s))
      }
      (b'P', b'W', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::WhitePlayer(s))
      }
      (b'W', b'R', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::WhiteRank(s))
      }
      (b'P', b'B', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::BlackPlayer(s))
      }
      (b'B', b'R', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::BlackRank(s))
      }
      (b'D', b'T', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Date(s))
      }
      (b'P', b'C', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Place(s))
      }
      (b'E', b'V', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Event(s))
      }
      (b'R', b'O', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Round(s))
      }
      (b'R', b'E', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Result(s))
      }
      (b'R', b'U', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Rules(s))
      }
      (b'O', b'T', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Overtime(s))
      }
      (b'T', b'M', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::TimeLimit(s.parse().ok()))
      }
      (b'G', b'C', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Commentary(s))
      }
      (b'U', b'S', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::User(s))
      }
      (b'S', b'O', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::Source(s))
      }
      (b'K', b'M', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::GoKomi(s.parse().ok()))
      }
      (b'H', b'A', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        let s = match String::from_utf8(value.clone()) {
          Ok(s) => s,
          Err(_) => {
            println!("WARNING: encoding error: value '{:?}'", String::from_utf8_lossy(&value));
            return ptr;
          }
        };
        Property::GameInfo(GameInfoProperty::GoHandicap(s.parse().ok()))
      }
      (b'A', b'B', b'[') => {
        ptr += 2;
        let mut values = Vec::new();
        while text[ptr] == b'[' {
          value.clear();
          ptr = read_property_value(text, ptr, &mut value);
          values.push(value.clone());
          loop {
            match text[ptr] {
              b'\n' | b'\r' | b' ' => {
                ptr += 1;
                continue;
              }
              _ => {
                break;
              }
            }
          }
        }
        Property::Setup(SetupProperty::AddBlack(
            values.iter()
              .map(|v| Coord::from_sgf(&v))
              .collect()
        ))
      }
      (b'A', b'W', b'[') => {
        ptr += 2;
        let mut values = Vec::new();
        while text[ptr] == b'[' {
          value.clear();
          ptr = read_property_value(text, ptr, &mut value);
          values.push(value.clone());
          loop {
            match text[ptr] {
              b'\n' | b'\r' | b' ' => {
                ptr += 1;
                continue;
              }
              _ => {
                break;
              }
            }
          }
        }
        Property::Setup(SetupProperty::AddWhite(
            values.iter()
              .map(|v| Coord::from_sgf(&v))
              .collect()
        ))
      }
      (x0, x1, x2) => {
        if x2 == b'[' {
          //panic!("FATAL: unimpl property: '{}{}{}'", x.0 as char, x.1 as char, x.2 as char)
          //println!("DEBUG: unimpl property: '{}{}'", x0 as char, x1 as char);
          ptr += 2;
          ptr = read_property_value(text, ptr, &mut value);
          Property::GameInfo(GameInfoProperty::Unknown(vec![x0, x1], value.clone()))
        } else if &text[ptr .. ptr + 10] == b"MULTIGOGM[" {
          ptr += 9;
          ptr = read_property_value(text, ptr, &mut value);
          Property::GameInfo(GameInfoProperty::Unknown(text[ptr .. ptr + 10].to_vec(), value.clone()))
        } else {
          //panic!("FATAL: unhandled property: '{}'", from_utf8(&text[ptr .. ]).unwrap());
          println!("DEBUG: unhandled property");
          return ptr;
        }
      }
    };
    props.push(prop);
  }
  return ptr;
}

fn read_property_value(text: &[u8], mut ptr: usize, value: &mut Vec<u8>) -> usize {
  //println!("DEBUG: in read_property_value: {}", ptr);
  if b'[' != text[ptr] {
    panic!("property syntax error: '{}'", from_utf8(text).unwrap());
  }
  ptr += 1;
  while text[ptr] != b']' {
    value.push(text[ptr]);
    ptr += 1;
  }
  ptr += 1;
  return ptr;
}
