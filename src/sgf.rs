use gtp_board::{Coord};

use std::str::{from_utf8};

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
  Game(i32),
  FileFormat(i32),
  CharSet(String),
  Application(String),
  Variations(i32),
  BoardSize(i32),
}

#[derive(Debug)]
pub enum GameInfoProperty {
  BlackPlayer(String),
  BlackRank(String),
  WhitePlayer(String),
  WhiteRank(String),
  Date(String),
  Place(String),
  Result(String),
  Rules(String),
  Overtime(String),
  TimeLimit(f32),
  GoKomi(f32),
  GoHandicap(i32),
}

#[derive(Debug)]
pub struct Node {
  pub properties: Vec<Property>,
}

#[derive(Debug)]
pub struct Sgf {
  pub nodes: Vec<Node>,
}

pub fn parse_sgf(text: &[u8]) -> Sgf {
  let mut nodes = Vec::new();
  let mut ptr = 0;
  while ptr < text.len() {
    match text[ptr] {
      b'(' => {
        ptr += 1;
        ptr = parse_sequence(text, ptr, &mut nodes);
      }
      b')' => {
        break;
      }
      b'\n' | b'\r' | b' ' => {
        ptr += 1;
      }
      x => panic!("FATAL: unimpl: '{}'", x as char),
    }
  }
  Sgf{nodes: nodes}
}

fn parse_sequence(text: &[u8], mut ptr: usize, nodes: &mut Vec<Node>) -> usize {
  loop {
    //println!("DEBUG: in parse_sequence loop: {}", ptr);
    match text[ptr] {
      b')' => {
        ptr += 1;
        return ptr;
      }
      b';' => {
        ptr += 1;
        ptr = parse_node(text, ptr, nodes);
      }
      _ => {}
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
        if value.len() > 0 {
          Property::Move(MoveProperty::Black(Some(Coord::from_sgf(&value))))
        } else {
          Property::Move(MoveProperty::Black(None))
        }
      }
      (b'W', b'[', _) => {
        ptr += 1;
        ptr = read_property_value(text, ptr, &mut value);
        if value.len() > 0 {
          Property::Move(MoveProperty::White(Some(Coord::from_sgf(&value))))
        } else {
          Property::Move(MoveProperty::White(None))
        }
      }
      (b'G', b'M', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::Root(RootProperty::Game(String::from_utf8(value.clone()).unwrap().parse().unwrap()))
      }
      (b'F', b'F', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::Root(RootProperty::FileFormat(String::from_utf8(value.clone()).unwrap().parse().unwrap()))
      }
      (b'S', b'Z', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::Root(RootProperty::BoardSize(String::from_utf8(value.clone()).unwrap().parse().unwrap()))
      }
      (b'P', b'W', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::WhitePlayer(String::from_utf8(value.clone()).unwrap()))
      }
      (b'W', b'R', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::WhiteRank(String::from_utf8(value.clone()).unwrap()))
      }
      (b'P', b'B', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::BlackPlayer(String::from_utf8(value.clone()).unwrap()))
      }
      (b'B', b'R', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::BlackRank(String::from_utf8(value.clone()).unwrap()))
      }
      (b'D', b'T', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::Date(String::from_utf8(value.clone()).unwrap()))
      }
      (b'P', b'C', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::Place(String::from_utf8(value.clone()).unwrap()))
      }
      (b'K', b'M', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::GoKomi(String::from_utf8(value.clone()).unwrap().parse().unwrap()))
      }
      (b'R', b'E', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::Result(String::from_utf8(value.clone()).unwrap()))
      }
      (b'R', b'U', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::Rules(String::from_utf8(value.clone()).unwrap()))
      }
      (b'O', b'T', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::Overtime(String::from_utf8(value.clone()).unwrap()))
      }
      (b'C', b'A', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::Root(RootProperty::CharSet(String::from_utf8(value.clone()).unwrap()))
      }
      (b'S', b'T', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::Root(RootProperty::Variations(String::from_utf8(value.clone()).unwrap().parse().unwrap()))
      }
      (b'A', b'P', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::Root(RootProperty::Application(String::from_utf8(value.clone()).unwrap()))
      }
      (b'T', b'M', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::TimeLimit(String::from_utf8(value.clone()).unwrap().parse().unwrap()))
      }
      (b'H', b'A', b'[') => {
        ptr += 2;
        ptr = read_property_value(text, ptr, &mut value);
        Property::GameInfo(GameInfoProperty::GoHandicap(String::from_utf8(value.clone()).unwrap().parse().unwrap()))
      }
      (b'A', b'B', b'[') => {
        //println!("DEBUG: in 'AB'");
        ptr += 2;
        let mut values = Vec::new();
        while text[ptr] == b'[' {
          //println!("DEBUG: in 'AB': reading property value");
          value.clear();
          ptr = read_property_value(text, ptr, &mut value);
          //values.push(String::from_utf8(value.clone()).unwrap());
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
        //println!("DEBUG: done 'AB'");
        Property::Setup(SetupProperty::AddBlack(
            values.iter()
              .map(|v| Coord::from_sgf(&v))
              .collect()
        ))
      }
      x => panic!("FATAL: unimpl: '{}{}{}'", x.0 as char, x.1 as char, x.2 as char),
    };
    props.push(prop);
  }
  return ptr;
}

fn read_property_value(text: &[u8], mut ptr: usize, value: &mut Vec<u8>) -> usize {
  //println!("DEBUG: in read_property_value: {}", ptr);
  assert_eq!(b'[', text[ptr]);
  ptr += 1;
  while text[ptr] != b']' {
    value.push(text[ptr]);
    ptr += 1;
  }
  ptr += 1;
  return ptr;
}
