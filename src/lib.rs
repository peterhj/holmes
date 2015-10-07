#![feature(associated_consts)]

extern crate bit_set;
extern crate bit_vec;
extern crate bufstream;
extern crate byteorder;
extern crate rand;
extern crate time;

pub mod agent;
pub mod contains;
pub mod fastboard;
pub mod gtp;
pub mod gtp_board;
pub mod gtp_client;
pub mod policy;
pub mod random;
pub mod table;
pub mod tree;
pub mod util;
