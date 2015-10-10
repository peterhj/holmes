#![feature(associated_consts)]

extern crate statistics_avx2;

extern crate bit_set;
extern crate bit_vec;
extern crate bufstream;
extern crate byteorder;
extern crate rand;
extern crate time;

pub mod agent;
pub mod book;
pub mod contains;
pub mod fastboard;
pub mod fasttree;
pub mod gtp;
pub mod gtp_board;
pub mod gtp_client;
pub mod policy;
pub mod random;
pub mod sgf;
pub mod table;
//pub mod tree;
pub mod util;
