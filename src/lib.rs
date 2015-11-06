#![feature(associated_consts)]
#![feature(clone_from_slice)]
#![feature(drain)]
#![feature(slice_bytes)]
#![feature(wrapping)]

extern crate array;
extern crate async_cuda;
extern crate rembrandt;
extern crate statistics_avx2;

extern crate bit_set;
extern crate bit_vec;
extern crate bufstream;
extern crate byteorder;
extern crate rand;
extern crate rustc_serialize;
extern crate time;

pub mod agent;
pub mod book;
pub mod contains;
pub mod fastboard;
//pub mod fasttree;
pub mod game;
pub mod gtp;
pub mod gtp_board;
pub mod gtp_client;
pub mod mctree;
pub mod policy;
pub mod random;
//pub mod search;
pub mod sgf;
pub mod table;
pub mod txnboard;
pub mod util;
