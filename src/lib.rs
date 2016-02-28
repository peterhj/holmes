#![feature(associated_consts)]
#![feature(associated_type_defaults)]
#![feature(clone_from_slice)]
#![feature(slice_bytes)]
//#![feature(wrapping)]

extern crate array;
extern crate array_new;
extern crate array_cuda;
extern crate async_cuda;
extern crate cuda;
extern crate float;
extern crate gsl;
extern crate episodb;
extern crate rembrandt;
extern crate rng;
//extern crate statistics_avx2;

extern crate bit_set;
extern crate bit_vec;
extern crate bufstream;
extern crate byteorder;
extern crate libc;
extern crate rand;
extern crate rustc_serialize;
extern crate threadpool;
extern crate time;
extern crate toml;
extern crate vec_map;

pub mod agents;
pub mod array_util;
pub mod board;
//pub mod book;
pub mod contains;
pub mod convnet;
pub mod convnet_new;
pub mod data;
pub mod discrete;
//pub mod fastboard;
pub mod fix;
//pub mod game;
pub mod gtp;
pub mod gtp_board;
pub mod gtp_client;
pub mod gtp_ctrl;
pub mod hyper;
pub mod pattern;
pub mod pg;
pub mod random;
pub mod search;
pub mod sgf;
//pub mod table;
pub mod txnstate;
pub mod util;
pub mod worker;
