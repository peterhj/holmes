use gtp::{
  Entity, GtpId, GtpMessage, ClientState,
  handle_client_stream,
  create_command_string,
  parse_response_string,
};
use gtp::Entity::{
  StringEntity, IntEntity, FloatEntity, BooleanEntity,
  VertexEntity, ColorEntity, MoveEntity,
  MultilineListEntity,
  ErrorEntity,
};
use gtp_board::{Player, Coord, Vertex, Move};

use std::net::{TcpListener};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread::{spawn};

pub struct GtpRefereedController {
  black_listener:   TcpListener,
  white_listener:   TcpListener,
  referee_listener: TcpListener,
  komi:             f32,
  allow_pass:       bool,
  id: GtpId,
}

impl GtpRefereedController {
  pub fn new(b_port: u16, w_port: u16, r_port: u16, komi: f32, allow_pass: bool) -> GtpRefereedController {
    //sleep(Duration::seconds(3));
    let black_listener = TcpListener::bind(&format!("127.0.0.1:{}", b_port) as &str)
      .ok().expect("FATAL: server: failed to bind address (black)");
    let white_listener = TcpListener::bind(&format!("127.0.0.1:{}", w_port) as &str)
      .ok().expect("FATAL: server: failed to bind address (white)");
    let referee_listener = TcpListener::bind(&format!("127.0.0.1:{}", r_port) as &str)
      .ok().expect("FATAL: server: failed to bind address (referee)");
    GtpRefereedController{
      black_listener:   black_listener,
      white_listener:   white_listener,
      referee_listener: referee_listener,
      komi:             komi,
      allow_pass:       allow_pass,
      id: GtpId::new(),
    }
  }

  fn setup_client(&mut self, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) {
    // Perform GTP administrative functions.
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Wakeup => {
        println!("DEBUG: server: acking wakeup to client");
        ctl_to_client_tx.send(GtpMessage::AckWakeup(self.id.increment_by(4)))
          .unwrap();
      },
      _ => (),
    }

    // Clear the board.
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"clear_board".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }

    // Set the board size.
    let cmd_str = create_command_string(self.id.increment(), &[
      // FIXME: fixed sample board size for now.
      StringEntity(b"boardsize".to_vec()),
      IntEntity(19),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }

    // Set the komi level.
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"komi".to_vec()),
      FloatEntity(self.komi),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }

    // Set the time settings (KGS extension).
    let cmd_str = create_command_string(self.id.increment(), &[
      // FIXME: fixed sample time settings for now.
      //StringEntity(b"kgs-time_settings".to_vec()),
      // Variant ("none", "absolute", "byoyomi", or "canadian").
      //StringEntity(b"byoyomi".to_vec()),
      StringEntity(b"time_settings".to_vec()),
      // FIXME(20151020): "blitz" settings for fast testing.
      // Main time in seconds.
      //IntEntity(300),
      IntEntity(5400),
      // Byo-yomi time in seconds.
      //IntEntity(40),
      IntEntity(30),
      // Byo-yomi stones (Canadian) or periods (Japanese).
      IntEntity(1), // 5
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }
  }

  fn show_board(&mut self, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) {
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"showboard".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        // TODO
      },
      _ => (),
    }
  }

  fn play_client(&mut self, player: Player, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) -> Move {
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"genmove".to_vec()),
      StringEntity(player.to_bytestring()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    // FIXME(20151002): not checking for time limit!
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: failed to recv from client")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        match Entity::parse_vertex_or_resign(&res.lines[0][0]) {
          VertexEntity(action) => Move::new(player, action),
          StringEntity(_) => Move::new(player, Vertex::Resign),
          _ => unreachable!(),
        }
      },
      _ => unreachable!(),
    }
  }

  fn update_client(&mut self, current_move: Move, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) -> Result<(), ()> {
    match current_move.vertex {
      Vertex::Pass | Vertex::Play(_) => {
        let cmd_str = create_command_string(self.id.increment(), &[
          StringEntity(b"play".to_vec()),
          MoveEntity(current_move.player, current_move.vertex),
        ]);
        ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
          .unwrap();
        match client_to_ctl_rx.recv()
          .ok().expect("FATAL: TODO")
        {
          GtpMessage::Response(res_str) => {
            let res = parse_response_string(&res_str);
            if res.has_error { // TODO: check for string b"illegal move"?
              return Err(());
            }
          },
          _ => (),
        }
      },
      Vertex::Resign => (), // FIXME
    }
    Ok(())
  }

  fn undo_client(&mut self, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) {
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"undo".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv() {
      _ => (),
    }
  }

  fn shutdown_client(&mut self, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) {
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"final_score".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv() {
      _ => (),
    }
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"quit".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv() {
      _ => (),
    }
  }

  pub fn runloop(&mut self) {
    loop {
      // TODO: We start the controller with game parameters.
      // Then the two engines connect to the respective ports, and the game is
      // considered to begin (after boilerplate GTP commands).
      let black_listener = self.black_listener.try_clone()
        .ok().expect("failed to clone listener!");
      let white_listener = self.white_listener.try_clone()
        .ok().expect("failed to clone listener!");
      let referee_listener = self.referee_listener.try_clone()
        .ok().expect("failed to clone listener!");
      let (black_to_ctl_tx, black_to_ctl_rx) = channel::<GtpMessage>();
      let (white_to_ctl_tx, white_to_ctl_rx) = channel::<GtpMessage>();
      let (referee_to_ctl_tx, referee_to_ctl_rx) = channel::<GtpMessage>();
      let (ctl_to_black_tx, ctl_to_black_rx) = channel::<GtpMessage>();
      let (ctl_to_white_tx, ctl_to_white_rx) = channel::<GtpMessage>();
      let (ctl_to_referee_tx, ctl_to_referee_rx) = channel::<GtpMessage>();
      spawn(move || {
        for stream in referee_listener.incoming() {
          match stream {
            Ok(stream) => {
              handle_client_stream(referee_to_ctl_tx, ctl_to_referee_rx, stream);
              break;
            },
            Err(_) => (),
          }
        }
      });
      spawn(move || {
        for stream in black_listener.incoming() {
          match stream {
            Ok(stream) => {
              handle_client_stream(black_to_ctl_tx, ctl_to_black_rx, stream);
              break;
            },
            Err(_) => (),
          }
        }
      });
      spawn(move || {
        for stream in white_listener.incoming() {
          match stream {
            Ok(stream) => {
              handle_client_stream(white_to_ctl_tx, ctl_to_white_rx, stream);
              break;
            },
            Err(_) => (),
          }
        }
      });

      self.setup_client(&referee_to_ctl_rx, &ctl_to_referee_tx);
      self.setup_client(&black_to_ctl_rx, &ctl_to_black_tx);
      self.setup_client(&white_to_ctl_rx, &ctl_to_white_tx);

      let mut black_state = ClientState::new();
      let mut white_state = ClientState::new();

      let mut current_player = Player::Black;
      loop {
        let result = match current_player {
          Player::Black => {
            let referee_move = self.play_client(Player::Black, &referee_to_ctl_rx, &ctl_to_referee_tx);
            if Vertex::Resign != referee_move.vertex {
              self.undo_client(&referee_to_ctl_rx, &ctl_to_referee_tx);
            }
            let current_move = if self.allow_pass {
              match referee_move.vertex {
                Vertex::Pass => {
                  self.update_client(referee_move, &black_to_ctl_rx, &ctl_to_black_tx);
                  referee_move
                }
                Vertex::Resign => {
                  referee_move
                }
                _ => {
                  self.play_client(Player::Black, &black_to_ctl_rx, &ctl_to_black_tx)
                }
              }
            } else {
              self.play_client(Player::Black, &black_to_ctl_rx, &ctl_to_black_tx)
            };
            let result = self.update_client(current_move, &white_to_ctl_rx, &ctl_to_white_tx);
            if Vertex::Resign != current_move.vertex {
              self.update_client(current_move, &referee_to_ctl_rx, &ctl_to_referee_tx);
            }
            self.show_board(&referee_to_ctl_rx, &ctl_to_referee_tx);
            black_state.previous_move = Some(current_move);
            if let Vertex::Resign = current_move.vertex {
              break;
            }
            result
          },
          Player::White => {
            let referee_move = self.play_client(Player::White, &referee_to_ctl_rx, &ctl_to_referee_tx);
            if Vertex::Resign != referee_move.vertex {
              self.undo_client(&referee_to_ctl_rx, &ctl_to_referee_tx);
            }
            let current_move = if self.allow_pass {
              match referee_move.vertex {
                Vertex::Pass => {
                  self.update_client(referee_move, &white_to_ctl_rx, &ctl_to_white_tx);
                  referee_move
                }
                Vertex::Resign => {
                  referee_move
                }
                _ => {
                  self.play_client(Player::White, &white_to_ctl_rx, &ctl_to_white_tx)
                }
              }
            } else {
              self.play_client(Player::White, &white_to_ctl_rx, &ctl_to_white_tx)
            };
            let result = self.update_client(current_move, &black_to_ctl_rx, &ctl_to_black_tx);
            if Vertex::Resign != current_move.vertex {
              self.update_client(current_move, &referee_to_ctl_rx, &ctl_to_referee_tx);
            }
            self.show_board(&referee_to_ctl_rx, &ctl_to_referee_tx);
            white_state.previous_move = Some(current_move);
            if let Vertex::Resign = current_move.vertex {
              break;
            }
            result
          },
        };
        // TODO: if current_player's opponent claims current_move is illegal,
        // then what?
        match result {
          Ok(_) => (),
          Err(_) => {
            panic!("FATAL: player {} called out an illegal move!",
              current_player.opponent().to_string());
          },
        }
        if black_state.previous_move.is_some() && white_state.previous_move.is_some() {
          match black_state.previous_move.unwrap().vertex {
            Vertex::Pass => {
              match white_state.previous_move.unwrap().vertex {
                Vertex::Pass => {
                  // FIXME(20150104): We get a "receiving on a closed channel"
                  // panic message on termination.
                  break;
                },
                _ => (),
              }
            },
            _ => (),
          }
        }
        current_player = current_player.opponent();
      }
      self.shutdown_client(&referee_to_ctl_rx, &ctl_to_referee_tx);
      self.shutdown_client(&black_to_ctl_rx, &ctl_to_black_tx);
      self.shutdown_client(&white_to_ctl_rx, &ctl_to_white_tx);

      break; // FIXME: stop after one game.
    }
  }
}
