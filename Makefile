.PHONY: all debug clean test

all:
	cargo build --release

debug:
	cargo build

clean:
	cargo clean

test: test-sgf

test-sgf:
	cargo test --release --test sgf
