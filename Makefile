.PHONY: all debug clean test test-txn

all:
	cargo build --release

debug:
	cargo build

clean:
	cargo clean

test: test-txn

test-txn:
	cargo test --release --test txnstate
