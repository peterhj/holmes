.PHONY: all debug clean test \
	test-omega test-txnstate

all:
	cargo build --release

debug:
	cargo build

clean:
	cargo clean

test: test-omega test-txnstate
test-omega:
	cargo test --release --test omega -- --nocapture
test-txnstate:
	cargo test --release --test txnstate
