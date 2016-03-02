.PHONY: all debug clean test \
	test-bfilter test-omega test-txnstate

all:
	cargo build --release

debug:
	cargo build

clean:
	cargo clean

test: test-bfilter test-omega test-txnstate
test-bfilter:
	cargo test --release --test bfilter -- --nocapture
test-omega:
	cargo test --release --test omega -- --nocapture
test-txnstate:
	cargo test --release --test txnstate
