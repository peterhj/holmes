# holmes

This is the source for `holmes`, a computer Go program that participated in the
[9th UEC Cup](http://www.computer-go.jp/uec/public_html/past/2015/eng/index.shtml)
(March 19-20, 2016), and won the award for Most Original Program.  `holmes` is
written in Rust, using CUDA (via Rust bindings) to run convolutional neural net
policies.  Unlike most computer Go programs that base their MCTS rollouts around
some variant of UCB, `holmes` uses batched Thompson sampling along with a batch
rollout policy that is essentially a small convolutional neural net.

This code is not being updated; the last real commit date was from April 2016,
and it uses Rust version 1.5 (nightly).  There are also some researchy bits
which are undocumented.  YMMV.
